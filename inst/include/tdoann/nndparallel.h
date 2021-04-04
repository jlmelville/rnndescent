// BSD 2-Clause License
//
// Copyright 2020 James Melville
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// OF SUCH DAMAGE.

#ifndef TDOANN_NNDPARALLEL_H
#define TDOANN_NNDPARALLEL_H

#include <mutex>
#include <vector>

#include "graphupdate.h"
#include "heap.h"
#include "nndescent.h"
#include "parallel.h"
#include "progress.h"

namespace tdoann {

template <typename Distance> struct LockingHeapAdder {
  using Idx = typename Distance::Index;
  using Out = typename Distance::Output;

  static const constexpr std::size_t n_mutexes = 10;
  std::mutex mutexes[n_mutexes];

  LockingHeapAdder() {}
  LockingHeapAdder(LockingHeapAdder const &) = delete;
  LockingHeapAdder &operator=(LockingHeapAdder const &) = delete;

  void add(NNHeap<Out, Idx> &nbrs, Idx i, Idx idx, Out d) {
    {
      std::lock_guard<std::mutex> guard(mutexes[i % n_mutexes]);
      nbrs.checked_push(i, d, idx);
    }
    if (i != idx) {
      std::lock_guard<std::mutex> guard(mutexes[idx % n_mutexes]);
      nbrs.checked_push(idx, d, i);
    }
  }
};

template <typename ParallelRand, typename Distance>
void build_candidates(
    const NNDHeap<typename Distance::Output, typename Distance::Index>
        &current_graph,
    NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    NNHeap<typename Distance::Output, typename Distance::Index> &old_nbrs,
    ParallelRand &parallel_rand, LockingHeapAdder<Distance> &heap_adder,
    std::size_t begin, std::size_t end) {

  const std::size_t n_nbrs = current_graph.n_nbrs;
  auto rand = parallel_rand.get_rand(end);

  for (auto i = begin; i < end; i++) {
    std::size_t innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      std::size_t ij = innbrs + j;
      std::size_t idx = current_graph.idx[ij];
      char isn = current_graph.flags[ij];
      auto &nbrs = isn == 1 ? new_nbrs : old_nbrs;
      if (idx == nbrs.npos()) {
        continue;
      }
      auto d = rand.unif();
      heap_adder.add(nbrs, i, idx, d);
    }
  }
}

template <typename Parallel, typename Distance, typename ParallelRand>
void build_candidates(
    const NNDHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    NNHeap<typename Distance::Output, typename Distance::Index> &old_nbrs,
    ParallelRand &parallel_rand, LockingHeapAdder<Distance> &heap_adder,
    std::size_t n_threads, std::size_t grain_size) {
  parallel_rand.reseed();
  auto worker = [&](std::size_t begin, std::size_t end) {
    build_candidates(nn_heap, new_nbrs, old_nbrs, parallel_rand, heap_adder,
                     begin, end);
  };
  Parallel::parallel_for(0, nn_heap.n_points, worker, n_threads, grain_size);
}

template <typename Parallel, typename Distance>
void flag_new_candidates(
    NNDHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    const NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    std::size_t n_threads, std::size_t grain_size) {
  auto worker = [&](std::size_t begin, std::size_t end) {
    flag_retained_new_candidates(nn_heap, new_nbrs, begin, end);
  };
  Parallel::parallel_for(0, nn_heap.n_points, worker, n_threads, grain_size);
}

template <typename Distance, typename GraphUpdater>
void local_join(
    GraphUpdater &graph_updater,
    const NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    const NNHeap<typename Distance::Output, typename Distance::Index> &old_nbrs,
    std::size_t max_candidates, std::size_t begin, std::size_t end) {
  for (auto i = begin; i < end; i++) {
    std::size_t imaxc = i * max_candidates;
    for (std::size_t j = 0; j < max_candidates; j++) {
      std::size_t p = new_nbrs.idx[imaxc + j];
      if (p == new_nbrs.npos()) {
        continue;
      }
      for (std::size_t k = j; k < max_candidates; k++) {
        std::size_t q = new_nbrs.idx[imaxc + k];
        if (q == new_nbrs.npos()) {
          continue;
        }
        graph_updater.generate(p, q, i);
      }

      for (std::size_t k = 0; k < max_candidates; k++) {
        std::size_t q = old_nbrs.idx[imaxc + k];
        if (q == old_nbrs.npos()) {
          continue;
        }
        graph_updater.generate(p, q, i);
      }
    }
  }
}

template <typename Parallel, typename Distance, typename GraphUpdater,
          typename Progress>
auto local_join(
    GraphUpdater &graph_updater,
    const NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    const NNHeap<typename Distance::Output, typename Distance::Index> &old_nbrs,
    Progress &progress, std::size_t block_size, std::size_t n_threads,
    std::size_t grain_size) -> std::size_t {
  std::size_t c = 0;
  auto local_join_worker = [&](std::size_t begin, std::size_t end) {
    local_join<Distance, decltype(graph_updater)>(
        graph_updater, new_nbrs, old_nbrs, new_nbrs.n_nbrs, begin, end);
  };
  auto after_local_join = [&](std::size_t, std::size_t) {
    c += graph_updater.apply();
  };
  batch_parallel_for<Parallel>(local_join_worker, after_local_join, progress,
                               graph_updater.current_graph.n_points, block_size,
                               n_threads, grain_size);
  return c;
}

template <typename Parallel, typename ParallelRand,
          template <typename> class GraphUpdater, typename Distance,
          typename Progress>
void nnd_build(GraphUpdater<Distance> &graph_updater,
               std::size_t max_candidates, std::size_t n_iters, double delta,
               Progress &progress, ParallelRand &parallel_rand,
               std::size_t block_size = 16384, std::size_t n_threads = 0,
               std::size_t grain_size = 1) {

  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  auto &nn_heap = graph_updater.current_graph;
  const std::size_t n_points = nn_heap.n_points;
  const double tol = delta * nn_heap.n_nbrs * n_points;

  LockingHeapAdder<Distance> heap_adder;

  for (std::size_t n = 0; n < n_iters; n++) {
    NNHeap<DistOut, Idx> new_nbrs(n_points, max_candidates);
    decltype(new_nbrs) old_nbrs(n_points, max_candidates);

    build_candidates<Parallel, Distance>(nn_heap, new_nbrs, old_nbrs,
                                         parallel_rand, heap_adder, n_threads,
                                         grain_size);

    // mark any neighbor in the current graph that was retained in the new
    // candidates as true
    flag_new_candidates<Parallel, Distance>(nn_heap, new_nbrs, n_threads,
                                            grain_size);

    std::size_t c = local_join<Parallel, Distance>(
        graph_updater, new_nbrs, old_nbrs, progress, block_size, n_threads,
        grain_size);

    TDOANN_ITERFINISHED();
    progress.heap_report(nn_heap);
    TDOANN_CHECKCONVERGENCE();
  }
}

template <typename Parallel, typename DistOut, typename Idx>
auto build_query_candidates(std::size_t n_ref_points,
                            std::size_t max_candidates,
                            const std::vector<Idx> &reference_idx,
                            const std::vector<DistOut> &reference_dist,
                            std::size_t n_nbrs, std::size_t n_threads,
                            std::size_t grain_size) -> NNHeap<DistOut, Idx> {
  NNHeap<DistOut, Idx> query_candidates(n_ref_points, max_candidates);
  auto worker = [&](std::size_t begin, std::size_t end) {
    build_query_candidates(reference_idx, reference_dist, n_nbrs,
                           query_candidates, begin, end);
  };
  Parallel::parallel_for(0, query_candidates.n_points, worker, n_threads,
                         grain_size);
  return query_candidates;
}

template <typename Parallel, typename Distance, typename Progress>
void nnd_query(
    const std::vector<typename Distance::Index> &reference_idx,
    const std::vector<typename Distance::Output> &reference_dist,
    NNHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    const Distance &distance, std::size_t max_candidates, double epsilon,
    std::size_t n_iters, Progress &progress, std::size_t n_threads = 0,
    std::size_t grain_size = 1) {
  const std::size_t n_points = nn_heap.n_points;
  const std::size_t n_nbrs = nn_heap.n_nbrs;

  const std::size_t n_ref_points = distance.nx;
  auto query_candidates = build_query_candidates<Parallel>(
      n_ref_points, max_candidates, reference_idx, reference_dist, n_nbrs,
      n_threads, grain_size);

  NullProgress null_progress;
  auto query_non_search_worker = [&](std::size_t begin, std::size_t end) {
    non_search_query(nn_heap, distance, query_candidates, epsilon,
                     null_progress, n_iters, begin, end);
  };
  batch_parallel_for<Parallel>(query_non_search_worker, progress, n_points,
                               n_threads, grain_size);
}
} // namespace tdoann
#endif // TDOANN_NNDPARALLEL_H

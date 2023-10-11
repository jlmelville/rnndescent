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

#include <array>
#include <mutex>

#include "heap.h"

namespace tdoann {

template <typename Distance> class LockingHeapAdder {
private:
  using Idx = typename Distance::Index;
  using Out = typename Distance::Output;

  static const constexpr std::size_t n_mutexes = 10;
  std::array<std::mutex, n_mutexes> mutexes;

public:
  LockingHeapAdder() = default;
  LockingHeapAdder(LockingHeapAdder const &) = delete;
  auto operator=(LockingHeapAdder const &) -> LockingHeapAdder & = delete;
  LockingHeapAdder(LockingHeapAdder &&) = delete;
  auto operator=(LockingHeapAdder &&) -> LockingHeapAdder & = delete;
  ~LockingHeapAdder() = default;

  void add(NNHeap<Out, Idx> &nbrs, Idx item_i, Idx item_j, Out dist_ij) {
    {
      std::lock_guard<std::mutex> guard(mutexes[item_i % n_mutexes]);
      nbrs.checked_push(item_i, dist_ij, item_j);
    }
    if (item_i != item_j) {
      std::lock_guard<std::mutex> guard(mutexes[item_j % n_mutexes]);
      nbrs.checked_push(item_j, dist_ij, item_i);
    }
  }
  void add(NNHeap<Out, Idx> &nbrs, Idx item_i, Idx item_j, Out dist_ij,
           Out dist_ji) {
    {
      std::lock_guard<std::mutex> guard(mutexes[item_i % n_mutexes]);
      nbrs.checked_push(item_i, dist_ij, item_j);
    }
    if (item_i != item_j) {
      std::lock_guard<std::mutex> guard(mutexes[item_j % n_mutexes]);
      nbrs.checked_push(item_j, dist_ji, item_i);
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

  for (std::size_t i = begin, idx_offset = begin * n_nbrs; i < end;
       i++, idx_offset += n_nbrs) {
    for (std::size_t j = 0, idx_ij = idx_offset; j < n_nbrs; j++, idx_ij++) {
      auto nbr = current_graph.idx[idx_ij];
      uint8_t isn = current_graph.flags[idx_ij];
      auto &nbrs = isn == 1 ? new_nbrs : old_nbrs;
      if (nbr == nbrs.npos()) {
        continue;
      }
      auto rand_weight = rand.unif();
      heap_adder.add(nbrs, i, nbr, rand_weight);
    }
  }
}

template <typename Parallel, typename Distance, typename ParallelRand>
void build_candidates(
    const NNDHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    NNHeap<typename Distance::Output, typename Distance::Index> &old_nbrs,
    ParallelRand &parallel_rand, LockingHeapAdder<Distance> &heap_adder,
    std::size_t n_threads) {
  parallel_rand.reseed();
  auto worker = [&](std::size_t begin, std::size_t end) {
    build_candidates(nn_heap, new_nbrs, old_nbrs, parallel_rand, heap_adder,
                     begin, end);
  };
  const std::size_t grain_size = 1;
  Parallel::parallel_for(0, nn_heap.n_points, worker, n_threads, grain_size);
}

template <typename Parallel, typename Distance>
void flag_new_candidates(
    NNDHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    const NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    std::size_t n_threads) {
  auto worker = [&](std::size_t begin, std::size_t end) {
    flag_retained_new_candidates(nn_heap, new_nbrs, begin, end);
  };
  const std::size_t grain_size = 1;
  Parallel::parallel_for(0, nn_heap.n_points, worker, n_threads, grain_size);
}

template <typename Distance, typename GraphUpdater>
void local_join(
    GraphUpdater &graph_updater,
    const NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    const NNHeap<typename Distance::Output, typename Distance::Index> &old_nbrs,
    std::size_t max_candidates, std::size_t begin, std::size_t end) {
  for (std::size_t i = begin, idx_offset = begin * max_candidates; i < end;
  i++, idx_offset += max_candidates) {
    for (std::size_t j = 0; j < max_candidates; j++) {
      std::size_t item_p = new_nbrs.idx[idx_offset + j];
      if (item_p == new_nbrs.npos()) {
        continue;
      }
      for (std::size_t k = j; k < max_candidates; k++) {
        std::size_t item_new = new_nbrs.idx[idx_offset + k];
        if (item_new == new_nbrs.npos()) {
          continue;
        }
        graph_updater.generate(item_p, item_new, i);
      }
      for (std::size_t k = 0; k < max_candidates; k++) {
        std::size_t item_old = old_nbrs.idx[idx_offset + k];
        if (item_old == old_nbrs.npos()) {
          continue;
        }
        graph_updater.generate(item_p, item_old, i);
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
    Progress &progress, std::size_t n_threads) -> std::size_t {
  std::size_t num_updated = 0;
  auto local_join_worker = [&](std::size_t begin, std::size_t end) {
    local_join<Distance, decltype(graph_updater)>(
        graph_updater, new_nbrs, old_nbrs, new_nbrs.n_nbrs, begin, end);
  };
  auto after_local_join = [&](std::size_t, std::size_t) {
    num_updated += graph_updater.apply();
  };
  const std::size_t block_size = 16384;
  const std::size_t grain_size = 1;
  batch_parallel_for<Parallel>(local_join_worker, after_local_join, progress,
                               graph_updater.current_graph.n_points, block_size,
                               n_threads, grain_size);
  return num_updated;
}

template <typename Parallel, typename ParallelRand,
          template <typename> class GraphUpdater, typename Distance,
          typename Progress>
void nnd_build(GraphUpdater<Distance> &graph_updater,
               std::size_t max_candidates, std::size_t n_iters, double delta,
               Progress &progress, ParallelRand &parallel_rand,
               std::size_t n_threads = 0) {

  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  auto &nn_heap = graph_updater.current_graph;
  const std::size_t n_points = nn_heap.n_points;
  const double tol = delta * nn_heap.n_nbrs * n_points;

  LockingHeapAdder<Distance> heap_adder;

  for (std::size_t iter = 0; iter < n_iters; iter++) {
    NNHeap<DistOut, Idx> new_nbrs(n_points, max_candidates);
    decltype(new_nbrs) old_nbrs(n_points, max_candidates);

    build_candidates<Parallel, Distance>(nn_heap, new_nbrs, old_nbrs,
                                         parallel_rand, heap_adder, n_threads);

    // mark any neighbor in the current graph that was retained in the new
    // candidates as true
    flag_new_candidates<Parallel, Distance>(nn_heap, new_nbrs, n_threads);

    std::size_t num_updated = local_join<Parallel, Distance>(
        graph_updater, new_nbrs, old_nbrs, progress, n_threads);

    TDOANN_ITERFINISHED();
    progress.heap_report(nn_heap);
    if (is_converged(num_updated, tol)) {
      progress.converged(num_updated, tol);
      break;
    }
  }
}

} // namespace tdoann
#endif // TDOANN_NNDPARALLEL_H

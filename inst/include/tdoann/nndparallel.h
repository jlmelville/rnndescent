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

template <typename ParallelRand, typename Distance>
struct LockingCandidatesWorker {
  const NNDHeap<typename Distance::Output, typename Distance::Index>
      &current_graph;
  std::size_t n_points;
  std::size_t n_nbrs;
  std::size_t max_candidates;
  NNHeap<typename Distance::Output, typename Distance::Index>
      &new_candidate_neighbors;
  NNHeap<typename Distance::Output, typename Distance::Index>
      &old_candidate_neighbors;
  static const constexpr std::size_t n_mutexes = 10;
  std::mutex mutexes[n_mutexes];

  ParallelRand &parallel_rand;

  LockingCandidatesWorker(
      const NNDHeap<typename Distance::Output, typename Distance::Index>
          &current_graph,
      NNHeap<typename Distance::Output, typename Distance::Index>
          &new_candidate_neighbors,
      NNHeap<typename Distance::Output, typename Distance::Index>
          &old_candidate_neighbors,
      ParallelRand &parallel_rand)
      : current_graph(current_graph), n_points(current_graph.n_points),
        n_nbrs(current_graph.n_nbrs),
        max_candidates(new_candidate_neighbors.n_nbrs),
        new_candidate_neighbors(new_candidate_neighbors),
        old_candidate_neighbors(old_candidate_neighbors),
        parallel_rand(parallel_rand) {
    parallel_rand.reseed();
  }

  void operator()(std::size_t begin, std::size_t end) {

    auto rand = parallel_rand.get_rand(end);

    for (auto i = begin; i < end; i++) {
      std::size_t innbrs = i * n_nbrs;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        std::size_t ij = innbrs + j;
        std::size_t idx = current_graph.idx[ij];
        char isn = current_graph.flags[ij];
        auto &nbrs =
            isn == 1 ? new_candidate_neighbors : old_candidate_neighbors;
        if (idx == nbrs.npos()) {
          continue;
        }
        auto d = rand.unif();
        {
          std::lock_guard<std::mutex> guard(mutexes[i % n_mutexes]);
          nbrs.checked_push(i, d, idx);
        }
        if (i != idx) {
          std::lock_guard<std::mutex> guard(mutexes[idx % n_mutexes]);
          nbrs.checked_push(idx, d, i);
        }
      }
    }
  }
};

// mark any neighbor in the current graph that was retained in the new
// candidates as true
template <typename Distance> struct FlagNewCandidatesWorker {
  const NNHeap<typename Distance::Output, typename Distance::Index>
      &new_candidate_neighbors;
  NNDHeap<typename Distance::Output, typename Distance::Index> &current_graph;
  std::size_t n_points;
  std::size_t n_nbrs;
  std::size_t max_candidates;

  FlagNewCandidatesWorker(
      const NNHeap<typename Distance::Output, typename Distance::Index>
          &new_candidate_neighbors,
      NNDHeap<typename Distance::Output, typename Distance::Index>
          &current_graph)
      : new_candidate_neighbors(new_candidate_neighbors),
        current_graph(current_graph), n_points(current_graph.n_points),
        n_nbrs(current_graph.n_nbrs),
        max_candidates(new_candidate_neighbors.n_nbrs) {}

  void operator()(std::size_t begin, std::size_t end) {
    flag_retained_new_candidates(current_graph, new_candidate_neighbors, begin,
                                 end);
  }
};

template <typename Distance, typename GraphUpdater> struct LocalJoinWorker {
  const NNDHeap<typename Distance::Output, typename Distance::Index>
      &current_graph;
  const NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs;
  const NNHeap<typename Distance::Output, typename Distance::Index> &old_nbrs;
  std::size_t n_nbrs;
  std::size_t max_candidates;
  GraphUpdater &graph_updater;
  std::size_t c;

  LocalJoinWorker(
      const NNDHeap<typename Distance::Output, typename Distance::Index>
          &current_graph,
      NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
      NNHeap<typename Distance::Output, typename Distance::Index> &old_nbrs,
      GraphUpdater &graph_updater)
      : current_graph(current_graph), new_nbrs(new_nbrs), old_nbrs(old_nbrs),
        n_nbrs(current_graph.n_nbrs), max_candidates(new_nbrs.n_nbrs),
        graph_updater(graph_updater), c(0) {}

  void operator()(std::size_t begin, std::size_t end) {
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
  void after_parallel(std::size_t, std::size_t) { c += graph_updater.apply(); }
};

template <typename Parallel, typename ParallelRand,
          template <typename> class GraphUpdater, typename Distance,
          typename Progress>
void nnd_build_parallel(GraphUpdater<Distance> &graph_updater,
                        std::size_t max_candidates, std::size_t n_iters,
                        double delta, Progress &progress,
                        ParallelRand &parallel_rand, std::size_t n_threads = 0,
                        std::size_t block_size = 16384,
                        std::size_t grain_size = 1) {

  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  auto &nn_heap = graph_updater.current_graph;
  const std::size_t n_points = nn_heap.n_points;
  const double tol = delta * nn_heap.n_nbrs * n_points;

  for (std::size_t n = 0; n < n_iters; n++) {
    NNHeap<DistOut, Idx> new_nbrs(n_points, max_candidates);
    decltype(new_nbrs) old_nbrs(n_points, max_candidates);

    LockingCandidatesWorker<ParallelRand, Distance> candidates_worker(
        nn_heap, new_nbrs, old_nbrs, parallel_rand);
    Parallel::parallel_for(0, n_points, candidates_worker, n_threads,
                           grain_size);

    FlagNewCandidatesWorker<Distance> flag_new_candidates_worker(new_nbrs,
                                                                 nn_heap);
    Parallel::parallel_for(0, n_points, flag_new_candidates_worker, n_threads,
                           grain_size);

    LocalJoinWorker<Distance, decltype(graph_updater)> local_join_worker(
        nn_heap, new_nbrs, old_nbrs, graph_updater);
    batch_parallel_for<Parallel>(local_join_worker, progress, n_points,
                                 n_threads, block_size, grain_size);
    TDOANN_ITERFINISHED();
    progress.heap_report(nn_heap);
    std::size_t c = local_join_worker.c;
    TDOANN_CHECKCONVERGENCE();
  }
}

template <typename Distance> struct QueryCandidatesWorker {
  NNDHeap<typename Distance::Output, typename Distance::Index> &current_graph;
  std::size_t n_points;
  std::size_t n_nbrs;
  std::size_t max_candidates;
  bool flag_on_add;

  NNHeap<typename Distance::Output, typename Distance::Index>
      &new_candidate_neighbors;

  QueryCandidatesWorker(
      NNDHeap<typename Distance::Output, typename Distance::Index>
          &current_graph,
      NNHeap<typename Distance::Output, typename Distance::Index>
          &new_candidate_neighbors)
      : current_graph(current_graph), n_points(current_graph.n_points),
        n_nbrs(current_graph.n_nbrs),
        max_candidates(new_candidate_neighbors.n_nbrs),
        flag_on_add(new_candidate_neighbors.n_nbrs >= current_graph.n_nbrs),
        new_candidate_neighbors(new_candidate_neighbors) {}

  void operator()(std::size_t begin, std::size_t end) {
    build_query_candidates(current_graph, new_candidate_neighbors, begin, end,
                           flag_on_add);
  }
};

template <template <typename> class GraphUpdater, typename Distance>
struct QueryNoNSearchWorker : public BatchParallelWorker {
  NNDHeap<typename Distance::Output, typename Distance::Index> &current_graph;
  GraphUpdater<Distance> &graph_updater;
  const NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs;
  const NNHeap<typename Distance::Output, typename Distance::Index> &ref_heap;
  std::size_t max_candidates;
  std::mutex mutex;
  NullProgress progress;
  std::size_t n_updates;

  QueryNoNSearchWorker(
      NNDHeap<typename Distance::Output, typename Distance::Index>
          &current_graph,
      GraphUpdater<Distance> &graph_updater,
      const NNHeap<typename Distance::Output, typename Distance::Index>
          &new_nbrs,
      const NNHeap<typename Distance::Output, typename Distance::Index>
          &ref_heap,
      std::size_t max_candidates)
      : current_graph(current_graph), graph_updater(graph_updater),
        new_nbrs(new_nbrs), ref_heap(ref_heap), max_candidates(max_candidates),
        progress(), n_updates(0) {}

  void operator()(std::size_t begin, std::size_t end) {
    std::size_t ref_idx = 0;
    std::size_t nbr_ref_idx = 0;
    std::size_t n_nbrs = current_graph.n_nbrs;
    typename GraphUpdater<Distance>::NeighborSet seen(n_nbrs);

    for (std::size_t query_idx = begin; query_idx < end; query_idx++) {
      for (std::size_t j = 0; j < max_candidates; j++) {
        ref_idx = new_nbrs.index(query_idx, j);
        if (ref_idx == new_nbrs.npos()) {
          continue;
        }
        std::size_t rnidx = ref_idx * max_candidates;
        for (std::size_t k = 0; k < max_candidates; k++) {
          nbr_ref_idx = ref_heap.idx[rnidx + k];
          if (nbr_ref_idx == ref_heap.npos() || seen.contains(nbr_ref_idx)) {
            continue;
          } else {
            graph_updater.generate(query_idx, nbr_ref_idx, -1);
          }
        }
      }
      TDOANN_BLOCKFINISHED();
      seen.clear();
    }
  }
  void after_parallel(std::size_t, std::size_t) {
    n_updates += graph_updater.apply();
  }
};

template <typename DistOut, typename Idx> struct RNWorker {
  const std::vector<Idx> &reference_idx;
  const std::vector<DistOut> &reference_dist;
  std::size_t n_nbrs;
  NNHeap<DistOut, Idx> &ref_heap;

  RNWorker(const std::vector<Idx> &reference_idx,
           const std::vector<DistOut> &reference_dist, std::size_t n_nbrs,
           NNHeap<DistOut, Idx> &ref_heap)
      : reference_idx(reference_idx), reference_dist(reference_dist),
        n_nbrs(n_nbrs), ref_heap(ref_heap) {}

  void operator()(std::size_t begin, std::size_t end) {
    build_ref_nbrs(reference_idx, reference_dist, n_nbrs, ref_heap, begin, end);
  }
};

template <typename Parallel, typename DistOut, typename Idx>
auto build_ref_nbrs_parallel(std::size_t n_ref_points,
                             std::size_t max_candidates,
                             const std::vector<Idx> &reference_idx,
                             const std::vector<DistOut> &reference_dist,
                             std::size_t n_nbrs, std::size_t n_threads,
                             std::size_t grain_size) -> NNHeap<DistOut, Idx> {
  NNHeap<DistOut, Idx> ref_heap(n_ref_points, max_candidates);
  RNWorker<DistOut, Idx> worker(reference_idx, reference_dist, n_nbrs,
                                ref_heap);
  Parallel::parallel_for(0, ref_heap.n_points, worker, n_threads, grain_size);
  return ref_heap;
}

template <typename Parallel, template <typename> class GraphUpdater,
          typename Distance, typename Progress>
void nnd_query_parallel(
    const std::vector<typename Distance::Index> &reference_idx,
    const std::vector<typename Distance::Output> &reference_dist,
    GraphUpdater<Distance> &graph_updater, std::size_t max_candidates,
    std::size_t n_iters, double delta, Progress &progress,
    std::size_t n_threads = 0, std::size_t block_size = 16384,
    std::size_t grain_size = 1) {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  auto &nn_heap = graph_updater.current_graph;
  const std::size_t n_points = nn_heap.n_points;
  const std::size_t n_nbrs = nn_heap.n_nbrs;
  const double tol = delta * n_nbrs * n_points;

  const std::size_t n_ref_points = graph_updater.distance.nx;
  auto ref_heap = build_ref_nbrs_parallel<Parallel>(
      n_ref_points, max_candidates, reference_idx, reference_dist, n_nbrs,
      n_threads, grain_size);

  for (std::size_t n = 0; n < n_iters; n++) {
    NNHeap<DistOut, Idx> new_nbrs(n_points, max_candidates);
    QueryCandidatesWorker<Distance> query_candidates_worker(nn_heap, new_nbrs);
    Parallel::parallel_for(0, n_points, query_candidates_worker, n_threads,
                           grain_size);

    if (!query_candidates_worker.flag_on_add) {
      FlagNewCandidatesWorker<Distance> flag_new_candidates_worker(new_nbrs,
                                                                   nn_heap);
      Parallel::parallel_for(0, n_points, flag_new_candidates_worker, n_threads,
                             grain_size);
    }

    QueryNoNSearchWorker<GraphUpdater, Distance> query_non_search_worker(
        nn_heap, graph_updater, new_nbrs, ref_heap, max_candidates);
    batch_parallel_for<Parallel>(query_non_search_worker, progress, n_points,
                                 n_threads, block_size, grain_size);

    TDOANN_ITERFINISHED();
    progress.heap_report(nn_heap);
    std::size_t c = query_non_search_worker.n_updates;
    TDOANN_CHECKCONVERGENCE();
  }
}
} // namespace tdoann
#endif // TDOANN_NNDPARALLEL_H

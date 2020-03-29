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

template <typename CandidatePriorityFactoryImpl>
struct LockingCandidatesWorker {
  const NeighborHeap &current_graph;
  CandidatePriorityFactoryImpl candidate_priority_factory;
  std::size_t n_points;
  std::size_t n_nbrs;
  std::size_t max_candidates;
  NeighborHeap &new_candidate_neighbors;
  NeighborHeap &old_candidate_neighbors;
  static const constexpr std::size_t n_mutexes = 10;
  std::mutex mutexes[n_mutexes];

  LockingCandidatesWorker(
      const NeighborHeap &current_graph,
      CandidatePriorityFactoryImpl &candidate_priority_factory,
      NeighborHeap &new_candidate_neighbors,
      NeighborHeap &old_candidate_neighbors)
      : current_graph(current_graph),
        candidate_priority_factory(candidate_priority_factory),
        n_points(current_graph.n_points), n_nbrs(current_graph.n_nbrs),
        max_candidates(new_candidate_neighbors.n_nbrs),
        new_candidate_neighbors(new_candidate_neighbors),
        old_candidate_neighbors(old_candidate_neighbors) {}

  void operator()(std::size_t begin, std::size_t end) {
    auto candidate_priority = candidate_priority_factory.create(begin, end);
    for (auto i = begin; i < end; i++) {
      std::size_t innbrs = i * n_nbrs;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        std::size_t ij = innbrs + j;
        std::size_t idx = current_graph.idx[ij];
        double d = candidate_priority(current_graph, ij);
        char isn = current_graph.flags[ij];
        auto &nbrs =
            isn == 1 ? new_candidate_neighbors : old_candidate_neighbors;
        {
          std::lock_guard<std::mutex> guard(mutexes[i % n_mutexes]);
          nbrs.checked_push(i, d, idx, isn);
        }
        if (i != idx) {
          std::lock_guard<std::mutex> guard(mutexes[idx % n_mutexes]);
          nbrs.checked_push(idx, d, i, isn);
        }
      }
    }
  }
};

// mark any neighbor in the current graph that was retained in the new
// candidates as true
struct FlagNewCandidatesWorker {
  const NeighborHeap &new_candidate_neighbors;
  NeighborHeap &current_graph;
  std::size_t n_points;
  std::size_t n_nbrs;
  std::size_t max_candidates;

  FlagNewCandidatesWorker(const NeighborHeap &new_candidate_neighbors,
                          NeighborHeap &current_graph)
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
  const NeighborHeap &current_graph;
  const NeighborHeap &new_nbrs;
  const NeighborHeap &old_nbrs;
  std::size_t n_nbrs;
  std::size_t max_candidates;
  GraphUpdater &graph_updater;
  std::size_t c;

  LocalJoinWorker(const NeighborHeap &current_graph, NeighborHeap &new_nbrs,
                  NeighborHeap &old_nbrs, GraphUpdater &graph_updater)
      : current_graph(current_graph), new_nbrs(new_nbrs), old_nbrs(old_nbrs),
        n_nbrs(current_graph.n_nbrs), max_candidates(new_nbrs.n_nbrs),
        graph_updater(graph_updater), c(0) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; i++) {
      std::size_t imaxc = i * max_candidates;
      for (std::size_t j = 0; j < max_candidates; j++) {
        std::size_t p = new_nbrs.idx[imaxc + j];
        if (p == NeighborHeap::npos()) {
          continue;
        }
        for (std::size_t k = j; k < max_candidates; k++) {
          std::size_t q = new_nbrs.idx[imaxc + k];
          if (q == NeighborHeap::npos()) {
            continue;
          }
          graph_updater.generate(p, q, i);
        }

        for (std::size_t k = 0; k < max_candidates; k++) {
          std::size_t q = old_nbrs.idx[imaxc + k];
          if (q == NeighborHeap::npos()) {
            continue;
          }
          graph_updater.generate(p, q, i);
        }
      }
    }
  }
  void after_parallel(std::size_t begin, std::size_t end) {
    c += graph_updater.apply();
  }
};

template <typename Distance, typename GUFactoryT, typename Progress,
          typename Parallel, typename CandidatePriorityFactoryImpl>
NNGraph nnd_build_parallel(
    const std::vector<typename Distance::Input> &data, std::size_t ndim,
    const NNGraph &nn_init, std::size_t max_candidates, std::size_t n_iters,
    CandidatePriorityFactoryImpl &candidate_priority_factory, double delta,
    std::size_t n_threads = 0, std::size_t block_size = 16384,
    std::size_t grain_size = 1, bool verbose = false) {
  Distance distance(data, ndim);

  std::size_t n_points = nn_init.n_points;
  std::size_t n_nbrs = nn_init.n_nbrs;
  double tol = delta * n_nbrs * n_points;

  NeighborHeap current_graph(n_points, n_nbrs);
  graph_to_heap_parallel<LockingHeapAddSymmetric>(
      current_graph, nn_init, n_threads, 1000, grain_size, true);

  Progress progress(current_graph, n_iters, verbose);
  auto graph_updater = GUFactoryT::create(current_graph, distance);

  for (std::size_t n = 0; n < n_iters; n++) {
    NeighborHeap new_candidate_neighbors(n_points, max_candidates);
    NeighborHeap old_candidate_neighbors(n_points, max_candidates);

    LockingCandidatesWorker<CandidatePriorityFactoryImpl> candidates_worker(
        current_graph, candidate_priority_factory, new_candidate_neighbors,
        old_candidate_neighbors);
    Parallel::parallel_for(0, n_points, candidates_worker, n_threads,
                           grain_size);
    if (CandidatePriorityFactoryImpl::should_sort) {
      sort_heap_parallel(new_candidate_neighbors, n_threads, block_size,
                         grain_size);
      sort_heap_parallel(old_candidate_neighbors, n_threads, block_size,
                         grain_size);
    }

    FlagNewCandidatesWorker flag_new_candidates_worker(new_candidate_neighbors,
                                                       current_graph);
    Parallel::parallel_for(0, n_points, flag_new_candidates_worker, n_threads,
                           grain_size);

    LocalJoinWorker<Distance, decltype(graph_updater)> local_join_worker(
        current_graph, new_candidate_neighbors, old_candidate_neighbors,
        graph_updater);
    batch_parallel_for<Parallel>(local_join_worker, progress, n_points,
                                 n_threads, block_size, grain_size);
    TDOANN_ITERFINISHED();
    std::size_t c = local_join_worker.c;
    TDOANN_CHECKCONVERGENCE();
  }
  sort_heap_parallel(current_graph, n_threads, block_size, grain_size);

  return heap_to_graph(current_graph);
}

template <typename CandidatePriorityFactoryImpl> struct QueryCandidatesWorker {
  NeighborHeap &current_graph;
  std::size_t n_points;
  std::size_t n_nbrs;
  std::size_t max_candidates;
  const bool flag_on_add;

  NeighborHeap &new_candidate_neighbors;
  CandidatePriorityFactoryImpl &candidate_priority_factory;

  QueryCandidatesWorker(
      NeighborHeap &current_graph, NeighborHeap &new_candidate_neighbors,
      CandidatePriorityFactoryImpl &candidate_priority_factory)
      : current_graph(current_graph), n_points(current_graph.n_points),
        n_nbrs(current_graph.n_nbrs),
        max_candidates(new_candidate_neighbors.n_nbrs),
        flag_on_add(new_candidate_neighbors.n_nbrs >= current_graph.n_nbrs),
        new_candidate_neighbors(new_candidate_neighbors),
        candidate_priority_factory(candidate_priority_factory) {}

  void operator()(std::size_t begin, std::size_t end) {
    auto candidate_priority = candidate_priority_factory.create(begin, end);
    build_query_candidates(current_graph, candidate_priority,
                           new_candidate_neighbors, begin, end, flag_on_add);
  }
};

template <typename Distance, typename GraphUpdater>
struct QueryNoNSearchWorker : public BatchParallelWorker {
  NeighborHeap &current_graph;
  GraphUpdater &graph_updater;
  const NeighborHeap &new_nbrs;
  const NeighborHeap &gn_graph;
  std::size_t max_candidates;
  std::mutex mutex;
  NullProgress progress;
  std::size_t n_updates;

  QueryNoNSearchWorker(NeighborHeap &current_graph, GraphUpdater &graph_updater,
                       const NeighborHeap &new_nbrs,
                       const NeighborHeap &gn_graph, std::size_t max_candidates)
      : current_graph(current_graph), graph_updater(graph_updater),
        new_nbrs(new_nbrs), gn_graph(gn_graph), max_candidates(max_candidates),
        progress(), n_updates(0) {}

  void operator()(std::size_t begin, std::size_t end) {
    std::size_t ref_idx = 0;
    std::size_t nbr_ref_idx = 0;
    std::size_t n_nbrs = current_graph.n_nbrs;
    typename GraphUpdater::NeighborSet seen(n_nbrs);

    for (std::size_t query_idx = begin; query_idx < end; query_idx++) {
      for (std::size_t j = 0; j < max_candidates; j++) {
        ref_idx = new_nbrs.index(query_idx, j);
        if (ref_idx == NeighborHeap::npos()) {
          continue;
        }
        std::size_t rnidx = ref_idx * max_candidates;
        for (std::size_t k = 0; k < max_candidates; k++) {
          nbr_ref_idx = gn_graph.idx[rnidx + k];
          if (nbr_ref_idx == NeighborHeap::npos() ||
              seen.contains(nbr_ref_idx)) {
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
  void after_parallel(std::size_t begin, std::size_t end) {
    n_updates += graph_updater.apply();
  }
};

template <typename GUFactoryT, typename Progress, typename Parallel,
          typename Distance, typename CandidatePriorityFactoryImpl>
void nnd_query_parallel(
    Distance &distance, NeighborHeap &current_graph,
    const std::vector<std::size_t> &reference_idx, std::size_t n_ref_points,
    std::size_t max_candidates, std::size_t n_iters,
    CandidatePriorityFactoryImpl &candidate_priority_factory, double tol,
    std::size_t n_threads = 0, std::size_t block_size = 16384,
    std::size_t grain_size = 1, bool verbose = false) {

  Progress progress(current_graph, n_iters, verbose);
  auto graph_updater = GUFactoryT::create(current_graph, distance);

  std::size_t n_points = current_graph.n_points;
  std::size_t n_nbrs = current_graph.n_nbrs;

  NeighborHeap gn_graph(n_ref_points, max_candidates);
  auto candidate_priority = candidate_priority_factory.create();
  build_general_nbrs(reference_idx, gn_graph, candidate_priority, n_ref_points,
                     n_nbrs);
  for (std::size_t n = 0; n < n_iters; n++) {
    NeighborHeap new_nbrs(n_points, max_candidates);
    QueryCandidatesWorker<CandidatePriorityFactoryImpl> query_candidates_worker(
        current_graph, new_nbrs, candidate_priority_factory);
    Parallel::parallel_for(0, n_points, query_candidates_worker, n_threads,
                           grain_size);

    if (!query_candidates_worker.flag_on_add) {
      FlagNewCandidatesWorker flag_new_candidates_worker(new_nbrs,
                                                         current_graph);
      Parallel::parallel_for(0, n_points, flag_new_candidates_worker, n_threads,
                             grain_size);
    }

    QueryNoNSearchWorker<Distance, decltype(graph_updater)>
        query_non_search_worker(current_graph, graph_updater, new_nbrs,
                                gn_graph, max_candidates);
    batch_parallel_for<Parallel>(query_non_search_worker, progress, n_points,
                                 n_threads, block_size, grain_size);

    TDOANN_ITERFINISHED();
    std::size_t c = query_non_search_worker.n_updates;
    TDOANN_CHECKCONVERGENCE();
  }
  sort_heap_parallel(current_graph, n_threads, block_size, grain_size);
}
} // namespace tdoann
#endif // TDOANN_NNDPARALLEL_H

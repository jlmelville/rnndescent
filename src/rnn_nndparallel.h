//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2019 James Melville
//
//  This file is part of rnndescent
//
//  rnndescent is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  rnndescent is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with rnndescent.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RNN_NNDPARALLEL_H
#define RNN_NNDPARALLEL_H

#include <mutex>
#include <vector>

#include <Rcpp.h>

#include "tdoann/graphupdate.h"
#include "tdoann/heap.h"
#include "tdoann/nndescent.h"
#include "tdoann/parallel.h"
#include "tdoann/progress.h"

#include "RcppPerpendicular.h"
#include "rnn_heapsort.h"

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

template <typename Distance, template <typename> class GraphUpdater>
struct LocalJoinWorker {
  const NeighborHeap &current_graph;
  const NeighborHeap &new_nbrs;
  const NeighborHeap &old_nbrs;
  std::size_t n_nbrs;
  std::size_t max_candidates;
  GraphUpdater<Distance> &graph_updater;
  std::size_t c;

  LocalJoinWorker(const NeighborHeap &current_graph, NeighborHeap &new_nbrs,
                  NeighborHeap &old_nbrs, GraphUpdater<Distance> &graph_updater)
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

template <typename Distance, typename CandidatePriorityFactoryImpl,
          typename Progress, template <typename> class GraphUpdater,
          typename Parallel>
void nnd_parallel(NeighborHeap &current_graph,
                  GraphUpdater<Distance> &graph_updater,
                  std::size_t max_candidates, std::size_t n_iters,
                  CandidatePriorityFactoryImpl &candidate_priority_factory,
                  Progress &progress, double tol, std::size_t n_threads = 0,
                  std::size_t block_size = 16384, std::size_t grain_size = 1,
                  bool verbose = false) {
  std::size_t n_points = current_graph.n_points;

  for (std::size_t n = 0; n < n_iters; n++) {
    NeighborHeap new_candidate_neighbors(n_points, max_candidates);
    NeighborHeap old_candidate_neighbors(n_points, max_candidates);

    LockingCandidatesWorker<CandidatePriorityFactoryImpl> candidates_worker(
        current_graph, candidate_priority_factory, new_candidate_neighbors,
        old_candidate_neighbors);
    RcppPerpendicular::parallel_for(0, n_points, candidates_worker, n_threads,
                                    grain_size);
    if (CandidatePriorityFactoryImpl::should_sort) {
      sort_heap_parallel(new_candidate_neighbors, n_threads, block_size,
                         grain_size);
      sort_heap_parallel(old_candidate_neighbors, n_threads, block_size,
                         grain_size);
    }

    FlagNewCandidatesWorker flag_new_candidates_worker(new_candidate_neighbors,
                                                       current_graph);
    RcppPerpendicular::parallel_for(0, n_points, flag_new_candidates_worker,
                                    n_threads, grain_size);

    LocalJoinWorker<Distance, GraphUpdater> local_join_worker(
        current_graph, new_candidate_neighbors, old_candidate_neighbors,
        graph_updater);
    tdoann::batch_parallel_for<Progress, decltype(local_join_worker), Parallel>(
        local_join_worker, progress, n_points, n_threads, block_size,
        grain_size);
    TDOANN_ITERFINISHED();
    std::size_t c = local_join_worker.c;
    TDOANN_CHECKCONVERGENCE();
  }
  sort_heap_parallel(current_graph, n_threads, block_size, grain_size);
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

template <typename Distance, template <typename> class GraphUpdater>
struct QueryNoNSearchWorker : public tdoann::BatchParallelWorker {
  NeighborHeap &current_graph;
  GraphUpdater<Distance> &graph_updater;
  const NeighborHeap &new_nbrs;
  const NeighborHeap &gn_graph;
  std::size_t max_candidates;
  std::mutex mutex;
  tdoann::NullProgress progress;
  std::size_t n_updates;

  QueryNoNSearchWorker(NeighborHeap &current_graph,
                       GraphUpdater<Distance> &graph_updater,
                       const NeighborHeap &new_nbrs,
                       const NeighborHeap &gn_graph, std::size_t max_candidates)
      : current_graph(current_graph), graph_updater(graph_updater),
        new_nbrs(new_nbrs), gn_graph(gn_graph), max_candidates(max_candidates),
        progress(), n_updates(0) {}

  void operator()(std::size_t begin, std::size_t end) {
    std::size_t ref_idx = 0;
    std::size_t nbr_ref_idx = 0;
    std::size_t n_nbrs = current_graph.n_nbrs;
    typename GraphUpdater<Distance>::NeighborSet seen(n_nbrs);

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

template <typename Distance, typename CandidatePriorityFactoryImpl,
          typename Progress, template <typename> class GraphUpdater,
          typename Parallel>
void nnd_query_parallel(
    NeighborHeap &current_graph, GraphUpdater<Distance> &graph_updater,
    const std::vector<std::size_t> &reference_idx, std::size_t n_ref_points,
    std::size_t max_candidates, std::size_t n_iters,
    CandidatePriorityFactoryImpl &candidate_priority_factory,
    Progress &progress, double tol, std::size_t n_threads = 0,
    std::size_t block_size = 16384, std::size_t grain_size = 1,
    bool verbose = false) {
  std::size_t n_points = current_graph.n_points;
  std::size_t n_nbrs = current_graph.n_nbrs;

  NeighborHeap gn_graph(n_ref_points, max_candidates);
  auto candidate_priority = candidate_priority_factory.create();
  tdoann::build_general_nbrs(reference_idx, gn_graph, candidate_priority,
                             n_ref_points, n_nbrs);
  for (std::size_t n = 0; n < n_iters; n++) {
    NeighborHeap new_nbrs(n_points, max_candidates);
    QueryCandidatesWorker<CandidatePriorityFactoryImpl> query_candidates_worker(
        current_graph, new_nbrs, candidate_priority_factory);
    RcppPerpendicular::parallel_for(0, n_points, query_candidates_worker,
                                    n_threads, grain_size);

    if (!query_candidates_worker.flag_on_add) {
      FlagNewCandidatesWorker flag_new_candidates_worker(new_nbrs,
                                                         current_graph);
      RcppPerpendicular::parallel_for(0, n_points, flag_new_candidates_worker,
                                      n_threads, grain_size);
    }

    QueryNoNSearchWorker<Distance, GraphUpdater> query_non_search_worker(
        current_graph, graph_updater, new_nbrs, gn_graph, max_candidates);
    tdoann::batch_parallel_for<Progress, decltype(query_non_search_worker),
                               Parallel>(query_non_search_worker, progress,
                                         n_points, n_threads, block_size,
                                         grain_size);

    TDOANN_ITERFINISHED();
    std::size_t c = query_non_search_worker.n_updates;
    TDOANN_CHECKCONVERGENCE();
  }
  sort_heap_parallel(current_graph, n_threads, block_size, grain_size);
}

#endif // RNN_NNDPARALLEL_H

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

#ifndef RNND_PARALLEL_H
#define RNND_PARALLEL_H

#include <vector>

#include <Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>

#include "graph_update.h"
#include "heap.h"
#include "nndescent.h"

#include "rrand.h"

struct LockingCandidatesWorker : public RcppParallel::Worker {
  const NeighborHeap& current_graph;
  const std::size_t n_points;
  const std::size_t n_nbrs;
  const std::size_t max_candidates;
  NeighborHeap& new_candidate_neighbors;
  NeighborHeap& old_candidate_neighbors;
  tthread::mutex mutex;

  LockingCandidatesWorker(
    const NeighborHeap& current_graph,
    NeighborHeap& new_candidate_neighbors,
    NeighborHeap& old_candidate_neighbors
  ) :
    current_graph(current_graph),
    n_points(current_graph.n_points),
    n_nbrs(current_graph.n_nbrs),
    max_candidates(new_candidate_neighbors.n_nbrs),
    new_candidate_neighbors(new_candidate_neighbors),
    old_candidate_neighbors(old_candidate_neighbors)
  {}

  void operator()(std::size_t begin, std::size_t end) {
    std::unique_ptr<TauRand> rand(nullptr);
    // Each window gets its own PRNG state, to prevent locking inside the loop.
    {
      tthread::lock_guard<tthread::mutex> guard(mutex);
      rand.reset(new TauRand());
    }

    for (std::size_t i = begin; i < end; i++) {
      std::size_t innbrs = i * n_nbrs;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        std::size_t ij = innbrs + j;
        std::size_t idx = current_graph.idx[ij];
        double d = rand->unif();
        bool isn = current_graph.flags[ij] == 1;
        if (isn) {
          {
            tthread::lock_guard<tthread::mutex> guard(mutex);
            new_candidate_neighbors.checked_push_pair(i, d, idx, isn);
          }
        }
        else {
          {
            tthread::lock_guard<tthread::mutex> guard(mutex);
            old_candidate_neighbors.checked_push_pair(i, d, idx, isn);
          }
        }
      }
    }
  }
};

// mark any neighbor in the current graph that was retained in the new
// candidates as true
struct NewCandidatesWorker : public RcppParallel::Worker {
  const NeighborHeap& new_candidate_neighbors;
  NeighborHeap& current_graph;
  const std::size_t n_points;
  const std::size_t n_nbrs;
  const std::size_t max_candidates;

  NewCandidatesWorker(
    const NeighborHeap& new_candidate_neighbors,
    NeighborHeap& current_graph
  ) :
    new_candidate_neighbors(new_candidate_neighbors),
    current_graph(current_graph),
    n_points(current_graph.n_points),
    n_nbrs(current_graph.n_nbrs),
    max_candidates(new_candidate_neighbors.n_nbrs)
  {}

  void operator()(std::size_t begin, std::size_t end)
  {
    for (std::size_t i = begin; i < end; i++) {
      std::size_t innbrs = i * n_nbrs;
      std::size_t innbrs_new = i * max_candidates;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        std::size_t ij = innbrs + j;
        std::size_t idx = current_graph.idx[ij];
        for (std::size_t k = 0; k < max_candidates; k++) {
          if (new_candidate_neighbors.idx[innbrs_new + k] == idx) {
            current_graph.flags[ij] = 1;
            break;
          }
        }
      }
    }
  }
};

template <typename Distance,
          template<typename> class GraphUpdaterT>
struct LocalJoinWorker : public RcppParallel::Worker {
  const NeighborHeap& current_graph;
  const NeighborHeap& new_nbrs;
  const NeighborHeap& old_nbrs;
  const std::size_t n_nbrs;
  const std::size_t max_candidates;
  GraphUpdaterT<Distance>& graph_updater;

  LocalJoinWorker(
    const NeighborHeap& current_graph,
    const NeighborHeap& new_nbrs,
    const NeighborHeap& old_nbrs,
    GraphUpdaterT<Distance>& graph_updater
  ) :
    current_graph(current_graph),
    new_nbrs(new_nbrs),
    old_nbrs(old_nbrs),
    n_nbrs(current_graph.n_nbrs),
    max_candidates(new_nbrs.n_nbrs),
    graph_updater(graph_updater)
  {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      const std::size_t imaxc = i * max_candidates;
      for (std::size_t j = 0; j < max_candidates; j++) {
        std::size_t p = new_nbrs.idx[imaxc + j];
        if (p == NeighborHeap::npos()) {
          continue;
        }
        const std::size_t pnnbrs = p * n_nbrs;
        for (std::size_t k = j; k < max_candidates; k++) {
          std::size_t q = new_nbrs.idx[imaxc + k];
          if (q == NeighborHeap::npos()) {
            continue;
          }
          graph_updater.generate(current_graph, i, p, q, pnnbrs);
        }

        for (std::size_t k = 0; k < max_candidates; k++) {
          std::size_t q = old_nbrs.idx[imaxc + k];
          if (q == NeighborHeap::npos()) {
            continue;
          }
          graph_updater.generate(current_graph, i, p, q, pnnbrs);
        }
      }
    }
  }
};

struct UpdateWorker : RcppParallel::Worker {
  NeighborHeap& current_graph;
  std::vector<std::vector<Update>>& updates;
  std::size_t n_updates;
  tthread::mutex mutex;

  UpdateWorker(
    NeighborHeap& current_graph,
    std::vector<std::vector<Update>>& updates
  ) :
    current_graph(current_graph),
    updates(updates),
    n_updates(0)
  {}

  void operator()(std::size_t begin, std::size_t end) {
    std::size_t c = 0;
    for (std::size_t i = begin; i < end; i++) {
      const std::size_t n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        const auto& update = updates[i][j];
        {
          tthread::lock_guard<tthread::mutex> guard(mutex);
          c += current_graph.checked_push_pair(update.p, update.d, update.q, true);
        }
      }
    }
    {
      tthread::lock_guard<tthread::mutex> guard(mutex);
      n_updates += c;
    }
  }
};

template <typename Distance,
          typename Rand,
          typename Progress,
          template<typename> class GraphUpdaterT>
void nnd_parallel(
    NeighborHeap& current_graph,
    GraphUpdaterT<Distance>& graph_updater,
    const std::size_t max_candidates,
    const std::size_t n_iters,
    Rand& rand,
    Progress& progress,
    const double tol,
    std::size_t grain_size = 1,
    const std::size_t block_size = 16384,
    bool verbose = false
)
{
  const std::size_t n_points = current_graph.n_points;
  const auto n_blocks = (n_points / block_size) + 1;

  for (std::size_t n = 0; n < n_iters; n++) {
    NeighborHeap new_candidate_neighbors(n_points, max_candidates);
    NeighborHeap old_candidate_neighbors(n_points, max_candidates);

    LockingCandidatesWorker candidates_worker(
        current_graph,
        new_candidate_neighbors,
        old_candidate_neighbors);
    RcppParallel::parallelFor(0, n_points, candidates_worker, grain_size);

    NewCandidatesWorker new_candidates_worker(
        new_candidate_neighbors,
        current_graph);
    RcppParallel::parallelFor(0, n_points, new_candidates_worker, grain_size);

    std::size_t c = 0;
    for (std::size_t i = 0; i < n_blocks; i++) {
      const auto block_start = i * block_size;
      const auto block_end = std::min<std::size_t>(n_points, (i + 1) * block_size);

      LocalJoinWorker<Distance, GraphUpdaterT> local_join_worker(
          current_graph,
          new_candidate_neighbors,
          old_candidate_neighbors,
          graph_updater
      );
      RcppParallel::parallelFor(block_start, block_end, local_join_worker, grain_size);

      c += graph_updater.apply(current_graph);

      if (progress.check_interrupt()) {
        break;
      }
    }
    progress.update(n);

    if (static_cast<double>(c) <= tol) {
      if (verbose) {
        Rcpp::Rcout << "c = " << c << " tol = " << tol << std::endl;
      }
      progress.stopping_early();
      break;
    }
  }
  current_graph.deheap_sort();
}

#endif // RNND_PARALLEL_H

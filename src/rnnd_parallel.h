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
#include <unordered_set>

#include <Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include "heap.h"
#include "tauprng.h"
#include "nndescent.h"

struct CandidatesWorker : public RcppParallel::Worker {
  NeighborHeap& current_graph;
  const std::size_t n_points;
  const std::size_t n_nbrs;
  const std::size_t max_candidates;
  const double rho;
  NeighborHeap new_candidate_neighbors;
  NeighborHeap old_candidate_neighbors;
  tthread::mutex mutex;

  CandidatesWorker(
    NeighborHeap& current_graph,
    const std::size_t max_candidates,
    const double rho) :
    current_graph(current_graph),
    n_points(current_graph.n_points),
    n_nbrs(current_graph.n_nbrs),
    max_candidates(max_candidates),
    rho(rho),
    new_candidate_neighbors(n_points, max_candidates),
    old_candidate_neighbors(n_points, max_candidates)
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
        std::size_t idx = current_graph.index(ij);
        if (idx == NeighborHeap::npos || rand->unif() >= rho) {
          continue;
        }
        double d = rand->unif();
        bool isn = current_graph.flag(ij) == 1;
        if (isn) {
          std::size_t c = new_candidate_neighbors.checked_push(i, d, idx, isn);
          if (c > 0) {
            current_graph.flag(ij) = 0;
          }
        }
        else {
          old_candidate_neighbors.checked_push(i, d, idx, isn);
        }
      }
    }
  }
};

struct ReverseCandidatesWorker : public RcppParallel::Worker {
  const std::size_t n_points;
  const std::size_t max_candidates;
  const NeighborHeap& new_candidate_neighbors;
  const NeighborHeap& old_candidate_neighbors;
  NeighborHeap reverse_new_candidate_neighbors;
  NeighborHeap reverse_old_candidate_neighbors;

  ReverseCandidatesWorker(
    const NeighborHeap& new_candidate_neighbors,
    const NeighborHeap& old_candidate_neighbors) :
    n_points(new_candidate_neighbors.n_points),
    max_candidates(new_candidate_neighbors.n_nbrs),
    new_candidate_neighbors(new_candidate_neighbors),
    old_candidate_neighbors(old_candidate_neighbors),
    reverse_new_candidate_neighbors(new_candidate_neighbors),
    reverse_old_candidate_neighbors(old_candidate_neighbors)
  {}

  void operator()(std::size_t begin, std::size_t end) {
    std::size_t idx;
    double d;
    bool isn;
    for (std::size_t i = 0; i < n_points; i++) {
      std::size_t innbrs = i * max_candidates;
      for (std::size_t j = 0; j < max_candidates; j++) {
        std::size_t ij = innbrs + j;
        idx = new_candidate_neighbors.index(ij);
        if (idx != NeighborHeap::npos && idx >= begin && idx < end) {
          new_candidate_neighbors.df(ij, d, isn);
          reverse_new_candidate_neighbors.checked_push(idx, d, i, isn);
        }
        idx = old_candidate_neighbors.index(ij);
        if (idx != NeighborHeap::npos && idx >= begin && idx < end) {
          old_candidate_neighbors.df(ij, d, isn);
          reverse_old_candidate_neighbors.checked_push(idx, d, i, isn);
        }
      }
    }
  }
};

template <template<typename> class Heap,
          typename Distance>
struct NoNSearchWorker : public RcppParallel::Worker {
  Heap<Distance>& updated_graph;
  const NeighborHeap& new_nbrs;
  const NeighborHeap& old_nbrs;
  const std::size_t n_points;
  const std::size_t max_candidates;
  std::size_t n_updates;
  tthread::mutex mutex;

  NoNSearchWorker(
    Heap<Distance>& current_graph,
    const NeighborHeap& new_nbrs,
    const NeighborHeap& old_nbrs
  ) :
    updated_graph(current_graph),
    new_nbrs(new_nbrs),
    old_nbrs(old_nbrs),
    n_points(current_graph.neighbor_heap.n_points),
    max_candidates(new_nbrs.n_nbrs),
    n_updates(0)
  {}

  void operator()(std::size_t begin, std::size_t end) {
    std::size_t n_local_updates = 0;
    std::size_t p;
    std::size_t pnnbrs;

    std::unordered_set<std::size_t> seen(new_nbrs.n_points);
    for (std::size_t i = begin; i < end; i++) {
      std::size_t innbrs = i * max_candidates;
      for (std::size_t j = 0; j < max_candidates; j++) {
        std::size_t ij = innbrs + j;
        p = new_nbrs.index(ij);
        if (p != NeighborHeap::npos) {
          pnnbrs = p * max_candidates;
          for (std::size_t k = 0; k < max_candidates; k++) {
            std::size_t pk = pnnbrs + k;
            n_local_updates += try_add(i, new_nbrs.index(pk), seen);
            n_local_updates += try_add(i, old_nbrs.index(pk), seen);
          }
        }

        p = old_nbrs.index(ij);
        if (p == NeighborHeap::npos) {
          continue;
        }
        pnnbrs = p * max_candidates;
        for (std::size_t k = 0; k < max_candidates; k++) {
          n_local_updates += try_add(i, new_nbrs.index(pnnbrs + k), seen);
        }
      }
      seen.clear();
    }
    {
      tthread::lock_guard<tthread::mutex> guard(mutex);
      n_updates += n_local_updates;
    }
  }

  std::size_t try_add
    (
        std::size_t i,
        std::size_t q,
        std::unordered_set<std::size_t>& seen
    )
  {
    if (q == NeighborHeap::npos || !seen.emplace(q).second) {
      return 0;
    }
    return updated_graph.add_pair_asymm(i, q, true);
  }
};

template <template<typename> class Heap,
          typename Distance,
          typename Rand,
          typename Progress>
void nnd_parallel(
    Heap<Distance>& current_graph,
    const std::size_t max_candidates,
    const std::size_t n_iters,
    Rand& rand,
    Progress progress,
    const double rho,
    const double tol,
    std::size_t grain_size = 1,
    bool verbose = false
)
{
  const std::size_t n_points = current_graph.neighbor_heap.n_points;
  RandomWeight<Rand> weight_measure(rand);

  for (std::size_t n = 0; n < n_iters; n++) {
    if (verbose) {
      progress.iter(n, n_iters, current_graph.neighbor_heap);
    }

    RandomHeap<Rand> new_candidate_neighbors(weight_measure, n_points,
                                             max_candidates);
    RandomHeap<Rand> old_candidate_neighbors(weight_measure, n_points,
                                             max_candidates);

    build_candidates_full<Rand>(current_graph.neighbor_heap,
                                new_candidate_neighbors,
                                old_candidate_neighbors,
                                rho, rand);

    NeighborHeap& new_nbrs = new_candidate_neighbors.neighbor_heap;
    NeighborHeap& old_nbrs = old_candidate_neighbors.neighbor_heap;

    // CandidatesWorker candidates_worker(current_graph.neighbor_heap, max_candidates, rho);
    // RcppParallel::parallelFor(0, n_points, candidates_worker, grain_size);
    // current_graph.neighbor_heap = candidates_worker.current_graph;
    //
    // ReverseCandidatesWorker reverse_candidates_worker(
    //     candidates_worker.new_candidate_neighbors,
    //     candidates_worker.old_candidate_neighbors);
    // RcppParallel::parallelFor(0, n_points, reverse_candidates_worker, grain_size);

    NoNSearchWorker<Heap, Distance> non_search_worker(
        current_graph,
        // reverse_candidates_worker.reverse_new_candidate_neighbors,
        // reverse_candidates_worker.reverse_old_candidate_neighbors
        new_nbrs,
        old_nbrs
    );
    RcppParallel::parallelFor(0, n_points, non_search_worker, grain_size);
    current_graph = non_search_worker.updated_graph;
    progress.check_interrupt();

    const std::size_t c = non_search_worker.n_updates;
    if (static_cast<double>(c) <= tol) {
      if (verbose) {
        progress.converged(c, tol);
      }
      break;
    }
  }
  current_graph.neighbor_heap.deheap_sort();
}

#endif // RNND_PARALLEL_H

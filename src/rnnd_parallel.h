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
#include "tauprng.h"


template <template<typename> class Heap,
          typename Distance>
struct NNDWorker : public RcppParallel::Worker {
  const Heap<Distance>& current_graph;
  Heap<Distance> updated_graph;
  const std::size_t n_points;
  const std::size_t n_nbrs;
  const double rho;
  std::size_t n_updates;
  tthread::mutex mutex;

  NNDWorker(const Heap<Distance>& current_graph, const double rho) :
    current_graph(current_graph),
    updated_graph(current_graph),
    n_points(current_graph.neighbor_heap.n_points),
    n_nbrs(current_graph.neighbor_heap.n_nbrs),
    rho(rho),
    n_updates(0)
    {}

  void operator()(std::size_t begin, std::size_t end) {
    std::unique_ptr<TauRand> rand(nullptr);
    // Each window gets its own PRNG state, to prevent locking inside the loop.
    {
      tthread::lock_guard<tthread::mutex> guard(mutex);
      rand.reset(new TauRand());
    }
    std::size_t n_local_updates = 0;
    std::unordered_set<std::size_t> seen(static_cast<std::size_t>(rho * n_nbrs * n_nbrs));
    for (std::size_t i = begin; i < end; i++) {
      for (std::size_t j = 0; j < n_nbrs; j++) {
        std::size_t p = current_graph.neighbor_heap.index(i, j);
        if (p == NeighborHeap::npos) {
          continue;
        }
        for (std::size_t k = 0; k < n_nbrs; k++) {
          std::size_t q = current_graph.neighbor_heap.index(p, k);
          if (i == q || q == NeighborHeap::npos || rand->unif() >= rho) {
            continue;
          }
          if (seen.find(q) == seen.end()) {
            seen.insert(q);
            {
              tthread::lock_guard<tthread::mutex> guard(mutex);
              n_local_updates += updated_graph.add_pair_asymm(i, q, true);
            }
          }
        }
      }
      seen.clear();
    }
    {
      tthread::lock_guard<tthread::mutex> guard(mutex);
      n_updates += n_local_updates;
    }
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
  const std::size_t n_nbrs = current_graph.neighbor_heap.n_nbrs;

  for (std::size_t n = 0; n < n_iters; n++) {
    if (verbose) {
      progress.iter(n, n_iters, current_graph.neighbor_heap);
    }
    NNDWorker<Heap, Distance> worker(current_graph, rho);
    RcppParallel::parallelFor(0, n_points, worker, grain_size);
    current_graph = worker.updated_graph;

    // resymmetrize
    for (std::size_t i = 0; i < n_points; i++) {
      for (std::size_t j = 0; j < n_nbrs; j++) {
        std::size_t p = current_graph.neighbor_heap.index(i, j);
        if (p == NeighborHeap::npos) {
          continue;
        }
        current_graph.add_pair_asymm(p, i, true);
      }
    }

    progress.check_interrupt();

    const std::size_t c = worker.n_updates;
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

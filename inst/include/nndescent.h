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

#ifndef RNND_NNDESCENT_H
#define RNND_NNDESCENT_H

#include <unordered_set>

#include "arrayheap.h"
#include "heap.h"

// Builds the general neighbors of each object, keeping up to max_candidates
// per object. The objects are associated with a random number rather than
// the true distance, and hence are stored in random order.
template <typename Rand>
NeighborHeap build_candidates(
    NeighborHeap& current_graph,
    std::size_t max_candidates,
    const std::size_t npoints,
    const std::size_t nnbrs,
    Rand& rand)
{

  RandomWeight<Rand> weight_measure(rand);
  RandomHeap<Rand> candidate_neighbors(weight_measure, npoints, max_candidates);

  for (std::size_t i = 0; i < npoints; i++) {
    for (std::size_t j = 0; j < nnbrs; j++) {
      if (current_graph.index(i, j) == NeighborHeap::npos()) {
        continue;
      }
      std::size_t idx = current_graph.index(i, j);
      bool isn = current_graph.flags(i, j) == 1;

      candidate_neighbors.add_pair(i, idx, isn);
      // incremental search: mark this object false to indicate it has
      // participated in the local join
      current_graph.flags(i, j) = 0;
    }
  }
  return candidate_neighbors.neighbor_heap;
}

// This corresponds to the construction of new, old, new' and old' in
// Algorithm 2, with some minor differences:
// 1. old' and new' (the reverse candidates) are build at the same time as old
// and new respectively, based on the fact that if j is a candidate of new[i],
// then i is a reverse candidate of new[j]. This saves on building the entire
// reverse candidates list and then down-sampling.
// 2. Not all old members of current KNN are placed in old, nor are rho * K
// new candidates sampled. Instead, rho * K total candidates are sampled from
// the KNN and these are assigned into old and new based on their flag value,
// i.e. the total number of sampled candidates (old + new) is rho * K.
template <typename Rand>
void build_candidates_full(
    NeighborHeap& current_graph,
    RandomHeap<Rand>& new_candidate_neighbors,
    RandomHeap<Rand>& old_candidate_neighbors,
    double rho,
    Rand& rand)
{
  const std::size_t n_points = current_graph.n_points;
  const std::size_t n_nbrs = current_graph.n_nbrs;

  for (std::size_t i = 0; i < n_points; i++) {
    std::size_t innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      std::size_t ij = innbrs + j;
      std::size_t idx = current_graph.index(ij);
      if (idx == NeighborHeap::npos() || rand.unif() >= rho) {
        continue;
      }
      bool isn = current_graph.flag(ij) == 1;
      if (isn) {
        std::size_t c = new_candidate_neighbors.add_pair(i, idx, isn);
        if (c > 0) {
          current_graph.flag(ij) = 0;
        }
      }
      else {
        old_candidate_neighbors.add_pair(i, idx, isn);
      }
    }
  }
}

// Closer to the basic NNDescent algorithm (#1 in the paper)
template <template<typename> class Heap,
          typename Distance,
          typename Rand,
          typename Progress>
void nnd_basic(
    Heap<Distance>& current_graph,
    const std::size_t max_candidates,
    const std::size_t n_iters,
    const std::size_t npoints,
    const std::size_t nnbrs,
    Rand& rand,
    Progress progress,
    const double rho,
    const double tol,
    bool verbose)
{
  for (std::size_t n = 0; n < n_iters; n++) {
    if (verbose) {
      progress.iter(n, n_iters, current_graph.neighbor_heap);
    }

    NeighborHeap candidate_neighbors = build_candidates<Rand>(
      current_graph.neighbor_heap, max_candidates, npoints, nnbrs, rand);

    std::size_t c = 0;
    for (std::size_t i = 0; i < npoints; i++) {
      // local join: for each pair of points p, q in the general neighbor list
      // of i, calculate dist(p, q) and update neighbor list of p and q
      // NB: the neighbor list of i is unchanged by this operation
      for (std::size_t j = 0; j < max_candidates; j++) {
        std::size_t p = candidate_neighbors.index(i, j);
        if (p == NeighborHeap::npos() || rand.unif() < rho) {
          // only sample rho * max_candidates of the general neighbors
          continue;
        }

        for (std::size_t k = 0; k < max_candidates; k++) {
          std::size_t q = candidate_neighbors.index(i, k);
          if (q == NeighborHeap::npos() ||
              (candidate_neighbors.flags(i, j) == 0 &&
               candidate_neighbors.flags(i, k) == 0))
          {
            // incremental search: two objects are only compared if at least
            // one of them is new
            continue;
          }
          c += current_graph.add_pair(p, q, true);
        }
      }
      progress.check_interrupt();
    }
    if (static_cast<double>(c) <= tol) {
      if (verbose) {
        progress.converged(c, tol);
      }
      break;
    }
  }
  current_graph.neighbor_heap.deheap_sort();
}

// Closer to the NNDescentFull algorithm (#2 in the paper)
template <template<typename> class Heap,
          typename Distance,
          typename Rand,
          typename Progress>
void nnd_full(
    Heap<Distance>& current_graph,
    const std::size_t max_candidates,
    const std::size_t n_iters,
    Rand& rand,
    Progress& progress,
    const double rho,
    const double tol,
    bool verbose)
{
  RandomWeight<Rand> weight_measure(rand);
  const std::size_t n_points = current_graph.neighbor_heap.n_points;

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

    std::size_t c = local_join(current_graph, new_nbrs, old_nbrs, n_points,
                               max_candidates, progress);

    progress.check_interrupt();
    if (static_cast<double>(c) <= tol) {
      if (verbose) {
        progress.converged(c, tol);
      }
      break;
    }
  }
  current_graph.neighbor_heap.deheap_sort();
}

// Local join update: instead of updating item i with the neighbors of the
// candidates of i, explore pairs (p, q) of candidates and treat q as a
// candidate for p, and vice versa.
template <template<typename> class Heap,
          typename Distance,
          typename Progress>
std::size_t local_join(
    Heap<Distance>& current_graph,
    const NeighborHeap& new_nbrs,
    const NeighborHeap& old_nbrs,
    const std::size_t n_points,
    const std::size_t max_candidates,
    Progress& progress
  )
{
  std::size_t c = 0;
  for (std::size_t i = 0; i < n_points; i++) {
    for (std::size_t j = 0; j < max_candidates; j++) {
      std::size_t p = new_nbrs.index(i, j);
      if (p == NeighborHeap::npos()) {
        continue;
      }
      for (std::size_t k = j; k < max_candidates; k++) {
        std::size_t q = new_nbrs.index(i, k);
        if (q == NeighborHeap::npos()) {
          continue;
        }
        c += current_graph.add_pair(p, q, true);
      }

      for (std::size_t k = 0; k < max_candidates; k++) {
        std::size_t q = old_nbrs.index(i, k);
        if (q == NeighborHeap::npos()) {
          continue;
        }
        c += current_graph.add_pair(p, q, true);
      }
    }
    progress.check_interrupt();
  }
  return c;
}


// Attempt to add q to i's knn, and vice versa
// If q is invalid or already seen or q > i, the addition is not attempted
// Used by neighbor-of-neighbor search
template <template<typename> class Heap,
          typename Distance>
std::size_t try_add(
    Heap<Distance>& current_graph,
    std::size_t i,
    std::size_t q,
    std::unordered_set<std::size_t>& seen
)
{
  if (q > i || q == NeighborHeap::npos() || !seen.emplace(q).second) {
    return 0;
  }
  return current_graph.add_pair(i, q, true);
}


// Use neighbor-of-neighbor search rather than local join to update the kNN.
// To implement incremental search, for a new candidate, both its new and
// old candidates will be searched. For an old candidate, only the new
// candidates are used.
template <template<typename> class Heap,
          typename Distance,
          typename Progress>
std::size_t non_join(
    Heap<Distance>& current_graph,
    const NeighborHeap& new_nbrs,
    const NeighborHeap& old_nbrs,
    const std::size_t n_points,
    const std::size_t max_candidates,
    Progress& progress
  )
{
  std::size_t c = 0;
  std::size_t p = 0;
  std::size_t q = 0;
  std::unordered_set<std::size_t> seen;
  for (std::size_t i = 0; i < n_points; i++) {
    for (std::size_t j = 0; j < max_candidates; j++) {
      p = new_nbrs.index(i, j);
      if (p != NeighborHeap::npos()) {
        for (std::size_t k = 0; k < max_candidates; k++) {
          q = new_nbrs.index(p, k);
          c += try_add(current_graph, i, q, seen);
          q = old_nbrs.index(p, k);
          c += try_add(current_graph, i, q, seen);
        }
      }

      p = old_nbrs.index(i, j);
      if (p == NeighborHeap::npos()) {
        continue;
      }
      for (std::size_t k = 0; k < max_candidates; k++) {
        q = new_nbrs.index(p, k);
        c += try_add(current_graph, i, q, seen);
      }
    }
    progress.check_interrupt();
    seen.clear();
  }
  return c;
}

#endif // RNND_NNDESCENT_H

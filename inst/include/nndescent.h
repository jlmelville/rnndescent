// BSD 2-Clause License
//
// Copyright 2019 James Melville
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

#ifndef NND_NNDESCENT_H
#define NND_NNDESCENT_H

#include <unordered_set>

#include "heap.h"

// This corresponds to the construction of new, old, new' and old' in
// Algorithm 2, with some minor differences:
// 1. old' and new' (the reverse candidates) are build at the same time as old
// and new respectively, based on the fact that if j is a candidate of new[i],
// then i is a reverse candidate of new[j]. This saves on building the entire
// reverse candidates list and then down-sampling.
// 2. Not all old members of current KNN are retained in the old candidates
// list, nor are rho * K new candidates sampled. Instead, the current members
// of the KNN are assigned into old and new based on their flag value, with the
// size of the final candidate list controlled by max_candidates.
template <typename Rand>
void build_candidates_full(NeighborHeap &current_graph, Rand &rand,
                           NeighborHeap &new_candidate_neighbors,
                           NeighborHeap &old_candidate_neighbors) {
  const std::size_t n_points = current_graph.n_points;
  const std::size_t n_nbrs = current_graph.n_nbrs;

  for (std::size_t i = 0; i < n_points; i++) {
    std::size_t innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      std::size_t ij = innbrs + j;
      std::size_t idx = current_graph.idx[ij];

      double d = rand.unif();
      char isn = current_graph.flags[ij];
      if (isn == 1) {
        new_candidate_neighbors.checked_push_pair(i, d, idx, isn);
      } else {
        old_candidate_neighbors.checked_push_pair(i, d, idx, isn);
      }
    }
  }

  // mark any neighbor in the current graph that was retained in the new
  // candidates as true
  const std::size_t max_candidates = new_candidate_neighbors.n_nbrs;
  for (std::size_t i = 0; i < n_points; i++) {
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

// Pretty close to the NNDescentFull algorithm (#2 in the paper)
template <template <typename> class GraphUpdater, typename Distance,
          typename Rand, typename Progress>
void nnd_full(NeighborHeap &current_graph,
              GraphUpdater<Distance> &graph_updater,
              const std::size_t max_candidates, const std::size_t n_iters,
              Rand &rand, Progress &progress, const double tol, bool verbose) {
  const std::size_t n_points = current_graph.n_points;

  for (std::size_t n = 0; n < n_iters; n++) {
    NeighborHeap new_nbrs(n_points, max_candidates);
    NeighborHeap old_nbrs(n_points, max_candidates);

    build_candidates_full(current_graph, rand, new_nbrs, old_nbrs);
    std::size_t c = local_join(current_graph, graph_updater, new_nbrs, old_nbrs,
                               n_points, max_candidates, progress);

    progress.update(n);
    if (progress.check_interrupt()) {
      break;
    }
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

// Local join update: instead of updating item i with the neighbors of the
// candidates of i, explore pairs (p, q) of candidates and treat q as a
// candidate for p, and vice versa.
template <template <typename> class GraphUpdater, typename Distance,
          typename Progress>
std::size_t local_join(NeighborHeap &current_graph,
                       GraphUpdater<Distance> &graph_updater,
                       const NeighborHeap &new_nbrs,
                       const NeighborHeap &old_nbrs, const std::size_t n_points,
                       const std::size_t max_candidates, Progress &progress) {
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
        c += graph_updater.generate_and_apply(p, q);
      }

      for (std::size_t k = 0; k < max_candidates; k++) {
        std::size_t q = old_nbrs.index(i, k);
        if (q == NeighborHeap::npos()) {
          continue;
        }
        c += graph_updater.generate_and_apply(p, q);
      }
    }
  }
  return c;
}

// Attempt to add q to i's knn, and vice versa
// If q is invalid or already seen or q > i, the addition is not attempted
// Used by neighbor-of-neighbor search
template <template <typename> class Heap, typename Distance>
std::size_t try_add(Heap<Distance> &current_graph, std::size_t i, std::size_t q,
                    std::unordered_set<std::size_t> &seen) {
  if (q > i || q == NeighborHeap::npos() || !seen.emplace(q).second) {
    return 0;
  }
  return current_graph.add_pair(i, q, 1);
}

// Use neighbor-of-neighbor search rather than local join to update the kNN.
// To implement incremental search, for a new candidate, both its new and
// old candidates will be searched. For an old candidate, only the new
// candidates are used.
template <template <typename> class Heap, typename Distance, typename Progress>
std::size_t non_join(Heap<Distance> &current_graph,
                     const NeighborHeap &new_nbrs, const NeighborHeap &old_nbrs,
                     const std::size_t n_points,
                     const std::size_t max_candidates, Progress &progress) {
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

#endif // NND_NNDESCENT_H

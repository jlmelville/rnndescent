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

#ifndef TDOANN_NNDESCENT_H
#define TDOANN_NNDESCENT_H

#include "heap.h"

namespace tdoann {
// mark any neighbor in the current graph that was retained in the new
// candidates as false
void flag_retained_new_candidates(NeighborHeap &current_graph,
                                  const NeighborHeap &new_candidate_neighbors,
                                  std::size_t begin, std::size_t end) {
  const std::size_t n_nbrs = current_graph.n_nbrs;
  for (std::size_t i = begin; i < end; i++) {
    std::size_t innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      std::size_t ij = innbrs + j;
      std::size_t idx = current_graph.idx[ij];

      if (new_candidate_neighbors.contains(i, idx)) {
        current_graph.flags[ij] = 0;
      }
    }
  }
}

// overload for serial processing case which does entire graph in one chunk
void flag_retained_new_candidates(NeighborHeap &current_graph,
                                  const NeighborHeap &new_candidate_neighbors) {
  flag_retained_new_candidates(current_graph, new_candidate_neighbors, 0,
                               current_graph.n_points);
}

// This corresponds to the construction of new, old, new' and old' in
// Algorithm 2, with some minor differences:
// 1. old' and new' (the reverse candidates) are build at the same time as old
// and new respectively, based on the fact that if j is a candidate of new[i],
// then i is a reverse candidate of new[j]. This saves on building the entire
// reverse candidates list and then down-sampling.
// 2. Not all old members of current KNN are retained in the old candidates
// list, nor are rho * K new candidates sampled. Instead, the current members
// of the KNN are assigned into old and new based on their flag value, with the
// size of the final candidate list controlled by the maximum size of
// the candidates neighbors lists.
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
  flag_retained_new_candidates(current_graph, new_candidate_neighbors);
}

bool is_converged(std::size_t n_updates, double tol) {
  return static_cast<double>(n_updates) <= tol;
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
    if (is_converged(c, tol)) {
      progress.converged(c, tol);
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

template <template <typename> class GraphUpdater, typename Distance,
          typename Rand, typename Progress>
void nnd_query(NeighborHeap &current_graph,
               GraphUpdater<Distance> &graph_updater,
               const std::vector<std::size_t> &reference_idx,
               const std::size_t max_candidates, const std::size_t n_iters,
               Rand &rand, Progress &progress, const double tol, bool verbose) {
  const std::size_t n_points = current_graph.n_points;
  const std::size_t n_nbrs = current_graph.n_nbrs;
  // if the candidate heap size is as large or larger than the number of
  // neighbors then we definitely know anything that is added won't be evicted
  // due to size, so we can mark at the same time as we add
  const bool flag_on_add = max_candidates >= n_nbrs;

  for (std::size_t n = 0; n < n_iters; n++) {
    NeighborHeap new_nbrs(n_points, max_candidates);

    build_query_candidates(current_graph, rand, new_nbrs, flag_on_add);
    if (!flag_on_add) {
      // Can't be sure all candidates that were pushed were retained, so we
      // check now: mark any neighbor in the current graph that was retained in
      // the new candidates
      flag_retained_new_candidates(current_graph, new_nbrs);
    }

    std::size_t c = non_search_query(current_graph, graph_updater, new_nbrs,
                                     reference_idx, max_candidates, progress);

    progress.update(n);
    if (progress.check_interrupt()) {
      break;
    }
    if (is_converged(c, tol)) {
      progress.converged(c, tol);
      break;
    }
  }
  current_graph.deheap_sort();
}

template <typename Rand>
void build_query_candidates(NeighborHeap &current_graph, Rand &rand,
                            NeighborHeap &new_candidate_neighbors,
                            std::size_t begin, std::size_t end,
                            bool flag_on_add) {
  const std::size_t n_nbrs = current_graph.n_nbrs;
  for (std::size_t i = begin; i < end; i++) {
    std::size_t innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      std::size_t ij = innbrs + j;
      char isn = current_graph.flags[ij];
      if (isn == 1) {
        double d = rand.unif();
        new_candidate_neighbors.checked_push(i, d, current_graph.idx[ij], isn);
        if (flag_on_add) {
          current_graph.flags[ij] = 0;
        }
      }
    }
  }
}

template <typename Rand>
void build_query_candidates(NeighborHeap &current_graph, Rand &rand,
                            NeighborHeap &new_candidate_neighbors,
                            bool flag_on_add) {
  const std::size_t n_points = current_graph.n_points;
  build_query_candidates(current_graph, rand, new_candidate_neighbors, 0,
                         n_points, flag_on_add);
}

// Use neighbor-of-neighbor search rather than local join to update the kNN.
template <template <typename> class GraphUpdater, typename Distance,
          typename Progress>
std::size_t non_search_query(NeighborHeap &current_graph,
                             GraphUpdater<Distance> &graph_updater,
                             const NeighborHeap &new_nbrs,
                             const std::vector<std::size_t> &reference_idx,
                             const std::size_t max_candidates,
                             const std::size_t begin, const std::size_t end,
                             Progress &progress) {
  std::size_t c = 0;
  std::size_t ref_idx = 0;
  std::size_t nbr_ref_idx = 0;
  const std::size_t n_nbrs = current_graph.n_nbrs;
  std::unordered_set<std::size_t> seen(n_nbrs);

  for (std::size_t query_idx = begin; query_idx < end; query_idx++) {
    for (std::size_t j = 0; j < max_candidates; j++) {
      ref_idx = new_nbrs.index(query_idx, j);
      if (ref_idx == NeighborHeap::npos()) {
        continue;
      }
      const std::size_t rnidx = ref_idx * n_nbrs;
      for (std::size_t k = 0; k < n_nbrs; k++) {
        nbr_ref_idx = reference_idx[rnidx + k];
        if (nbr_ref_idx == NeighborHeap::npos() ||
            !seen.emplace(nbr_ref_idx).second) {
          continue;
        }
        c += graph_updater.generate_and_apply(query_idx, nbr_ref_idx);
      }
    }
    progress.check_interrupt();
    seen.clear();
  }
  return c;
}

template <template <typename> class GraphUpdater, typename Distance,
          typename Progress>
std::size_t non_search_query(NeighborHeap &current_graph,
                             GraphUpdater<Distance> &graph_updater,
                             const NeighborHeap &new_nbrs,
                             const std::vector<std::size_t> &reference_idx,
                             const std::size_t max_candidates,
                             Progress &progress) {
  const std::size_t n_points = current_graph.n_points;
  return non_search_query(current_graph, graph_updater, new_nbrs, reference_idx,
                          max_candidates, 0, n_points, progress);
}
} // namespace tdoann
#endif // TDOANN_NNDESCENT_H

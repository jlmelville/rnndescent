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

#include <sstream>

#include "heap.h"
#include "nndprogress.h"

namespace tdoann {
// mark any neighbor in the current graph that was retained in the new
// candidates as false
template <typename DistOut, typename Idx>
void flag_retained_new_candidates(NNDHeap<DistOut, Idx> &current_graph,
                                  const NNHeap<DistOut, Idx> &new_nbrs,
                                  std::size_t begin, std::size_t end) {
  const std::size_t n_nbrs = current_graph.n_nbrs;
  for (std::size_t i = begin, idx_offset = 0; i < end;
       i++, idx_offset += n_nbrs) {
    for (std::size_t j = 0, idx_ij = idx_offset; j < n_nbrs; j++, idx_ij++) {
      if (new_nbrs.contains(i, current_graph.idx[idx_ij])) {
        current_graph.flags[idx_ij] = 0;
      }
    }
  }
}

// overload for serial processing case which does entire graph in one chunk
template <typename DistOut, typename Idx>
void flag_retained_new_candidates(
    NNDHeap<DistOut, Idx> &current_graph,
    const NNHeap<DistOut, Idx> &new_candidate_neighbors) {
  flag_retained_new_candidates(current_graph, new_candidate_neighbors, 0,
                               current_graph.n_points);
}

// This corresponds to the construction of new, old, new' and old' in
// Algorithm 2, with some minor differences:
// 1. old' and new' (the reverse candidates) are built at the same time as old
// and new respectively, based on the fact that if j is a candidate of new[i],
// then i is a reverse candidate of new[j]. This saves on building the entire
// reverse candidates list and then down-sampling.
// 2. Not all old members of current KNN are retained in the old candidates
// list, nor are rho * K new candidates sampled. Instead, the current members
// of the KNN are assigned into old and new based on their flag value, with the
// size of the final candidate list controlled by the maximum size of
// the candidates neighbors lists.
template <typename DistOut, typename Idx, typename Rand>
void build_candidates_full(NNDHeap<DistOut, Idx> &current_graph,
                           NNHeap<DistOut, Idx> &new_nbrs,
                           decltype(new_nbrs) &old_nbrs, Rand &rand) {
  const std::size_t n_points = current_graph.n_points;
  const std::size_t n_nbrs = current_graph.n_nbrs;

  for (std::size_t i = 0, idx_offset = 0; i < n_points;
       i++, idx_offset += n_nbrs) {
    for (std::size_t j = 0, idx_ij = idx_offset; j < n_nbrs; j++, idx_ij++) {
      auto &nbrs = current_graph.flags[idx_ij] == 1 ? new_nbrs : old_nbrs;
      if (current_graph.idx[idx_ij] == nbrs.npos()) {
        continue;
      }
      auto rand_weight = rand.unif();
      nbrs.checked_push_pair(i, rand_weight, current_graph.idx[idx_ij]);
    }
  }
  flag_retained_new_candidates(current_graph, new_nbrs);
}

// Pretty close to the NNDescentFull algorithm (#2 in the paper)
template <template <typename> class GraphUpdater, typename Distance,
          typename Rand>
void nnd_build(GraphUpdater<Distance> &graph_updater,
               std::size_t max_candidates, std::size_t n_iters, double delta,
               Rand &rand, NNDProgressBase &progress) {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  auto &nn_heap = graph_updater.current_graph;
  const std::size_t n_points = nn_heap.n_points;
  const double tol = delta * nn_heap.n_nbrs * n_points;

  for (std::size_t iter = 0; iter < n_iters; iter++) {
    NNHeap<DistOut, Idx> new_nbrs(n_points, max_candidates);
    decltype(new_nbrs) old_nbrs(n_points, max_candidates);

    build_candidates_full(nn_heap, new_nbrs, old_nbrs, rand);
    std::size_t num_updated =
        local_join(graph_updater, new_nbrs, old_nbrs, progress);

    bool stop_early = nnd_should_stop(progress, nn_heap, num_updated, tol);
    if (stop_early) {
      break;
    }
  }
}

// Local join update: instead of updating item i with the neighbors of the
// candidates of i, explore pairs (p, q) of candidates and treat q as a
// candidate for p, and vice versa.
template <template <typename> class GraphUpdater, typename Distance>
auto local_join(
    GraphUpdater<Distance> &graph_updater,
    const NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    decltype(new_nbrs) &old_nbrs, NNDProgressBase &progress) -> std::size_t {

  using Idx = typename Distance::Index;
  const auto n_points = new_nbrs.n_points;
  const auto max_candidates = new_nbrs.n_nbrs;
  progress.set_n_blocks(n_points);
  std::size_t num_updates = 0;
  for (Idx i = 0; i < n_points; i++) {
    for (Idx j = 0; j < max_candidates; j++) {
      auto p_nbr = new_nbrs.index(i, j);
      if (p_nbr == new_nbrs.npos()) {
        continue;
      }
      for (Idx k = j; k < max_candidates; k++) {
        auto new_nbr = new_nbrs.index(i, k);
        if (new_nbr == new_nbrs.npos()) {
          continue;
        }
        num_updates += graph_updater.update(p_nbr, new_nbr);
      }

      for (Idx k = 0; k < max_candidates; k++) {
        auto old_nbr = old_nbrs.index(i, k);
        if (old_nbr == old_nbrs.npos()) {
          continue;
        }
        num_updates += graph_updater.update(p_nbr, old_nbr);
      }
    }
    if (progress.check_interrupt()) {
      break;
    }
    progress.block_finished();
  }
  return num_updates;
}
} // namespace tdoann
#endif // TDOANN_NNDESCENT_H

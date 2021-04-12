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

#include <queue>
#include <utility>

#include "bitvec.h"
#include "graphupdate.h"
#include "heap.h"
#include "hub.h"
#include "nngraph.h"
#include "progress.h"

namespace tdoann {
// mark any neighbor in the current graph that was retained in the new
// candidates as false
template <typename DistOut, typename Idx>
void flag_retained_new_candidates(NNDHeap<DistOut, Idx> &current_graph,
                                  const NNHeap<DistOut, Idx> &new_nbrs,
                                  std::size_t begin, std::size_t end) {
  const std::size_t n_nbrs = current_graph.n_nbrs;
  std::size_t innbrs = 0;
  std::size_t ij = 0;
  for (auto i = begin; i < end; i++) {
    innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      ij = innbrs + j;
      if (new_nbrs.contains(i, current_graph.idx[ij])) {
        current_graph.flags[ij] = 0;
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

template <typename NbrHeap>
auto koccur_weights(const NbrHeap &current_graph) -> std::vector<double> {
  auto koccurs =
      reverse_nbr_counts(current_graph.idx, current_graph.n_points, false);
  auto norm = 1.0 / static_cast<double>(current_graph.n_nbrs - 1);
  std::vector<double> weights(koccurs.size());
  std::transform(koccurs.begin(), koccurs.end(), weights.begin(),
                 [&norm](typename NbrHeap::Index ko) { return ko * norm; });
  return weights;
}

template <typename DistOut, typename Idx>
void build_candidates_full_weighted(NNDHeap<DistOut, Idx> &current_graph,
                                    NNHeap<DistOut, Idx> &new_nbrs,
                                    decltype(new_nbrs) &old_nbrs) {
  const std::size_t n_points = current_graph.n_points;
  const std::size_t n_nbrs = current_graph.n_nbrs;
  std::size_t innbrs = 0;
  std::size_t ij = 0;

  std::vector<double> weights = koccur_weights(current_graph);

  for (std::size_t i = 0; i < n_points; i++) {
    innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      ij = innbrs + j;
      auto &nbrs = current_graph.flags[ij] == 1 ? new_nbrs : old_nbrs;
      if (current_graph.idx[ij] == nbrs.npos()) {
        continue;
      }
      auto nbr = current_graph.idx[ij];
      nbrs.checked_push(i, weights[nbr], nbr);
      if (nbr != i) {
        nbrs.checked_push(nbr, weights[i], i);
      }
    }
  }
  flag_retained_new_candidates(current_graph, new_nbrs);
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
  std::size_t innbrs = 0;
  std::size_t ij = 0;

  for (std::size_t i = 0; i < n_points; i++) {
    innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      ij = innbrs + j;
      auto &nbrs = current_graph.flags[ij] == 1 ? new_nbrs : old_nbrs;
      if (current_graph.idx[ij] == nbrs.npos()) {
        continue;
      }
      auto d = rand.unif();
      nbrs.checked_push_pair(i, d, current_graph.idx[ij]);
    }
  }
  flag_retained_new_candidates(current_graph, new_nbrs);
}

auto is_converged(std::size_t n_updates, double tol) -> bool {
  return static_cast<double>(n_updates) <= tol;
}

// Pretty close to the NNDescentFull algorithm (#2 in the paper)
template <template <typename> class GraphUpdater, typename Distance,
          typename Progress, typename Rand>
void nnd_build(GraphUpdater<Distance> &graph_updater,
               std::size_t max_candidates, std::size_t n_iters, double delta,
               Rand &rand, Progress &progress, bool weighted = false) {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  auto &nn_heap = graph_updater.current_graph;
  const std::size_t n_points = nn_heap.n_points;
  const double tol = delta * nn_heap.n_nbrs * n_points;

  for (std::size_t n = 0; n < n_iters; n++) {
    NNHeap<DistOut, Idx> new_nbrs(n_points, max_candidates);
    decltype(new_nbrs) old_nbrs(n_points, max_candidates);

    if (weighted) {
      build_candidates_full_weighted(nn_heap, new_nbrs, old_nbrs);
    } else {
      build_candidates_full(nn_heap, new_nbrs, old_nbrs, rand);
    }
    std::size_t c = local_join(graph_updater, new_nbrs, old_nbrs, progress);

    TDOANN_ITERFINISHED();
    progress.heap_report(nn_heap);
    TDOANN_CHECKCONVERGENCE();
  }
}

// Local join update: instead of updating item i with the neighbors of the
// candidates of i, explore pairs (p, q) of candidates and treat q as a
// candidate for p, and vice versa.
template <template <typename> class GraphUpdater, typename Distance,
          typename Progress>
auto local_join(
    GraphUpdater<Distance> &graph_updater,
    const NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    decltype(new_nbrs) &old_nbrs, Progress &progress) -> std::size_t {

  using Idx = typename Distance::Index;
  const auto n_points = new_nbrs.n_points;
  const auto max_candidates = new_nbrs.n_nbrs;
  progress.set_n_blocks(n_points);
  std::size_t c = 0;
  for (Idx i = 0; i < n_points; i++) {
    for (Idx j = 0; j < max_candidates; j++) {
      auto p = new_nbrs.index(i, j);
      if (p == new_nbrs.npos()) {
        continue;
      }
      for (Idx k = j; k < max_candidates; k++) {
        auto q = new_nbrs.index(i, k);
        if (q == new_nbrs.npos()) {
          continue;
        }
        c += graph_updater.generate_and_apply(p, q);
      }

      for (Idx k = 0; k < max_candidates; k++) {
        auto q = old_nbrs.index(i, k);
        if (q == old_nbrs.npos()) {
          continue;
        }
        c += graph_updater.generate_and_apply(p, q);
      }
    }
    TDOANN_BLOCKFINISHED();
  }
  return c;
}

// Create the neighbor list for each reference item, i.e. the
// neighbor-of-neighbors that are used when doing NND queries
template <typename DistOut, typename Idx>
void build_query_candidates(const std::vector<Idx> &reference_idx,
                            const std::vector<DistOut> &reference_dist,
                            std::size_t n_nbrs,
                            NNHeap<DistOut, Idx> &query_candidates,
                            std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end; i++) {
    std::size_t innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      auto nbr = reference_idx[innbrs + j];
      if (nbr == query_candidates.npos()) {
        continue;
      }
      query_candidates.checked_push(i, reference_dist[innbrs + j], nbr);
    }
  }
}

template <typename DistOut, typename Idx>
void build_query_candidates(const std::vector<Idx> &reference_idx,
                            const std::vector<DistOut> &reference_dist,
                            std::size_t n_nbrs,
                            NNHeap<DistOut, Idx> &query_candidates) {
  build_query_candidates(reference_idx, reference_dist, n_nbrs,
                         query_candidates, 0, query_candidates.n_points);
}

// No local join available when querying because there's no symmetry in the
// distances to take advantage of, so this is similar to algo #1 in the NND
// paper with the following differences:
// 1. The existing "reference" knn graph doesn't get updated during a query,
//    so each query item has no reverse neighbors, only the "forward" neighbors,
//    i.e. the knn.
// 2. The members of the query knn are from the reference knn and those *do*
//    have reverse neighbors, but from testing, there is a noticeable
//    difference for datasets with hubs, where only looking at the forward
//    neighbors gives better results (other datasets are unaffected). Perhaps
//    this is due to a lack of diversity in the general neighbor list:
//    increasing max_candidates for ameliorates the difference. The overall
//    search is: for each neighbor in the "forward" neighbors (the current
//    query knn), try each of its forward general neighbors.
// 3. Because the reference knn doesn't get updated during the query, the
//    reference general neighbor list only needs to get built once.
// 4. Incremental search is also simplified. Each member of the query knn
//    is marked as new when it's searched and because the update isn't
//    symmetric, we can operate on the graph directly. And because of the
//    static nature of the reference general neighbors, we don't need to keep
//    track of old neighbors: if a neighbor is "new" we search all its general
//    neighbors; otherwise, we don't search it at all because we must have
//    already tried those candidates.
template <typename Distance, typename Progress>
void nnd_query(
    const std::vector<typename Distance::Index> &reference_idx,
    std::size_t n_reference_nbrs,
    const std::vector<typename Distance::Output> &reference_dist,
    NNHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    const Distance &distance, std::size_t max_candidates, double epsilon,
    std::size_t n_iters, Progress &progress) {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  const std::size_t n_reference_points = distance.nx;
  NNHeap<DistOut, Idx> query_candidates(n_reference_points, max_candidates);
  build_query_candidates(reference_idx, reference_dist, n_reference_nbrs,
                         query_candidates);
  non_search_query(nn_heap, distance, query_candidates, epsilon, progress,
                   n_iters);
}

template <typename T, typename Container, typename Compare>
auto pop(std::priority_queue<T, Container, Compare> &pq) -> T {
  auto result = pq.top();
  pq.pop();
  return result;
}

template <typename T> void mark_visited(BitVec &table, T candidate) {
  auto res = std::ldiv(candidate, BITVEC_BIT_WIDTH);
  table[res.quot].set(res.rem);
}

template <typename T>
auto has_been_and_mark_visited(BitVec &table, T candidate) -> bool {
  auto res = std::ldiv(candidate, BITVEC_BIT_WIDTH);
  auto &chunk = table[res.quot];
  auto is_visited = chunk.test(res.rem);
  chunk.set(res.rem);
  return is_visited;
}

template <typename Distance, typename Progress>
void non_search_query(
    NNHeap<typename Distance::Output, typename Distance::Index> &current_graph,
    const Distance &distance,
    const NNHeap<typename Distance::Output, typename Distance::Index>
        &query_candidates,
    double epsilon, Progress &progress, std::size_t n_iters, std::size_t begin,
    std::size_t end) {

  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  using Seed = std::pair<DistOut, Idx>;

  const std::size_t n_nbrs = current_graph.n_nbrs;
  const std::size_t max_candidates = query_candidates.n_nbrs;

  // std::priority_queue is a max heap, so we need to implement the comparison
  // as "greater than" to get the smallest distance first
  auto cmp = [](Seed left, Seed right) { return left.first > right.first; };
  const double distance_scale = 1.0 + epsilon;
  const std::size_t n_bitsets = bitvec_size(query_candidates.n_points);

  for (std::size_t query_idx = begin; query_idx < end; query_idx++) {
    BitVec visited(n_bitsets);
    std::priority_queue<Seed, std::vector<Seed>, decltype(cmp)> seed_set(cmp);
    for (std::size_t j = 0; j < n_nbrs; j++) {
      Idx candidate_idx = current_graph.index(query_idx, j);
      if (candidate_idx == current_graph.npos()) {
        continue;
      }
      seed_set.emplace(current_graph.distance(query_idx, j), candidate_idx);
      mark_visited(visited, candidate_idx);
    }

    double distance_bound =
        distance_scale *
        static_cast<double>(current_graph.max_distance(query_idx));

    bool stop_early = false;
    for (std::size_t n = 0; n < n_iters; n++) {
      for (std::size_t n2 = 0; n2 < max_candidates; n2++) {
        if (seed_set.empty()) {
          stop_early = true;
          break;
        }

        Seed vertex = pop(seed_set);
        DistOut d_vertex = vertex.first;
        if (d_vertex >= distance_bound) {
          stop_early = true;
          break;
        }
        Idx vertex_idx = vertex.second;
        for (std::size_t k = 0; k < max_candidates; k++) {
          Idx candidate_idx = query_candidates.index(vertex_idx, k);
          if (candidate_idx == query_candidates.npos() ||
              has_been_and_mark_visited(visited, candidate_idx)) {
            continue;
          }
          DistOut d = distance(candidate_idx, query_idx);
          if (static_cast<double>(d) >= distance_bound) {
            continue;
          }
          current_graph.checked_push(query_idx, d, candidate_idx);
          seed_set.emplace(d, candidate_idx);
          distance_bound =
              distance_scale *
              static_cast<double>(current_graph.max_distance(query_idx));
        }
      };
      if (stop_early) {
        break;
      }
    }

    TDOANN_ITERFINISHED();
    visited.clear();
  }
}

// Use neighbor-of-neighbor search rather than local join to update the kNN.
template <typename Distance, typename Progress>
void non_search_query(
    NNHeap<typename Distance::Output, typename Distance::Index> &current_graph,
    const Distance &distance,
    const NNHeap<typename Distance::Output, typename Distance::Index>
        &query_candidates,
    double epsilon, Progress &progress, std::size_t n_iters) {

  non_search_query(current_graph, distance, query_candidates, epsilon, progress,
                   n_iters, 0, current_graph.n_points);
}
} // namespace tdoann
#endif // TDOANN_NNDESCENT_H

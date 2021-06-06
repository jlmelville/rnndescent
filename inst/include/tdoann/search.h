// BSD 2-Clause License
//
// Copyright 2021 James Melville
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

#ifndef TDOANN_SEARCH_H
#define TDOANN_SEARCH_H

#include <queue>

#include "bvset.h"
#include "nngraph.h"

namespace tdoann {

// Create the neighbor list for each reference item, i.e. the
// neighbor-of-neighbors
template <typename DistOut, typename Idx>
void build_query_candidates(const SparseNNGraph<DistOut, Idx> &reference_graph,
                            NNHeap<DistOut, Idx> &query_candidates,
                            std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end; i++) {
    for (std::size_t j = reference_graph.row_ptr[i];
         j < reference_graph.row_ptr[i + 1]; j++) {
      auto nbr = reference_graph.col_idx[j];
      // for querying, a reference that is a neighbor of itself is not
      // interesting
      if (nbr == query_candidates.npos() || i == nbr) {
        continue;
      }
      query_candidates.checked_push(i, reference_graph.dist[j], nbr);
    }
  }
}

template <typename DistOut, typename Idx>
auto build_query_candidates(const SparseNNGraph<DistOut, Idx> &reference_graph,
                            std::size_t max_candidates)
    -> NNHeap<DistOut, Idx> {
  NNHeap<DistOut, Idx> query_candidates(reference_graph.n_points,
                                        max_candidates);
  build_query_candidates(reference_graph, query_candidates, 0,
                         query_candidates.n_points);
  return query_candidates;
}

template <typename Parallel, typename DistOut, typename Idx>
auto build_query_candidates(const SparseNNGraph<DistOut, Idx> &reference_graph,
                            std::size_t max_candidates, std::size_t n_threads,
                            std::size_t grain_size) -> NNHeap<DistOut, Idx> {
  NNHeap<DistOut, Idx> query_candidates(reference_graph.n_points,
                                        max_candidates);
  auto worker = [&](std::size_t begin, std::size_t end) {
    build_query_candidates(reference_graph, query_candidates, begin, end);
  };
  Parallel::parallel_for(0, query_candidates.n_points, worker, n_threads,
                         grain_size);
  return query_candidates;
}

template <typename Distance, typename Progress>
void nn_query(
    const SparseNNGraph<typename Distance::Output, typename Distance::Index>
        &reference_graph,
    NNHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    const Distance &distance, std::size_t max_candidates, double epsilon,
    std::size_t n_iters, Progress &progress) {
  auto query_candidates =
      build_query_candidates(reference_graph, max_candidates);
  non_search_query(nn_heap, distance, query_candidates, epsilon, progress,
                   n_iters);
}

template <typename Parallel, typename Distance, typename Progress>
void nn_query(
    const SparseNNGraph<typename Distance::Output, typename Distance::Index>
        &reference_graph,
    NNHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    const Distance &distance, std::size_t max_candidates, double epsilon,
    std::size_t n_iters, Progress &progress, std::size_t n_threads = 0,
    std::size_t grain_size = 1) {
  const std::size_t n_points = nn_heap.n_points;
  auto query_candidates = build_query_candidates<Parallel>(
      reference_graph, max_candidates, n_threads, grain_size);

  NullProgress null_progress;
  auto query_non_search_worker = [&](std::size_t begin, std::size_t end) {
    non_search_query(nn_heap, distance, query_candidates, epsilon,
                     null_progress, n_iters, begin, end);
  };
  batch_parallel_for<Parallel>(query_non_search_worker, progress, n_points,
                               n_threads, grain_size);
}

template <typename T, typename Container, typename Compare>
auto pop(std::priority_queue<T, Container, Compare> &pq) -> T {
  auto result = pq.top();
  pq.pop();
  return result;
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

  const std::size_t n_nbrs = current_graph.n_nbrs;
  const std::size_t max_candidates = query_candidates.n_nbrs;
  const double distance_scale = 1.0 + epsilon;

  using Seed = std::pair<DistOut, Idx>;
  // std::priority_queue is a max heap, so we need to implement the comparison
  // as "greater than" to get the smallest distance first
  auto cmp = [](Seed left, Seed right) { return left.first > right.first; };

  for (std::size_t query_idx = begin; query_idx < end; query_idx++) {
    auto visited = create_set(query_candidates.n_points);
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
        if (static_cast<double>(d_vertex) >= distance_bound) {
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
      } // n2 next candidate
      if (stop_early) {
        break;
      }
    } // n next iteration

    TDOANN_ITERFINISHED();
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

#endif // TDOANN_SEARCH_H

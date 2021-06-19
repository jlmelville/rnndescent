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

#include "bvset.h"
#include "nbrqueue.h"
#include "nngraph.h"

namespace tdoann {

template <typename Distance, typename Progress>
void nn_query(
    const SparseNNGraph<typename Distance::Output, typename Distance::Index>
        &reference_graph,
    NNHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    const Distance &distance, double epsilon, Progress &progress) {

  non_search_query(nn_heap, distance, reference_graph, epsilon, progress);
}

template <typename Parallel, typename Distance, typename Progress>
void nn_query(
    const SparseNNGraph<typename Distance::Output, typename Distance::Index>
        &reference_graph,
    NNHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    const Distance &distance, double epsilon, Progress &progress,
    std::size_t n_threads = 0, std::size_t grain_size = 1) {
  const std::size_t n_points = nn_heap.n_points;

  NullProgress null_progress;
  auto query_non_search_worker = [&](std::size_t begin, std::size_t end) {
    non_search_query(nn_heap, distance, reference_graph, epsilon, null_progress,
                     begin, end);
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
    const SparseNNGraph<typename Distance::Output, typename Distance::Index>
        &search_graph,
    double epsilon, Progress &progress, std::size_t begin, std::size_t end) {

  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;

  const std::size_t n_nbrs = current_graph.n_nbrs;

  const double distance_scale = 1.0 + epsilon;

  for (std::size_t query_idx = begin; query_idx < end; query_idx++) {
    auto visited = create_set(search_graph.n_points);
    NbrQueue<DistOut, Idx> seed_set;
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

    while (!seed_set.empty()) {
      auto vertex = seed_set.pop();
      DistOut d_vertex = vertex.first;
      if (static_cast<double>(d_vertex) >= distance_bound) {
        break;
      }
      Idx vertex_idx = vertex.second;
      const std::size_t max_candidates = search_graph.n_nbrs(vertex_idx);
      for (std::size_t k = 0; k < max_candidates; k++) {
        Idx candidate_idx = search_graph.index(vertex_idx, k);
        if (candidate_idx == search_graph.npos() ||
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
    } // next candidate

    TDOANN_ITERFINISHED();
  }
}

template <typename Distance, typename Progress>
void non_search_query(
    NNHeap<typename Distance::Output, typename Distance::Index> &current_graph,
    const Distance &distance,
    const SparseNNGraph<typename Distance::Output, typename Distance::Index>
        &search_graph,
    double epsilon, Progress &progress) {

  non_search_query(current_graph, distance, search_graph, epsilon, progress, 0,
                   current_graph.n_points);
}

} // namespace tdoann

#endif // TDOANN_SEARCH_H

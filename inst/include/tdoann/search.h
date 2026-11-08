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
// POSSIBILITY OF SUCH DAMAGE.

#ifndef TDOANN_SEARCH_H
#define TDOANN_SEARCH_H

#include "bvset.h"
#include "distancebase.h"
#include "nbrqueue.h"
#include "nngraph.h"
#include "parallel.h"
#include "progressbase.h"
#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace tdoann {

template <typename Out, typename Idx, typename CalculateFour>
void nn_query(const SparseNNGraph<Out, Idx> &search_graph,
              NNHeap<Out, Idx> &nn_heap, const BaseDistance<Out, Idx> &distance,
              double epsilon, std::size_t max_distance_calculations,
              std::vector<std::size_t> &distance_counts, std::size_t n_threads,
              ProgressBase &progress, const Executor &executor,
              const CalculateFour &calculate_four) {
  auto worker = [&](std::size_t begin, std::size_t end) {
    non_search_query(nn_heap, distance, search_graph, epsilon,
                     max_distance_calculations, distance_counts, begin, end,
                     calculate_four);
  };
  progress.set_n_iters(1);
  ExecutionParams exec_params{100 * n_threads};
  dispatch_work(worker, nn_heap.n_points, n_threads, exec_params, progress,
                executor);
}

template <typename Out, typename Idx>
void nn_query(const SparseNNGraph<Out, Idx> &search_graph,
              NNHeap<Out, Idx> &nn_heap, const BaseDistance<Out, Idx> &distance,
              double epsilon, std::size_t max_distance_calculations,
              std::vector<std::size_t> &distance_counts, std::size_t n_threads,
              ProgressBase &progress, const Executor &executor) {
  nn_query(search_graph, nn_heap, distance, epsilon, max_distance_calculations,
           distance_counts, n_threads, progress, executor, nullptr);
}

template <typename T, typename Container, typename Compare>
auto pop(std::priority_queue<T, Container, Compare> &queue) -> T {
  auto result = queue.top();
  queue.pop();
  return result;
}

template <typename Out, typename Idx, typename CalculateFour>
void non_search_query(NNHeap<Out, Idx> &current_graph,
                      const BaseDistance<Out, Idx> &distance,
                      const SparseNNGraph<Out, Idx> &search_graph,
                      double epsilon, std::size_t max_distance_calculations,
                      std::vector<std::size_t> &distance_counts,
                      std::size_t begin, std::size_t end,
                      const CalculateFour &calculate_four) {
  constexpr auto npos = static_cast<Idx>(-1);
  constexpr bool batched = !std::is_same_v<CalculateFour, std::nullptr_t>;

  const std::size_t n_nbrs = current_graph.n_nbrs;
  const double distance_scale = 1.0 + epsilon;

  for (std::size_t query_idx = begin; query_idx < end; query_idx++) {
    auto visited = create_set(search_graph.n_points);
    NbrQueue<Out, Idx> seed_set;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      Idx candidate_idx = current_graph.index(query_idx, j);
      if (candidate_idx == npos) {
        continue;
      }
      seed_set.emplace(current_graph.distance(query_idx, j), candidate_idx);
      mark_visited(visited, candidate_idx);
    }

    double distance_bound =
        distance_scale *
        static_cast<double>(current_graph.max_distance(query_idx));

    std::size_t n_searches_for_query = 0;
    while (!seed_set.empty() &&
           n_searches_for_query < max_distance_calculations) {
      auto vertex = seed_set.pop();
      auto d_vertex = vertex.first;
      if (static_cast<double>(d_vertex) >= distance_bound) {
        break;
      }
      auto vertex_idx = vertex.second;
      const std::size_t max_candidates = search_graph.n_nbrs(vertex_idx);
      auto accept = [&](Idx candidate_idx, Out dist) {
        if (static_cast<double>(dist) >= distance_bound) {
          return;
        }
        current_graph.checked_push(query_idx, dist, candidate_idx);
        seed_set.emplace(dist, candidate_idx);
        distance_bound =
            distance_scale *
            static_cast<double>(current_graph.max_distance(query_idx));
      };

      std::array<Idx, 4> candidates;
      std::size_t size = 0;
      // Every eligible candidate uses one slot in the budget, even if its
      // distance is still buffered. Apply distances in adjacency order.
      for (std::size_t k = 0; k < max_candidates &&
                              n_searches_for_query < max_distance_calculations;
           ++k) {
        auto candidate_idx = search_graph.index(vertex_idx, k);
        if (candidate_idx == npos ||
            has_been_and_mark_visited(visited, candidate_idx)) {
          continue;
        }
        ++n_searches_for_query;
        if constexpr (batched) {
          candidates[size++] = candidate_idx;
          if (size == candidates.size()) {
            const auto distances = calculate_four(query_idx, candidates);
            for (std::size_t j = 0; j < candidates.size(); ++j) {
              accept(candidates[j], distances[j]);
            }
            size = 0;
          }
        } else {
          accept(candidate_idx, distance.calculate(candidate_idx, query_idx));
        }
      }
      if constexpr (batched) {
        for (std::size_t j = 0; j < size; ++j) {
          accept(candidates[j], distance.calculate(candidates[j], query_idx));
        }
      }
    } // next candidate
    distance_counts[query_idx] = n_searches_for_query;
  }
}

template <typename Out, typename Idx>
void non_search_query(NNHeap<Out, Idx> &current_graph,
                      const BaseDistance<Out, Idx> &distance,
                      const SparseNNGraph<Out, Idx> &search_graph,
                      double epsilon, std::size_t max_distance_calculations,
                      std::vector<std::size_t> &distance_counts,
                      std::size_t begin, std::size_t end) {
  non_search_query(current_graph, distance, search_graph, epsilon,
                   max_distance_calculations, distance_counts, begin, end,
                   nullptr);
}

} // namespace tdoann

#endif // TDOANN_SEARCH_H

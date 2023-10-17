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

#ifndef TDOANN_PREPARE_H
#define TDOANN_PREPARE_H

#include <algorithm>
#include <numeric>
#include <vector>

#include "distancebase.h"
#include "nngraph.h"
#include "parallel.h"
#include "random.h"

namespace tdoann {

// Returns a vector of indices that represents the sorted order of elements in
// the range [first, last).
template <typename It>
auto order(It first, It last) -> std::vector<std::size_t> {
  std::vector<std::size_t> idx(last - first);
  std::iota(idx.begin(), idx.end(), static_cast<std::size_t>(0));

  auto cmp = [&first](std::size_t diff_a, std::size_t diff_b) -> bool {
    return *(first + diff_a) < *(first + diff_b);
  };
  std::stable_sort(idx.begin(), idx.end(), cmp);

  return idx;
}

template <typename Out, typename Idx>
auto kth_smallest_distance(const SparseNNGraph<Out, Idx> &graph,
                           std::size_t item_i, std::size_t k_small) -> Out {
  // This is coupled to the internals of SparseGraph
  auto start_itr = graph.dist.begin() + graph.row_ptr[item_i];
  auto end_itr = graph.dist.begin() + graph.row_ptr[item_i + 1];
  std::vector<Out> distances(start_itr, end_itr);

  // Find the k-th smallest distance
  std::nth_element(distances.begin(), distances.begin() + k_small,
                   distances.end());
  return distances[k_small - 1];
}

template <typename Out, typename Idx>
void degree_prune_impl(const SparseNNGraph<Out, Idx> &graph,
                       SparseNNGraph<Out, Idx> &result, std::size_t max_degree,
                       std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end; i++) {
    const auto unpruned_n_nbrs = graph.n_nbrs(i);
    if (unpruned_n_nbrs <= max_degree) {
      continue;
    }

    auto max_degree_dist = kth_smallest_distance(graph, i, max_degree);
    for (std::size_t j = 0; j < unpruned_n_nbrs; j++) {
      if (graph.distance(i, j) > max_degree_dist) {
        result.mark_for_deletion(i, j);
      }
    }
  }
}

template <typename Out, typename Idx>
auto degree_prune(const SparseNNGraph<Out, Idx> &graph, std::size_t max_degree,
                  std::size_t n_threads, ProgressBase &progress,
                  Executor &executor) -> SparseNNGraph<Out, Idx> {
  SparseNNGraph<Out, Idx> result(graph.row_ptr, graph.col_idx, graph.dist);
  auto worker = [&](std::size_t begin, std::size_t end) {
    degree_prune_impl(graph, result, max_degree, begin, end);
  };
  dispatch_work(worker, graph.n_points, n_threads, progress, executor);
  return result;
}

// remove neighbors which are "occlusions"
// for point i with neighbors p and q, if d(p, q) < d(i, p), then p occludes q
template <typename Out, typename Idx>
void remove_long_edges_impl(const SparseNNGraph<Out, Idx> &graph,
                            const BaseDistance<Out, Idx> &distance,
                            RandomGenerator &rand, double prune_probability,
                            SparseNNGraph<Out, Idx> &result, std::size_t begin,
                            std::size_t end) {
  for (std::size_t i = begin; i < end; i++) {
    const std::size_t n_nbrs = graph.n_nbrs(i);
    if (n_nbrs == 0) {
      continue;
    }
    // order neighbors by increasing distance
    auto ordered = order(graph.dist.begin() + graph.row_ptr[i],
                         graph.dist.begin() + graph.row_ptr[i + 1]);
    // loop starts at 1: we always keep the nearest neighbor so we start with
    // the next nearest neighbor
    for (std::size_t j = 1; j < n_nbrs; j++) {
      const auto jth_nearest = ordered[j];
      Idx nbr_j = graph.index(i, jth_nearest);
      Out dist_ij = graph.distance(i, jth_nearest);
      // check the distance between j and all retained neighbors (k) so far
      for (std::size_t k = 0; k < j; k++) {
        const auto kth_nearest = ordered[k];
        if (result.is_marked_for_deletion(i, kth_nearest)) {
          // k was already considered an occlusion, no need to test
          continue;
        }
        Idx nbr_k = graph.index(i, kth_nearest);
        Out dist_jk = distance.calculate(nbr_j, nbr_k);
        auto rand_val = rand.unif();
        if (dist_jk < dist_ij && rand_val < prune_probability) {
          // j occludes k, mark j for deletion
          result.mark_for_deletion(i, jth_nearest);
          break;
        }
      }
    }
  }
}

template <typename Out, typename Idx>
auto remove_long_edges(const SparseNNGraph<Out, Idx> &graph,
                       const BaseDistance<Out, Idx> &distance,
                       RandomGenerator &rand, double prune_probability)
    -> SparseNNGraph<Out, Idx> {
  SparseNNGraph<Out, Idx> result(graph.row_ptr, graph.col_idx, graph.dist);
  remove_long_edges_impl(graph, distance, rand, prune_probability, result, 0,
                         graph.n_points);
  return result;
}

template <typename Out, typename Idx>
auto remove_long_edges(const SparseNNGraph<Out, Idx> &graph,
                       const BaseDistance<Out, Idx> &distance,
                       ParallelRandomProvider &parallel_rand,
                       double prune_probability, std::size_t n_threads,
                       ProgressBase &progress, Executor &executor)
    -> SparseNNGraph<Out, Idx> {
  SparseNNGraph<Out, Idx> result(graph.row_ptr, graph.col_idx, graph.dist);
  parallel_rand.initialize();
  auto worker = [&](std::size_t begin, std::size_t end) {
    auto rand = parallel_rand.get_parallel_instance(end);
    remove_long_edges_impl(graph, distance, *rand, prune_probability, result,
                           begin, end);
  };
  dispatch_work(worker, graph.n_points, n_threads, progress, executor);
  return result;
}

template <typename Out, typename Idx>
auto merge_graphs(const SparseNNGraph<Out, Idx> &graph1,
                  const SparseNNGraph<Out, Idx> &graph2)
    -> SparseNNGraph<Out, Idx> {
  const std::size_t n_points = graph1.n_points;

  std::vector<std::size_t> merged_row_ptr(n_points + 1, 0);
  std::vector<Idx> merged_col_idx;
  std::vector<Out> merged_dist;

  for (std::size_t i = 0; i < n_points; i++) {
    const auto begin1 = graph1.row_ptr[i];
    const auto end1 = graph1.row_ptr[i + 1];
    const auto begin2 = graph2.row_ptr[i];
    const auto end2 = graph2.row_ptr[i + 1];

    // Insert neighbors from graph1
    merged_col_idx.insert(merged_col_idx.end(), graph1.col_idx.begin() + begin1,
                          graph1.col_idx.begin() + end1);
    merged_dist.insert(merged_dist.end(), graph1.dist.begin() + begin1,
                       graph1.dist.begin() + end1);

    // Insert neighbors from graph2 only if they're not already present in
    // graph1
    for (auto j = begin2; j < end2; j++) {
      if (!std::binary_search(graph1.col_idx.begin() + begin1,
                              graph1.col_idx.begin() + end1,
                              graph2.col_idx[j])) {
        merged_col_idx.push_back(graph2.col_idx[j]);
        merged_dist.push_back(graph2.dist[j]);
      }
    }

    // Update the merged_row_ptr for the next iteration
    merged_row_ptr[i + 1] = merged_col_idx.size();
  }
  return SparseNNGraph<Out, Idx>(merged_row_ptr, merged_col_idx, merged_dist);
}

} // namespace tdoann
#endif // TDOANN_PREPARE_H

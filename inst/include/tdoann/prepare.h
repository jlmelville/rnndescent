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

namespace tdoann {

template <typename It>
auto order(It first, It last) -> std::vector<std::size_t> {
  std::vector<std::size_t> idx(last - first);
  std::iota(idx.begin(), idx.end(), static_cast<std::size_t>(0));

  auto cmp = [&first](std::size_t a, std::size_t b) {
    return *(first + a) < *(first + b);
  };
  std::stable_sort(idx.begin(), idx.end(), cmp);

  return idx;
}

template <typename SparseNNGraph>
auto degree_prune(const SparseNNGraph &graph, std::size_t max_degree)
    -> SparseNNGraph {
  using DistOut = typename SparseNNGraph::DistanceOut;
  using Idx = typename SparseNNGraph::Index;

  const std::size_t n_points = graph.n_points;

  std::vector<std::size_t> new_row_ptr(n_points + 1);
  std::vector<Idx> new_col_idx;
  std::vector<DistOut> new_dist;

  for (std::size_t i = 0; i < n_points; i++) {
    const std::size_t i1 = i + 1;
    new_row_ptr[i1] = new_row_ptr[i];

    const auto begin = graph.row_ptr[i];
    const auto end = graph.row_ptr[i1];

    auto ordered = order(graph.dist.begin() + begin, graph.dist.begin() + end);

    const auto unpruned_n_nbrs = end - begin;
    const auto n_nbrs = std::min(unpruned_n_nbrs, max_degree);

    for (std::size_t j = 0; j < n_nbrs; j++) {
      new_col_idx.push_back(graph.index(i, ordered[j]));
      new_dist.push_back(graph.distance(i, ordered[j]));
    }
    new_row_ptr[i1] += n_nbrs;
  }
  return SparseNNGraph(new_row_ptr, new_col_idx, new_dist);
}

template <typename SparseNNGraph, typename Distance, typename Rand>
auto remove_long_edges_sp(const SparseNNGraph &graph, const Distance &distance,
                          Rand &rand, double prune_probability)
    -> SparseNNGraph {
  using DistOut = typename SparseNNGraph::DistanceOut;
  using Idx = typename SparseNNGraph::Index;

  const std::size_t n_points = graph.n_points;

  std::vector<std::size_t> new_row_ptr(n_points + 1);
  std::vector<Idx> new_col_idx;
  std::vector<DistOut> new_dist;

  for (std::size_t i = 0; i < n_points; i++) {
    const std::size_t i1 = i + 1;
    new_row_ptr[i1] = new_row_ptr[i];

    const std::size_t n_nbrs = graph.n_nbrs(i);
    if (n_nbrs == 0) {
      continue;
    }

    auto ordered = order(graph.dist.begin() + graph.row_ptr[i],
                         graph.dist.begin() + graph.row_ptr[i1]);

    // initialize new graph with closest neighbor
    new_col_idx.push_back(graph.index(i, ordered[0]));
    new_dist.push_back(graph.distance(i, ordered[0]));
    ++new_row_ptr[i1];

    // search all other neighbors (NB: start at 1)
    for (std::size_t j = 1; j < n_nbrs; j++) {
      Idx nbr = graph.index(i, ordered[j]);
      DistOut nbr_dist = graph.distance(i, ordered[j]);

      // Compare this neighbor with those that previously passed the filter
      bool add_nbr = true;
      for (std::size_t k = new_row_ptr[i]; k < new_row_ptr[i1]; k++) {
        Idx ng_nbr = new_col_idx[k];
        DistOut d = distance(nbr, ng_nbr);
        if (d < nbr_dist && rand.unif() < prune_probability) {
          add_nbr = false;
          break;
        }
      }

      if (add_nbr) {
        new_col_idx.push_back(nbr);
        new_dist.push_back(nbr_dist);
        ++new_row_ptr[i1];
      }
    }
  }
  return SparseNNGraph(new_row_ptr, new_col_idx, new_dist);
}

// remove neighbors which are "occlusions"
// for point i with neighbors p and q, if d(p, q) < d(i, p), then p occludes q
template <typename SparseNNGraph, typename Distance>
auto remove_long_edges_sp(const SparseNNGraph &graph, const Distance &distance)
    -> SparseNNGraph {
  using DistOut = typename SparseNNGraph::DistanceOut;
  using Idx = typename SparseNNGraph::Index;

  const std::size_t n_points = graph.n_points;

  std::vector<std::size_t> new_row_ptr(n_points + 1);
  std::vector<Idx> new_col_idx;
  std::vector<DistOut> new_dist;

  for (std::size_t i = 0; i < n_points; i++) {
    const std::size_t i1 = i + 1;
    new_row_ptr[i1] = new_row_ptr[i];

    const std::size_t n_nbrs = graph.n_nbrs(i);
    if (n_nbrs == 0) {
      continue;
    }

    auto ordered = order(graph.dist.begin() + graph.row_ptr[i],
                         graph.dist.begin() + graph.row_ptr[i1]);

    // initialize new graph with closest neighbor
    new_col_idx.push_back(graph.index(i, ordered[0]));
    new_dist.push_back(graph.distance(i, ordered[0]));
    ++new_row_ptr[i1];

    // search all other neighbors (NB: start at 1)
    for (std::size_t j = 1; j < n_nbrs; j++) {
      Idx nbr = graph.index(i, ordered[j]);
      DistOut nbr_dist = graph.distance(i, ordered[j]);

      // Compare this neighbor with those that previously passed the filter
      bool add_nbr = true;
      for (std::size_t k = new_row_ptr[i]; k < new_row_ptr[i1]; k++) {
        Idx ng_nbr = new_col_idx[k];
        DistOut d = distance(nbr, ng_nbr);
        if (d < nbr_dist) {
          add_nbr = false;
          break;
        }
      }

      if (add_nbr) {
        new_col_idx.push_back(nbr);
        new_dist.push_back(nbr_dist);
        ++new_row_ptr[i1];
      }
    }
  }
  return SparseNNGraph(new_row_ptr, new_col_idx, new_dist);
}

template <typename SparseNNGraph>
auto merge_graphs(const SparseNNGraph &g1, const SparseNNGraph &g2)
    -> SparseNNGraph {
  using DistOut = typename SparseNNGraph::DistanceOut;
  using Idx = typename SparseNNGraph::Index;

  const std::size_t n_points = g1.n_points;

  std::vector<std::size_t> merged_row_ptr(n_points + 1);
  std::vector<Idx> merged_col_idx;
  std::vector<DistOut> merged_dist;

  std::vector<Idx> search_idx = g1.col_idx;
  for (std::size_t i = 0; i < n_points; i++) {
    const auto begin = g1.row_ptr[i];
    const auto end = g1.row_ptr[i + 1];

    std::sort(search_idx.begin() + begin, search_idx.begin() + end);

    std::vector<Idx> col_idx_i(g1.col_idx.begin() + begin,
                               g1.col_idx.begin() + end);
    std::vector<DistOut> dist_i(g1.dist.begin() + begin, g1.dist.begin() + end);

    merged_row_ptr[i + 1] = merged_row_ptr[i] + col_idx_i.size();

    for (std::size_t j = g2.row_ptr[i]; j < g2.row_ptr[i + 1]; j++) {
      if (!std::binary_search(search_idx.begin() + begin,
                              search_idx.begin() + end, g2.col_idx[j])) {
        col_idx_i.push_back(g2.col_idx[j]);
        dist_i.push_back(g2.dist[j]);
        ++merged_row_ptr[i + 1];
      }
    }
    merged_col_idx.insert(merged_col_idx.end(), col_idx_i.begin(),
                          col_idx_i.end());
    merged_dist.insert(merged_dist.end(), dist_i.begin(), dist_i.end());
  }
  return SparseNNGraph(merged_row_ptr, merged_col_idx, merged_dist);
}

} // namespace tdoann
#endif // TDOANN_PREPARE_H

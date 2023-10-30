// BSD 2-Clause License
//
// Copyright 2023 James Melville
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

#ifndef TDOANN_RPTREEIMPLICIT_H
#define TDOANN_RPTREEIMPLICIT_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#include "distancebase.h"
#include "heap.h"
#include "parallel.h"
#include "random.h"

namespace tdoann {

// Tree Building

template <typename Idx> struct RPTreeImplicit {
  using Index = Idx;

  std::vector<std::pair<Idx, Idx>> normal_indices = {};
  std::vector<std::pair<std::size_t, std::size_t>> children = {};
  std::vector<std::vector<Idx>> indices = {};
  std::size_t leaf_size = 0;
  std::size_t ndim;

  RPTreeImplicit() = default;

  // Pre-allocate memory based on a rough best-case lower bound of nodes
  // (balanced tree)
  RPTreeImplicit(std::size_t num_indices, std::size_t leaf_size,
                 std::size_t ndim)
      : ndim(ndim) {
    // expression takes advantage of shifting the dividend by one so the
    // division truncation towards zero effectively rounds up to the nearest
    // integer. Avoids the cast and ceil in the equivalent (and clearer):
    // std::ceil(static_cast<double>(N) / L);
    // also need to safe guard if all indices can fit in one node
    std::size_t min_nodes =
        num_indices <= leaf_size ? 1 : (num_indices / (2 * leaf_size) + 1) - 1;

    normal_indices.reserve(min_nodes);
    children.reserve(min_nodes);
    indices.reserve(min_nodes);
  }

  void add_node(Idx left, Idx right, std::size_t left_node_num,
                std::size_t right_node_num) {
    // indices is never read from
    static const std::vector<Idx> dummy_indices =
        std::vector<Idx>(0, static_cast<Idx>(-1));
    indices.push_back(dummy_indices);

    // children is checked during leaf array construction
    normal_indices.emplace_back(left, right);
    children.emplace_back(left_node_num, right_node_num);
  }

  void add_leaf(const std::vector<Idx> &indices_) {
    // get_leaves_from_tree and recursive_convert looks for .first == sentinel
    // .second is ignored
    static constexpr auto child_sentinel = static_cast<std::size_t>(-1);
    static const std::pair<std::size_t, std::size_t> dummy_child = {
        child_sentinel, child_sentinel};
    children.push_back(dummy_child);

    static constexpr auto idx_sentinel = static_cast<Idx>(-1);
    static const std::pair<Idx, Idx> dummy_normal = {idx_sentinel,
                                                     idx_sentinel};
    normal_indices.push_back(dummy_normal);

    // used in leaf array, conversion, search tree
    indices.push_back(indices_);
    leaf_size = std::max(leaf_size, indices_.size());
  }

  bool is_leaf(std::size_t i) const {
    return children[i].first == static_cast<std::size_t>(-1);
  }
};

template <typename Out, typename Idx>
uint8_t select_side(Idx i, const BaseDistance<Out, Idx> &distance, Idx left,
                    Idx right, RandomIntGenerator<Idx> &rng) {
  constexpr Out EPS = 1e-8;

  Out margin = distance.calculate(right, i) - distance.calculate(left, i);

  if (std::abs(margin) < EPS) {
    return rng.rand_int(2);
  }
  return margin > 0 ? 0 : 1;
}

template <typename Out, typename Idx>
void split_indices(const BaseDistance<Out, Idx> &distance, Idx left_index,
                   Idx right_index, const std::vector<Idx> &indices,
                   std::vector<Idx> &indices_left,
                   std::vector<Idx> &indices_right,
                   RandomIntGenerator<Idx> &rng) {
  std::vector<uint8_t> side(indices.size(), 0);
  std::size_t n_left = 0;
  std::size_t n_right = 0;

  Idx left = indices[left_index];
  Idx right = indices[right_index];

  for (std::size_t i = 0; i < indices.size(); ++i) {
    side[i] =
        select_side(static_cast<Idx>(indices[i]), distance, left, right, rng);
    if (side[i] == 0) {
      ++n_left;
    } else {
      ++n_right;
    }
  }

  // If either side is empty, reset counts and assign sides randomly.

  if (n_left == 0 || n_right == 0) {
    n_left = 0;
    n_right = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      side[i] = rng.rand_int(2);
      if (side[i] == 0) {
        ++n_left;
      } else {
        ++n_right;
      }
    }
  }

  indices_left.resize(n_left);
  indices_right.resize(n_right);
  n_left = 0;
  n_right = 0;
  for (std::size_t i = 0; i < side.size(); ++i) {
    if (side[i] == 0) {
      indices_left[n_left++] = indices[i];
    } else {
      indices_right[n_right++] = indices[i];
    }
  }
}

template <typename Out, typename Idx>
std::tuple<std::vector<Idx>, std::vector<Idx>, Idx, Idx>
distance_random_projection_split(const BaseDistance<Out, Idx> &distance,
                                 const std::vector<Idx> &indices,
                                 RandomIntGenerator<Idx> &rng) {
  auto [left_index, right_index] = select_random_points(indices, rng);

  std::vector<Idx> indices_left;
  std::vector<Idx> indices_right;

  split_indices(distance, left_index, right_index, indices, indices_left,
                indices_right, rng);

  return std::make_tuple(std::move(indices_left), std::move(indices_right),
                         indices[left_index], indices[right_index]);
}

template <typename Out, typename Idx>
void make_tree_recursive(const BaseDistance<Out, Idx> &distance,
                         const std::vector<Idx> &indices,
                         RPTreeImplicit<Idx> &tree,
                         RandomIntGenerator<Idx> &rng, uint32_t leaf_size,
                         uint32_t max_depth) {
  if (indices.size() > leaf_size && max_depth > 0) {

    auto [left_indices, right_indices, left, right] =
        distance_random_projection_split(distance, indices, rng);

    make_tree_recursive(distance, left_indices, tree, rng, leaf_size,
                        max_depth - 1);

    std::size_t left_node_num = tree.indices.size() - 1;

    make_tree_recursive(distance, right_indices, tree, rng, leaf_size,
                        max_depth - 1);

    std::size_t right_node_num = tree.indices.size() - 1;

    tree.add_node(left, right, left_node_num, right_node_num);
  } else {
    // leaf node
    tree.add_leaf(indices);
  }
}

template <typename Out, typename Idx>
RPTreeImplicit<Idx>
make_dense_tree(const BaseDistance<Out, Idx> &distance, std::size_t ndim,
                RandomIntGenerator<Idx> &rng, uint32_t leaf_size) {
  std::vector<Idx> indices(distance.get_ny());
  std::iota(indices.begin(), indices.end(), 0);

  RPTreeImplicit<Idx> tree(indices.size(), leaf_size, ndim);
  constexpr uint32_t max_depth = 100;

  make_tree_recursive(distance, indices, tree, rng, leaf_size, max_depth);

  return tree;
}

template <typename Out, typename Idx>
std::vector<RPTreeImplicit<Idx>> make_forest(
    const BaseDistance<Out, Idx> &distance, std::size_t ndim, uint32_t n_trees,
    uint32_t leaf_size, ParallelRandomIntProvider<Idx> &parallel_rand,
    std::size_t n_threads, ProgressBase &progress, const Executor &executor) {
  std::vector<RPTreeImplicit<Idx>> rp_forest(n_trees);

  parallel_rand.initialize();

  auto worker = [&](std::size_t begin, std::size_t end) {
    auto rng_ptr = parallel_rand.get_parallel_instance(end);
    for (auto i = begin; i < end; ++i) {
      rp_forest[i] = make_dense_tree(distance, ndim, *rng_ptr, leaf_size);
    }
  };

  progress.set_n_iters(1);
  ExecutionParams exec_params{};
  dispatch_work(worker, n_trees, n_threads, exec_params, progress, executor);

  return rp_forest;
}

template <typename Idx> struct SearchTreeImplicit {
  using Index = Idx;

  std::vector<std::pair<Idx, Idx>> normal_indices;
  std::vector<std::pair<std::size_t, std::size_t>> children;
  std::vector<Idx> indices;
  Idx leaf_size;

  SearchTreeImplicit() = default;

  SearchTreeImplicit(std::size_t n_nodes, std::size_t n_points,
                     std::size_t ndim, Idx lsize)
      : normal_indices(n_nodes, std::make_pair(static_cast<Idx>(-1),
                                               static_cast<Idx>(-1))),
        children(n_nodes, std::make_pair(static_cast<std::size_t>(-1),
                                         static_cast<std::size_t>(-1))),
        indices(n_points, static_cast<Idx>(-1)), leaf_size(lsize) {}

  SearchTreeImplicit(std::vector<std::pair<Idx, Idx>> norm_idxs,
                     std::vector<std::pair<std::size_t, std::size_t>> chldrn,
                     std::vector<Idx> inds, Idx lsize)
      : normal_indices(std::move(norm_idxs)), children(std::move(chldrn)),
        indices(std::move(inds)), leaf_size(lsize) {}

  bool is_leaf(std::size_t i) const {
    return normal_indices[i].first == static_cast<Idx>(-1);
  }
};

template <typename Idx>
std::pair<std::size_t, std::size_t>
recursive_convert(const RPTreeImplicit<Idx> &tree,
                  SearchTreeImplicit<Idx> &search_tree, std::size_t node_num,
                  std::size_t leaf_start, std::size_t tree_node) {

  if (tree.is_leaf(tree_node)) {
    // leaf: read from tree.indices, tree.children (in if statement above)
    auto leaf_end = leaf_start + tree.indices[tree_node].size();
    search_tree.children[node_num] = std::make_pair(leaf_start, leaf_end);
    std::copy(tree.indices[tree_node].begin(), tree.indices[tree_node].end(),
              search_tree.indices.begin() + leaf_start);
    return {node_num + 1, leaf_end};
  } else {
    // node: read from tree.hyperplanes, tree.offsets, tree.children
    search_tree.normal_indices[node_num] = tree.normal_indices[tree_node];
    search_tree.children[node_num].first = node_num + 1;
    auto old_node_num = node_num;

    auto [new_node_num, new_leaf_start] =
        recursive_convert(tree, search_tree, node_num + 1, leaf_start,
                          tree.children[tree_node].first);

    search_tree.children[old_node_num].second = new_node_num;

    return recursive_convert(tree, search_tree, new_node_num, new_leaf_start,
                             tree.children[tree_node].second);
  }
}

template <typename Idx>
SearchTreeImplicit<Idx> convert_tree_format(const RPTreeImplicit<Idx> &tree,
                                            std::size_t n_points,
                                            std::size_t ndim) {
  const auto n_nodes = tree.children.size();
  SearchTreeImplicit<Idx> search_tree(n_nodes, n_points, ndim, tree.leaf_size);

  std::size_t node_num = 0;
  std::size_t leaf_start = 0;
  recursive_convert(tree, search_tree, node_num, leaf_start, n_nodes - 1);

  return search_tree;
}

template <typename Idx>
std::vector<SearchTreeImplicit<Idx>>
convert_rp_forest(const std::vector<RPTreeImplicit<Idx>> &rp_forest,
                  std::size_t n_points, std::size_t ndim) {
  std::vector<SearchTreeImplicit<Idx>> search_forest;
  search_forest.reserve(rp_forest.size());
  for (const auto &rp_tree : rp_forest) {
    search_forest.push_back(convert_tree_format(rp_tree, n_points, ndim));
  }
  return search_forest;
}

// Searching

template <typename Out, typename Idx>
std::pair<std::size_t, std::size_t>
search_leaf_range(const SearchTreeImplicit<Idx> &tree, Idx i,
                  const BaseDistance<Out, Idx> &distance,
                  RandomIntGenerator<Idx> &rng) {
  Idx current_node = 0;
  while (true) {
    auto child_pair = tree.children[current_node];

    // it's a leaf: child_pair contains pointers into the indices
    if (tree.is_leaf(current_node)) {
      return child_pair;
    }

    // it's a node, work out which way to go
    auto side =
        select_side(i, distance, tree.normal_indices[current_node].first,
                    tree.normal_indices[current_node].second, rng);
    if (side == 0) {
      current_node = child_pair.first; // go left
    } else {
      current_node = child_pair.second; // go right
    }
  }
}

template <typename Out, typename Idx>
std::vector<Idx> search_indices(const SearchTreeImplicit<Idx> &tree, Idx i,
                                const BaseDistance<Out, Idx> &distance,
                                RandomIntGenerator<Idx> &rng) {
  std::pair<std::size_t, std::size_t> range =
      search_leaf_range(tree, i, distance, rng);
  std::vector<Idx> leaf_indices(tree.indices.begin() + range.first,
                                tree.indices.begin() + range.second);

  return leaf_indices;
}

template <typename Out, typename Idx>
void search_tree_heap_cache(const SearchTreeImplicit<Idx> &tree,
                            const BaseDistance<Out, Idx> &distance, Idx i,
                            RandomIntGenerator<Idx> &rng,
                            NNHeap<Out, Idx> &current_graph,
                            std::unordered_set<Idx> &seen) {
  std::vector<Idx> leaf_indices = search_indices(tree, i, distance, rng);

  for (auto &idx : leaf_indices) {
    if (seen.find(idx) == seen.end()) { // not-contains
      const auto d = distance.calculate(idx, i);
      current_graph.checked_push(i, d, idx);
      seen.insert(idx);
    }
  }
}

template <typename Out, typename Idx>
void search_tree_heap(const SearchTreeImplicit<Idx> &tree,
                      const BaseDistance<Out, Idx> &distance, Idx i,
                      RandomIntGenerator<Idx> &rng,
                      NNHeap<Out, Idx> &current_graph) {
  std::vector<Idx> leaf_indices = search_indices(tree, i, distance, rng);

  for (auto &idx : leaf_indices) {
    const auto d = distance.calculate(idx, i);
    current_graph.checked_push(i, d, idx);
  }
}

template <typename Out, typename Idx>
void search_forest_cache(const std::vector<SearchTreeImplicit<Idx>> &forest,
                         const BaseDistance<Out, Idx> &distance, Idx i,
                         RandomIntGenerator<Idx> &rng,
                         NNHeap<Out, Idx> &current_graph) {
  std::unordered_set<Idx> seen;

  for (const auto &tree : forest) {
    search_tree_heap_cache(tree, distance, i, rng, current_graph, seen);
  }
}

template <typename Out, typename Idx>
void search_forest(const std::vector<SearchTreeImplicit<Idx>> &forest,
                   const BaseDistance<Out, Idx> &distance, Idx i,
                   RandomIntGenerator<Idx> &rng,
                   NNHeap<Out, Idx> &current_graph) {
  for (const auto &tree : forest) {
    search_tree_heap(tree, distance, i, rng, current_graph);
  }
}

template <typename Out, typename Idx>
NNHeap<Out, Idx>
search_forest(const std::vector<SearchTreeImplicit<Idx>> &forest,
              const BaseDistance<Out, Idx> &distance, uint32_t n_nbrs,
              ParallelRandomIntProvider<Idx> &rng_provider, bool cache,
              std::size_t n_threads, ProgressBase &progress,
              const Executor &executor) {
  const auto n_queries = distance.get_ny();
  NNHeap<Out, Idx> current_graph(n_queries, n_nbrs);

  rng_provider.initialize();
  auto worker = [&](std::size_t begin, std::size_t end) {
    auto rng_ptr = rng_provider.get_parallel_instance(end);
    for (auto i = begin; i < end; ++i) {
      if (cache) {
        search_forest_cache(forest, distance, static_cast<Idx>(i), *rng_ptr,
                            current_graph);
      } else {
        search_forest(forest, distance, static_cast<Idx>(i), *rng_ptr,
                      current_graph);
      }
    }
  };

  progress.set_n_iters(n_queries);
  ExecutionParams exec_params{};
  dispatch_work(worker, n_queries, n_threads, exec_params, progress, executor);

  return current_graph;
}

} // namespace tdoann

#endif // TDOANN_RPTREEIMPLICIT_H

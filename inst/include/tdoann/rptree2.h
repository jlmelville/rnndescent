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

#ifndef TDOANN_RPTREE2_H
#define TDOANN_RPTREE2_H

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

template <typename Idx> struct RPTree2 {
  using Index = Idx;

  std::vector<std::pair<Idx, Idx>> normal_indices = {};
  std::vector<std::pair<std::size_t, std::size_t>> children = {};
  std::vector<std::vector<Idx>> indices = {};
  std::size_t leaf_size = 0;
  std::size_t ndim;

  RPTree2() = default;

  // Pre-allocate memory based on a rough best-case lower bound of nodes
  // (balanced tree)
  RPTree2(std::size_t num_indices, std::size_t leaf_size, std::size_t ndim)
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

  void add_node(Idx rpi, Idx rpj, std::size_t left_node_num,
                std::size_t right_node_num) {
    // indices is never read from
    static const std::vector<Idx> dummy_indices =
        std::vector<Idx>(0, static_cast<Idx>(-1));
    indices.push_back(dummy_indices);

    // children is checked during leaf array construction
    normal_indices.emplace_back(rpi, rpj);
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
                         left_index, right_index);
}

template <typename Out, typename Idx>
void make_tree_recursive(const BaseDistance<Out, Idx> &distance,
                         const std::vector<Idx> &indices, RPTree2<Idx> &tree,
                         RandomIntGenerator<Idx> &rng, uint32_t leaf_size,
                         uint32_t max_depth) {
  if (indices.size() > leaf_size && max_depth > 0) {

    auto [left_indices, right_indices, rpi, rpj] =
        distance_random_projection_split(distance, indices, rng);

    make_tree_recursive(distance, left_indices, tree, rng, leaf_size,
                        max_depth - 1);

    std::size_t left_node_num = tree.indices.size() - 1;

    make_tree_recursive(distance, right_indices, tree, rng, leaf_size,
                        max_depth - 1);

    std::size_t right_node_num = tree.indices.size() - 1;

    tree.add_node(rpi, rpj, left_node_num, right_node_num);
  } else {
    // leaf node
    tree.add_leaf(indices);
  }
}

template <typename Out, typename Idx>
RPTree2<Idx> make_dense_tree(const BaseDistance<Out, Idx> &distance,
                             std::size_t ndim, RandomIntGenerator<Idx> &rng,
                             uint32_t leaf_size) {
  std::vector<Idx> indices(distance.get_ny());
  std::iota(indices.begin(), indices.end(), 0);

  RPTree2<Idx> tree(indices.size(), leaf_size, ndim);
  constexpr uint32_t max_depth = 100;

  make_tree_recursive(distance, indices, tree, rng, leaf_size, max_depth);

  return tree;
}

template <typename Out, typename Idx>
std::vector<RPTree2<Idx>> make_forest(
    const BaseDistance<Out, Idx> &distance, std::size_t ndim, uint32_t n_trees,
    uint32_t leaf_size, ParallelRandomIntProvider<Idx> &parallel_rand,
    std::size_t n_threads, ProgressBase &progress, const Executor &executor) {
  std::vector<RPTree2<Idx>> rp_forest(n_trees);

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

template <typename Idx> struct SearchTree2 {
  using Index = Idx;

  std::vector<std::pair<Idx, Idx>> normal_indices;
  std::vector<std::pair<std::size_t, std::size_t>> children;
  std::vector<Idx> indices;
  Idx leaf_size;

  SearchTree2() = default;

  SearchTree2(std::size_t n_nodes, std::size_t n_points, std::size_t ndim,
              Idx lsize)
      : normal_indices(n_nodes, std::make_pair(static_cast<Idx>(-1),
                                               static_cast<Idx>(-1))),
        children(n_nodes, std::make_pair(static_cast<std::size_t>(-1),
                                         static_cast<std::size_t>(-1))),
        indices(n_points, static_cast<Idx>(-1)), leaf_size(lsize) {}

  SearchTree2(std::vector<std::pair<Idx, Idx>> norm_idxs,
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
recursive_convert(const RPTree2<Idx> &tree, SearchTree2<Idx> &search_tree,
                  std::size_t node_num, std::size_t leaf_start,
                  std::size_t tree_node) {

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
SearchTree2<Idx> convert_tree_format(const RPTree2<Idx> &tree,
                                     std::size_t n_points, std::size_t ndim) {
  const auto n_nodes = tree.children.size();
  SearchTree2<Idx> search_tree(n_nodes, n_points, ndim, tree.leaf_size);

  std::size_t node_num = 0;
  std::size_t leaf_start = 0;
  recursive_convert(tree, search_tree, node_num, leaf_start, n_nodes - 1);

  return search_tree;
}

template <typename Idx>
std::vector<SearchTree2<Idx>>
convert_rp_forest(const std::vector<RPTree2<Idx>> &rp_forest,
                  std::size_t n_points, std::size_t ndim) {
  std::vector<SearchTree2<Idx>> search_forest;
  search_forest.reserve(rp_forest.size());
  for (const auto &rp_tree : rp_forest) {
    search_forest.push_back(convert_tree_format(rp_tree, n_points, ndim));
  }
  return search_forest;
}

} // namespace tdoann

#endif // TDOANN_RPTREE2_H

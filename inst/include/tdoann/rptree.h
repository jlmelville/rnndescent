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

#ifndef TDOANN_RPTREE_H
#define TDOANN_RPTREE_H

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

template <typename In, typename Idx> struct RPTree {
  using Index = Idx;

  std::vector<std::vector<In>> hyperplanes = {};
  std::vector<In> offsets = {};
  std::vector<std::pair<std::size_t, std::size_t>> children = {};
  std::vector<std::vector<Idx>> indices = {};
  std::size_t leaf_size = 0;
  std::size_t ndim = 0;

  RPTree() = default;

  // Pre-allocate memory based on a rough best-case lower bound of nodes
  // (balanced tree)
  RPTree(std::size_t num_indices, std::size_t leaf_size, std::size_t ndim)
      : ndim(ndim) {
    // expression takes advantage of shifting the dividend by one so the
    // division truncation towards zero effectively rounds up to the nearest
    // integer. Avoids the cast and ceil in the equivalent (and clearer):
    // std::ceil(static_cast<double>(N) / L);
    // also need to safe guard if all indices can fit in one node
    std::size_t min_nodes =
        num_indices <= leaf_size ? 1 : (num_indices / (2 * leaf_size) + 1) - 1;

    hyperplanes.reserve(min_nodes);
    offsets.reserve(min_nodes);
    children.reserve(min_nodes);
    indices.reserve(min_nodes);
  }

  void add_node(const std::vector<In> &hyperplane, In offset,
                std::size_t left_node_num, std::size_t right_node_num) {
    // indices is never read from
    static const std::vector<Idx> dummy_indices =
        std::vector<Idx>(0, static_cast<Idx>(-1));
    indices.push_back(dummy_indices);

    // children is checked during leaf array construction
    // hyperplanes, offsets and children are used in SearchTree conversion
    hyperplanes.push_back(hyperplane);
    offsets.push_back(offset);
    children.emplace_back(left_node_num, right_node_num);
  }

  void add_leaf(const std::vector<Idx> &indices_) {
    // hyperplane and offset are never read anywhere
    static const std::vector<In> dummy_hyperplane =
        std::vector<In>(0, static_cast<In>(-1));
    static const In dummy_offset = std::numeric_limits<In>::quiet_NaN();
    hyperplanes.push_back(dummy_hyperplane);
    offsets.push_back(dummy_offset);

    // get_leaves_from_tree and recursive_convert looks for .first == sentinel
    // .second is ignored
    static const std::pair<std::size_t, std::size_t> dummy_child = {
        static_cast<std::size_t>(-1), static_cast<std::size_t>(-1)};
    children.push_back(dummy_child);

    // used in leaf array, conversion, search tree
    indices.push_back(indices_);
    leaf_size = std::max(leaf_size, indices_.size());
  }

  bool is_leaf(std::size_t i) const {
    return children[i].first == static_cast<std::size_t>(-1);
  }
};

template <typename Idx>
std::pair<Idx, Idx> select_random_points(const std::vector<Idx> &indices,
                                         RandomIntGenerator<Idx> &rng) {
  const std::size_t n_points = indices.size();

  Idx left_index = rng.rand_int(n_points);
  Idx right_index = rng.rand_int(n_points - 1);
  if (left_index == right_index) {
    ++right_index;
  }

  return {left_index, right_index};
}

template <typename In, typename Idx>
uint8_t select_side(typename std::vector<In>::const_iterator data_it,
                    const std::vector<In> &hyperplane_vector,
                    In hyperplane_offset, RandomIntGenerator<Idx> &rng) {
  constexpr In EPS = 1e-8;

  In margin =
      std::inner_product(hyperplane_vector.begin(), hyperplane_vector.end(),
                         data_it, hyperplane_offset);
  if (std::abs(margin) < EPS) {
    return rng.rand_int(2);
  }
  return margin > 0 ? 0 : 1;
}

template <typename In, typename Idx>
void split_indices(const std::vector<In> &data, std::size_t ndim,
                   const std::vector<Idx> &indices,
                   const std::vector<In> &hyperplane_vector,
                   In hyperplane_offset, std::vector<Idx> &indices_left,
                   std::vector<Idx> &indices_right,
                   RandomIntGenerator<Idx> &rng) {
  std::vector<uint8_t> side(indices.size(), 0);
  std::size_t n_left = 0;
  std::size_t n_right = 0;

  for (std::size_t i = 0; i < indices.size(); ++i) {
    side[i] = select_side(data.begin() + indices[i] * ndim, hyperplane_vector,
                          hyperplane_offset, rng);
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

template <typename In, typename Idx>
std::tuple<std::vector<Idx>, std::vector<Idx>, std::vector<In>, In>
euclidean_random_projection_split(const std::vector<In> &data, std::size_t ndim,
                                  const std::vector<Idx> &indices,
                                  RandomIntGenerator<Idx> &rng) {
  auto [left_index, right_index] = select_random_points(indices, rng);

  Idx left = indices[left_index] * ndim;
  Idx right = indices[right_index] * ndim;

  // Compute hyperplane vector and offset
  std::vector<In> hyperplane_vector(ndim);

  In hyperplane_offset = 0.0;
  In sum = 0.0; // aux variable to avoid repeated division inside the loop
  for (std::size_t d = 0; d < ndim; ++d) {
    hyperplane_vector[d] = data[left + d] - data[right + d];
    sum += hyperplane_vector[d] * (data[left + d] + data[right + d]);
  }
  hyperplane_offset -= sum / 2.0;

  std::vector<Idx> indices_left;
  std::vector<Idx> indices_right;
  split_indices(data, ndim, indices, hyperplane_vector, hyperplane_offset,
                indices_left, indices_right, rng);

  return std::make_tuple(std::move(indices_left), std::move(indices_right),
                         std::move(hyperplane_vector),
                         std::move(hyperplane_offset));
}

template <typename In, typename Idx>
std::tuple<std::vector<Idx>, std::vector<Idx>, std::vector<In>, In>
angular_random_projection_split(const std::vector<In> &data, size_t ndim,
                                const std::vector<Idx> &indices,
                                RandomIntGenerator<Idx> &rng) {
  constexpr In EPS = 1e-8;

  auto [left_index, right_index] = select_random_points(indices, rng);

  Idx left = indices[left_index] * ndim;
  Idx right = indices[right_index] * ndim;

  // Compute normal vector to the hyperplane
  In left_norm = 0.0;
  In right_norm = 0.0;
  for (size_t d = 0; d < ndim; ++d) {
    left_norm += data[left + d] * data[left + d];
    right_norm += data[right + d] * data[right + d];
  }
  left_norm = std::sqrt(left_norm);
  right_norm = std::sqrt(right_norm);

  if (std::abs(left_norm) < EPS) {
    left_norm = 1.0;
  }
  if (std::abs(right_norm) < EPS) {
    right_norm = 1.0;
  }

  std::vector<In> hyperplane_vector(ndim);
  In hyperplane_norm = 0.0;
  for (size_t d = 0; d < ndim; ++d) {
    hyperplane_vector[d] =
        (data[left + d] / left_norm) - (data[right + d] / right_norm);
    hyperplane_norm += hyperplane_vector[d] * hyperplane_vector[d];
  }
  hyperplane_norm = std::sqrt(hyperplane_norm);
  if (std::abs(hyperplane_norm) < EPS) {
    hyperplane_norm = 1.0;
  }

  for (size_t d = 0; d < ndim; ++d) {
    hyperplane_vector[d] /= hyperplane_norm;
  }

  std::vector<Idx> indices_left;
  std::vector<Idx> indices_right;
  split_indices(data, ndim, indices, hyperplane_vector, In(0), indices_left,
                indices_right, rng);

  return std::make_tuple(std::move(indices_left), std::move(indices_right),
                         std::move(hyperplane_vector), In(0));
}

template <typename In, typename Idx, typename SplitFunc>
void make_tree_recursive(const std::vector<In> &data, std::size_t ndim,
                         const std::vector<Idx> &indices, RPTree<In, Idx> &tree,
                         RandomIntGenerator<Idx> &rng, SplitFunc split_function,
                         uint32_t leaf_size, uint32_t max_depth) {
  if (indices.size() > leaf_size && max_depth > 0) {

    auto [left_indices, right_indices, hyperplane, offset] =
        split_function(data, ndim, indices, rng);

    make_tree_recursive(data, ndim, left_indices, tree, rng, split_function,
                        leaf_size, max_depth - 1);

    std::size_t left_node_num = tree.indices.size() - 1;

    make_tree_recursive(data, ndim, right_indices, tree, rng, split_function,
                        leaf_size, max_depth - 1);

    std::size_t right_node_num = tree.indices.size() - 1;

    tree.add_node(hyperplane, offset, left_node_num, right_node_num);
  } else {
    // leaf node
    tree.add_leaf(indices);
  }
}

template <typename In, typename Idx>
RPTree<In, Idx> make_dense_tree(const std::vector<In> &data, std::size_t ndim,
                                RandomIntGenerator<Idx> &rng,
                                uint32_t leaf_size, bool angular) {
  std::vector<Idx> indices(data.size() / ndim);
  std::iota(indices.begin(), indices.end(), 0);

  RPTree<In, Idx> tree(indices.size(), leaf_size, ndim);
  constexpr uint32_t max_depth = 100;
  if (angular) {

    auto angular_splitter = [](const auto &data, auto ndim, const auto &indices,
                               auto &rng) {
      return angular_random_projection_split(data, ndim, indices, rng);
    };

    make_tree_recursive(data, ndim, indices, tree, rng, angular_splitter,
                        leaf_size, max_depth);
  } else {
    auto euclidean_splitter = [](const auto &data, auto ndim,
                                 const auto &indices, auto &rng) {
      return euclidean_random_projection_split(data, ndim, indices, rng);
    };

    make_tree_recursive(data, ndim, indices, tree, rng, euclidean_splitter,
                        leaf_size, max_depth);
  }
  return tree;
}

template <typename In, typename Idx>
std::vector<RPTree<In, Idx>>
make_forest(const std::vector<In> &data, std::size_t ndim, uint32_t n_trees,
            uint32_t leaf_size, ParallelRandomIntProvider<Idx> &parallel_rand,
            bool angular, std::size_t n_threads, ProgressBase &progress,
            const Executor &executor) {
  std::vector<RPTree<In, Idx>> rp_forest(n_trees);

  parallel_rand.initialize();

  auto worker = [&](std::size_t begin, std::size_t end) {
    auto rng = parallel_rand.get_parallel_instance(end);
    for (auto i = begin; i < end; ++i) {
      rp_forest[i] = make_dense_tree(data, ndim, *rng, leaf_size, angular);
    }
  };

  progress.set_n_iters(1);
  ExecutionParams exec_params{};
  dispatch_work(worker, n_trees, n_threads, exec_params, progress, executor);

  return rp_forest;
}

// KNN calculation

// Find the largest leaf size in the forest
template <typename Forest>
std::size_t find_max_leaf_size(const Forest &rp_forest) {
  auto it = std::max_element(
      rp_forest.begin(), rp_forest.end(),
      [](const auto &a, const auto &b) { return a.leaf_size < b.leaf_size; });
  return it->leaf_size;
}

template <typename Tree>
std::vector<typename Tree::Index>
get_leaves_from_tree(const Tree &tree, std::size_t max_leaf_size) {
  using Idx = typename Tree::Index;

  std::size_t n_leaves = 0;
  for (std::size_t i = 0; i < tree.children.size(); ++i) {
    if (tree.is_leaf(i)) {
      ++n_leaves;
    }
  }

  constexpr auto idx_sentinel = static_cast<Idx>(-1);

  std::vector<Idx> leaf_indices(n_leaves * max_leaf_size, idx_sentinel);
  std::size_t insert_position = 0;
  for (std::size_t i = 0; i < tree.children.size(); ++i) {
    if (tree.is_leaf(i)) {
      const auto &indices = tree.indices[i];
      std::copy(indices.begin(), indices.end(),
                leaf_indices.begin() + insert_position);
      insert_position += max_leaf_size;
    }
  }

  return leaf_indices;
}

template <typename Tree>
std::vector<typename Tree::Index>
get_leaves_from_forest(const std::vector<Tree> &forest,
                       std::size_t max_leaf_size) {
  using Idx = typename Tree::Index;

  // Calculate total number of leaves and reserve space
  std::size_t total_leaves = 0;
  for (const auto &tree : forest) {
    for (std::size_t i = 0; i < tree.children.size(); ++i) {
      if (tree.is_leaf(i)) {
        ++total_leaves;
      }
    }
  }

  std::vector<Idx> leaf_indices;
  leaf_indices.reserve(total_leaves * max_leaf_size);

  // Concatenate leaves from each tree
  for (const auto &tree : forest) {
    auto tree_leaves = get_leaves_from_tree(tree, max_leaf_size);
    leaf_indices.insert(leaf_indices.end(), tree_leaves.begin(),
                        tree_leaves.end());
  }

  return leaf_indices;
}

template <typename Out, typename Idx>
void generate_leaf_updates(
    const BaseDistance<Out, Idx> &distance,
    const NNHeap<Out, Idx> &current_graph, const std::vector<Idx> &leaves,
    std::size_t max_leaf_size,
    std::vector<std::vector<std::tuple<Idx, Idx, Out>>> &updates,
    std::size_t neighbor_begin, std::size_t begin, std::size_t end) {
  constexpr auto npos = static_cast<Idx>(-1);

  for (std::size_t n = begin; n < end; ++n) {
    auto leaf_begin = leaves.begin() + n * max_leaf_size;
    auto leaf_end = leaf_begin + max_leaf_size;
    auto &leaf_updates = updates[n];

    for (auto i = leaf_begin; i != leaf_end; ++i) {
      Idx p = *i;
      if (p == npos) {
        break;
      }

      // if neighbor_begin == 0 then we consider an item to be a neighbor of
      // itself, otherwise neighbor_begin = 1 and only non-degenerate pairs
      // are considered
      for (auto j = i + neighbor_begin; j != leaf_end; ++j) {
        Idx q = *j;
        if (q == npos) {
          break;
        }

        const auto d = distance.calculate(p, q);
        if (current_graph.accepts_either(p, q, d)) {
          leaf_updates.emplace_back(p, q, d);
        }
      }
    }
  }
}

template <typename Out, typename Idx>
void init_rp_tree(const BaseDistance<Out, Idx> &distance,
                  NNHeap<Out, Idx> &current_graph,
                  const std::vector<Idx> &leaves, std::size_t max_leaf_size,
                  bool include_self, std::size_t n_threads,
                  ProgressBase &progress, const Executor &executor) {

  const std::size_t n_leaves = leaves.size() / max_leaf_size;
  std::vector<std::vector<std::tuple<Idx, Idx, Out>>> updates(n_leaves);

  // if include_self = true, then an item can be a neighbor of itself
  std::size_t neighbor_begin = include_self ? 0 : 1;

  auto worker = [&distance, &current_graph, &leaves, &updates, neighbor_begin,
                 max_leaf_size](std::size_t begin, std::size_t end) {
    generate_leaf_updates(distance, current_graph, leaves, max_leaf_size,
                          updates, neighbor_begin, begin, end);
  };
  auto after_worker = [&current_graph, &updates](std::size_t, std::size_t) {
    for (const auto &block_updates : updates) {
      for (const auto &[p, q, d] : block_updates) {
        current_graph.checked_push_pair(p, d, q);
      }
    }
  };
  ExecutionParams exec_params{65536};
  progress.set_n_iters(1);
  dispatch_work(worker, after_worker, n_leaves, n_threads, exec_params,
                progress, executor);
}

template <typename Out, typename Idx>
auto init_rp_tree(const BaseDistance<Out, Idx> &distance,
                  const std::vector<Idx> &leaves, std::size_t max_leaf_size,
                  uint32_t n_nbrs, bool include_self, std::size_t n_threads,
                  ProgressBase &progress, const Executor &executor)
    -> NNHeap<Out, Idx> {
  NNHeap<Out, Idx> current_graph(distance.get_ny(), n_nbrs);

  init_rp_tree(distance, current_graph, leaves, max_leaf_size, include_self,
               n_threads, progress, executor);

  return current_graph;
}

// Index Building/Searching

// This looks a lot like the RPTree but differs in the following ways:
// 1. The indices vector is flattened.
// 2. The children pairs contain either:
//  a. if the node is not a leaf then the first item points to the "left" node
//     index, and the second item points to the "right" node index
//  b. if the node is a leaf then the first item points the start index into
//     the indices vector, and the second item points to (one past) the end
//     index, i.e. the leaf indices are in:
//     indices[children.first:children.second]
//    Node i is a leaf if offsets[i] is NaN.
// 3. Compared to the equivalent RPTree, the SearchTree is constructed via
//    pre-order traversal, so node i in the RPTree is NOT the same node in the
//    search tree. The SearchTree orders nodes so children always have a higher
//    index than the parent. This should result in better cache coherency during
//    a search.
template <typename In, typename Idx> struct SearchTree {
  using Index = Idx;

  std::vector<std::vector<In>> hyperplanes;
  std::vector<In> offsets;
  std::vector<std::pair<std::size_t, std::size_t>> children;
  std::vector<Idx> indices;
  Idx leaf_size;

  SearchTree() = default;

  SearchTree(std::size_t n_nodes, std::size_t n_points, std::size_t ndim,
             Idx lsize)
      : hyperplanes(n_nodes, std::vector<In>(ndim)),
        offsets(n_nodes, std::numeric_limits<In>::quiet_NaN()),
        children(n_nodes, std::make_pair(static_cast<std::size_t>(-1),
                                         static_cast<std::size_t>(-1))),
        indices(n_points, static_cast<Idx>(-1)), leaf_size(lsize) {}

  // transfer in data from e.g. R
  SearchTree(std::vector<std::vector<In>> hplanes, std::vector<In> offs,
             std::vector<std::pair<std::size_t, std::size_t>> chldrn,
             std::vector<Idx> inds, Idx lsize)
      : hyperplanes(std::move(hplanes)), offsets(std::move(offs)),
        children(std::move(chldrn)), indices(std::move(inds)),
        leaf_size(lsize) {}

  bool is_leaf(std::size_t i) const { return std::isnan(offsets[i]); }
};

template <typename In, typename Idx>
std::pair<std::size_t, std::size_t>
recursive_convert(RPTree<In, Idx> &tree, SearchTree<In, Idx> &search_tree,
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
    search_tree.hyperplanes[node_num] = std::move(tree.hyperplanes[tree_node]);
    search_tree.offsets[node_num] = std::move(tree.offsets[tree_node]);
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

// move into this function, afterwards recursive calls pass by value
template <typename In, typename Idx>
void convert_tree(RPTree<In, Idx> tree, SearchTree<In, Idx> &search_tree,
                  std::size_t node_num, std::size_t leaf_start,
                  std::size_t tree_node) {
  // purposely ignore return value here
  recursive_convert(tree, search_tree, node_num, leaf_start, tree_node);
}

template <typename In, typename Idx>
SearchTree<In, Idx> convert_tree_format(RPTree<In, Idx> &&tree,
                                        std::size_t n_points,
                                        std::size_t ndim) {
  const auto n_nodes = tree.children.size();
  SearchTree<In, Idx> search_tree(n_nodes, n_points, ndim, tree.leaf_size);

  std::size_t node_num = 0;
  std::size_t leaf_start = 0;
  convert_tree(std::move(tree), search_tree, node_num, leaf_start, n_nodes - 1);

  return search_tree;
}

template <typename In, typename Idx>
std::vector<SearchTree<In, Idx>>
convert_rp_forest(std::vector<RPTree<In, Idx>> &rp_forest,
                  std::size_t n_points, std::size_t ndim) {
  std::vector<SearchTree<In, Idx>> search_forest;
  search_forest.reserve(rp_forest.size());
  for (auto &rp_tree : rp_forest) {
    search_forest.push_back(
        convert_tree_format(std::move(rp_tree), n_points, ndim));
  }
  return search_forest;
}

// Searching

template <typename In, typename Idx>
std::pair<std::size_t, std::size_t>
search_leaf_range(const SearchTree<In, Idx> &tree,
                  typename std::vector<In>::const_iterator obs_it,
                  RandomIntGenerator<Idx> &rng) {
  Idx current_node = 0;

  while (true) {
    auto child_pair = tree.children[current_node];

    // it's a leaf: child_pair contains pointers into the indices
    if (tree.is_leaf(current_node)) {
      return child_pair;
    }

    // it's a node, find the child to go to
    auto side = select_side(obs_it, tree.hyperplanes[current_node],
                            tree.offsets[current_node], rng);
    if (side == 0) {
      current_node = child_pair.first; // go left
    } else {
      current_node = child_pair.second; // go right
    }
  }
}

template <typename In, typename Idx>
std::vector<Idx> search_indices(const SearchTree<In, Idx> &tree,
                                typename std::vector<In>::const_iterator obs_it,
                                RandomIntGenerator<Idx> &rng) {
  std::pair<std::size_t, std::size_t> range =
      search_leaf_range(tree, obs_it, rng);
  std::vector<Idx> leaf_indices(tree.indices.begin() + range.first,
                                tree.indices.begin() + range.second);
  return leaf_indices;
}

template <typename In, typename Out, typename Idx>
void search_tree_heap_cache(const SearchTree<In, Idx> &tree,
                            const VectorDistance<In, Out, Idx> &distance, Idx i,
                            RandomIntGenerator<Idx> &rng,
                            NNHeap<Out, Idx> &current_graph,
                            std::unordered_set<Idx> &seen) {
  std::vector<Idx> leaf_indices = search_indices(tree, distance.get_y(i), rng);

  for (auto &idx : leaf_indices) {
    if (seen.find(idx) == seen.end()) { // not-contains
      const auto d = distance.calculate(idx, i);
      current_graph.checked_push(i, d, idx);
      seen.insert(idx);
    }
  }
}

template <typename In, typename Out, typename Idx>
void search_tree_heap(const SearchTree<In, Idx> &tree,
                      const VectorDistance<In, Out, Idx> &distance, Idx i,
                      RandomIntGenerator<Idx> &rng,
                      NNHeap<Out, Idx> &current_graph) {
  std::vector<Idx> leaf_indices = search_indices(tree, distance.get_y(i), rng);

  for (auto &idx : leaf_indices) {
    const auto d = distance.calculate(idx, i);
    current_graph.checked_push(i, d, idx);
  }
}

template <typename In, typename Out, typename Idx>
void search_forest_cache(const std::vector<SearchTree<In, Idx>> &forest,
                         const VectorDistance<In, Out, Idx> &distance, Idx i,
                         RandomIntGenerator<Idx> &rng,
                         NNHeap<Out, Idx> &current_graph) {
  std::unordered_set<Idx> seen;

  for (const auto &tree : forest) {
    search_tree_heap_cache(tree, distance, i, rng, current_graph, seen);
  }
}

template <typename In, typename Out, typename Idx>
void search_forest(const std::vector<SearchTree<In, Idx>> &forest,
                   const VectorDistance<In, Out, Idx> &distance, Idx i,
                   RandomIntGenerator<Idx> &rng,
                   NNHeap<Out, Idx> &current_graph) {
  for (const auto &tree : forest) {
    search_tree_heap(tree, distance, i, rng, current_graph);
  }
}

template <typename In, typename Out, typename Idx>
NNHeap<Out, Idx>
search_forest(const std::vector<SearchTree<In, Idx>> &forest,
              const VectorDistance<In, Out, Idx> &distance, uint32_t n_nbrs,
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

// Search Tree scoring/filtering

template <typename Idx>
std::size_t compute_overlap(const std::unordered_set<Idx> &indices_set,
                            const std::vector<Idx> &nn_indices,
                            std::size_t n_neighbors) {
  constexpr auto sentinel = static_cast<Idx>(-1);
  std::size_t overlap = 0;

  for (const auto &idx : indices_set) {
    if (idx == sentinel) {
      continue;
    }

    auto nn_start = nn_indices.begin() + idx * n_neighbors;
    auto nn_end = nn_indices.begin() + (idx + 1) * n_neighbors;

    for (auto it = nn_start; it != nn_end; ++it) {
      auto nn_idx = *it;
      if (nn_idx == sentinel) {
        continue;
      }

      if (indices_set.find(nn_idx) != indices_set.end()) {
        ++overlap;
      }
    }
  }

  return overlap;
}

template <typename Tree>
double score_tree(const Tree &tree,
                  const std::vector<typename Tree::Index> &nn_indices,
                  uint32_t n_neighbors) {
  std::size_t overlap_sum = 0;
  for (std::size_t i = 0; i < tree.children.size(); ++i) {
    if (tree.is_leaf(i)) {
      auto [start, end] = tree.children[i];
      auto leaf_start = tree.indices.begin() + start;
      auto leaf_end = tree.indices.begin() + end;
      std::unordered_set<typename Tree::Index> indices_set(leaf_start,
                                                           leaf_end);
      overlap_sum += compute_overlap(indices_set, nn_indices, n_neighbors);
    }
  }
  return overlap_sum / static_cast<double>(nn_indices.size() / n_neighbors);
}

template <typename Tree>
std::vector<double>
score_forest(const std::vector<Tree> &forest,
             const std::vector<typename Tree::Index> &nn_indices,
             uint32_t n_neighbors, std::size_t n_threads,
             ProgressBase &progress, const Executor &executor) {
  const auto n_trees = forest.size();

  std::vector<double> scores(n_trees);

  auto worker = [&](std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; ++i) {
      scores[i] = score_tree(forest[i], nn_indices, n_neighbors);
    }
  };

  progress.set_n_iters(1);
  ExecutionParams exec_params{};
  dispatch_work(worker, n_trees, n_threads, exec_params, progress, executor);

  return scores;
}

// given the overlap scores for a search forest from score_forest, return the
// top n best scoring
template <typename Tree>
std::vector<Tree> filter_top_n_trees(const std::vector<Tree> &forest,
                                     const std::vector<double> &scores,
                                     std::size_t n) {
  // get the order of the scores
  std::vector<std::size_t> order(forest.size());
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + n, order.end(),
                    [&scores](std::size_t a, std::size_t b) {
                      return scores[a] > scores[b];
                    });
  // return the top n trees given by the order
  std::vector<Tree> top_n_trees;
  top_n_trees.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    top_n_trees.push_back(forest[order[i]]);
  }

  return top_n_trees;
}

} // namespace tdoann

#endif // TDOANN_RPTREE_H

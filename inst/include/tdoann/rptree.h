//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2023 James Melville
//
//  This file is part of rnndescent
//
//  rnndescent is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  rnndescent is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with rnndescent.  If not, see <http://www.gnu.org/licenses/>.

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
  std::vector<std::vector<In>> hyperplanes;
  std::vector<In> offsets;
  std::vector<std::pair<std::size_t, std::size_t>> children;
  std::vector<std::vector<Idx>> indices;
  std::size_t leaf_size;

  RPTree() = default; // Exists only so we can pre-allocate a vector

  RPTree(std::vector<std::vector<In>> hplanes, std::vector<In> offs,
         std::vector<std::pair<std::size_t, std::size_t>> chldrn,
         std::vector<std::vector<Idx>> p_indices, std::size_t leaf_siz)
      : hyperplanes(std::move(hplanes)), offsets(std::move(offs)),
        children(std::move(chldrn)), indices(std::move(p_indices)),
        leaf_size(leaf_siz) {}
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
  right_index = right_index % n_points;

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
    auto index = indices[i];
    if (side[i] == 0) {
      indices_left[n_left] = index;
      ++n_left;
    } else {
      indices_right[n_right] = index;
      ++n_right;
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
  for (size_t d = 0; d < ndim; ++d) {
    hyperplane_vector[d] = data[left + d] - data[right + d];
    hyperplane_offset -=
        hyperplane_vector[d] * (data[left + d] + data[right + d]) / 2.0;
  }

  std::vector<Idx> indices_left;
  std::vector<Idx> indices_right;
  split_indices(data, ndim, indices, hyperplane_vector, hyperplane_offset,
                indices_left, indices_right, rng);

  return std::make_tuple(indices_left, indices_right, hyperplane_vector,
                         hyperplane_offset);
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

  return std::make_tuple(indices_left, indices_right, hyperplane_vector, In(0));
}

template <typename In, typename Idx, typename SplitFunc>
void make_tree_recursive(
    const std::vector<In> &data, std::size_t ndim,
    const std::vector<Idx> &indices, std::vector<std::vector<In>> &hyperplanes,
    std::vector<In> &offsets,
    std::vector<std::pair<std::size_t, std::size_t>> &children,
    std::vector<std::vector<Idx>> &point_indices, RandomIntGenerator<Idx> &rng,
    SplitFunc split_function, uint32_t leaf_size = 30,
    uint32_t max_depth = 100) {
  constexpr auto child_sentinel = static_cast<std::size_t>(-1);
  constexpr auto idx_sentinel = static_cast<Idx>(-1);
  constexpr auto minus_one = static_cast<In>(-1);
  const auto qnan = std::numeric_limits<In>::quiet_NaN();

  if (indices.size() > leaf_size && max_depth > 0) {

    auto [left_indices, right_indices, hyperplane, offset] =
        split_function(data, ndim, indices, rng);

    make_tree_recursive(data, ndim, left_indices, hyperplanes, offsets,
                        children, point_indices, rng, split_function, leaf_size,
                        max_depth - 1);

    std::size_t left_node_num = point_indices.size() - 1;

    make_tree_recursive(data, ndim, right_indices, hyperplanes, offsets,
                        children, point_indices, rng, split_function, leaf_size,
                        max_depth - 1);

    std::size_t right_node_num = point_indices.size() - 1;

    hyperplanes.push_back(hyperplane);
    offsets.push_back(offset);
    children.emplace_back(left_node_num, right_node_num);
    point_indices.emplace_back(std::vector<Idx>(1, idx_sentinel));
  } else {
    // leaf node
    hyperplanes.emplace_back(std::vector<In>(1, minus_one));
    offsets.push_back(qnan);
    children.emplace_back(child_sentinel, child_sentinel);
    point_indices.push_back(indices);
  }
}

template <typename In, typename Idx>
RPTree<In, Idx> make_dense_tree(const std::vector<In> &data, std::size_t ndim,
                                RandomIntGenerator<Idx> &rng,
                                uint32_t leaf_size, bool angular) {

  std::vector<Idx> indices(data.size() / ndim);
  std::iota(indices.begin(), indices.end(), 0);

  std::vector<std::vector<In>> hyperplanes;
  std::vector<In> offsets;
  std::vector<std::pair<std::size_t, std::size_t>> children;
  std::vector<std::vector<Idx>> point_indices;

  if (angular) {

    auto angular_splitter = [](const auto &data, auto ndim, const auto &indices,
                               auto &rng) {
      return angular_random_projection_split(data, ndim, indices, rng);
    };

    make_tree_recursive(data, ndim, indices, hyperplanes, offsets, children,
                        point_indices, rng, angular_splitter, leaf_size);
  } else {
    auto euclidean_splitter = [](const auto &data, auto ndim,
                                 const auto &indices, auto &rng) {
      return euclidean_random_projection_split(data, ndim, indices, rng);
    };

    make_tree_recursive(data, ndim, indices, hyperplanes, offsets, children,
                        point_indices, rng, euclidean_splitter, leaf_size);
  }

  std::size_t max_leaf_size = leaf_size;
  for (const auto &points : point_indices) {
    max_leaf_size = std::max(max_leaf_size, points.size());
  }

  return RPTree<In, Idx>(std::move(hyperplanes), std::move(offsets),
                         std::move(children), std::move(point_indices),
                         max_leaf_size);
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
  ExecutionParams exec_params{n_threads};
  dispatch_work(worker, n_trees, n_threads, exec_params, progress, executor);

  return rp_forest;
}

// KNN calculation

// Find the largest leaf size in the forest
template <typename In, typename Idx>
std::size_t find_max_leaf_size(const std::vector<RPTree<In, Idx>> &rp_forest) {
  auto it = std::max_element(rp_forest.begin(), rp_forest.end(),
                             [](const RPTree<In, Idx> &a, decltype(a) b) {
                               return a.leaf_size < b.leaf_size;
                             });
  return it->leaf_size;
}

template <typename In, typename Idx>
std::vector<Idx> get_leaves_from_tree(const RPTree<In, Idx> &tree,
                                      std::size_t max_leaf_size) {
  constexpr auto sentinel = static_cast<std::size_t>(-1);
  constexpr auto idx_sentinel = static_cast<Idx>(-1);

  std::size_t n_leaves = 0;
  for (const auto &child : tree.children) {
    // child pairs contain either both sentinels or neither so we only need to
    // check one of the pair to detect a leaf
    if (child.first == sentinel) {
      ++n_leaves;
    }
  }

  std::vector<Idx> leaf_indices;
  leaf_indices.reserve(n_leaves * max_leaf_size);
  for (std::size_t i = 0; i < tree.children.size(); ++i) {
    if (tree.children[i].first == sentinel) {
      const auto &indices = tree.indices[i];
      leaf_indices.insert(leaf_indices.end(), indices.begin(), indices.end());
      const std::size_t padding = max_leaf_size - indices.size();
      leaf_indices.insert(leaf_indices.end(), padding, idx_sentinel);
    }
  }

  return leaf_indices;
}

template <typename In, typename Idx>
std::vector<Idx>
get_leaves_from_forest(const std::vector<RPTree<In, Idx>> &forest,
                       std::size_t max_leaf_size) {
  constexpr auto sentinel = static_cast<std::size_t>(-1);

  std::vector<Idx> leaf_indices;

  // Calculate total number of leaves and reserve space
  std::size_t total_leaves = 0;
  for (const auto &tree : forest) {
    for (const auto &child : tree.children) {
      if (child.first == sentinel) {
        ++total_leaves;
      }
    }
  }

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
  std::vector<std::vector<In>> hyperplanes;
  std::vector<In> offsets;
  std::vector<std::pair<std::size_t, std::size_t>> children;
  std::vector<Idx> indices;
  Idx leaf_size;

  SearchTree() = default;

  SearchTree(std::vector<std::vector<In>> hplanes, std::vector<In> offs,
             std::vector<std::pair<std::size_t, std::size_t>> chldrn,
             std::vector<Idx> inds, Idx lsize)
      : hyperplanes(std::move(hplanes)), offsets(std::move(offs)),
        children(std::move(chldrn)), indices(std::move(inds)),
        leaf_size(lsize) {}
};

template <typename In, typename Idx>
std::pair<std::size_t, std::size_t>
recursive_convert(const RPTree<In, Idx> &tree,
                  std::vector<std::vector<In>> &hyperplanes,
                  std::vector<In> &offsets,
                  std::vector<std::pair<std::size_t, std::size_t>> &children,
                  std::vector<Idx> &indices, std::size_t node_num,
                  std::size_t leaf_start, std::size_t tree_node) {

  constexpr auto leaf_sentinel = static_cast<std::size_t>(-1);

  if (tree.children[tree_node].first == leaf_sentinel) {
    std::size_t leaf_end = leaf_start + tree.indices[tree_node].size();
    children[node_num] = std::make_pair(leaf_start, leaf_end);
    std::copy(tree.indices[tree_node].begin(), tree.indices[tree_node].end(),
              indices.begin() + leaf_start);
    return {node_num, leaf_end};
  } else {
    hyperplanes[node_num] = tree.hyperplanes[tree_node];
    offsets[node_num] = tree.offsets[tree_node];
    children[node_num].first = node_num + 1;
    std::size_t old_node_num = node_num;

    std::tie(node_num, leaf_start) = recursive_convert(
        tree, hyperplanes, offsets, children, indices, node_num + 1, leaf_start,
        tree.children[tree_node].first);

    children[old_node_num].second = node_num + 1;

    return recursive_convert(tree, hyperplanes, offsets, children, indices,
                             node_num + 1, leaf_start,
                             tree.children[tree_node].second);
  }
}

template <typename In, typename Idx>
SearchTree<In, Idx> convert_tree_format(const RPTree<In, Idx> &tree,
                                        std::size_t n_points,
                                        std::size_t ndim) {
  constexpr auto idx_sentinel = static_cast<Idx>(-1);
  constexpr auto leaf_sentinel = static_cast<std::size_t>(-1);

  std::size_t n_nodes = tree.children.size();
  std::size_t n_leaves = 0;
  for (const auto &child_pair : tree.children) {
    if (child_pair.first == leaf_sentinel) {
      n_leaves++;
    }
  }

  std::vector<std::vector<In>> hyperplanes(n_nodes, std::vector<In>(ndim));
  std::vector<In> offsets(n_nodes, std::numeric_limits<In>::quiet_NaN());
  std::vector<std::pair<std::size_t, std::size_t>> children(
      n_nodes, std::make_pair(leaf_sentinel, leaf_sentinel));
  std::vector<Idx> indices(n_points, idx_sentinel);

  std::size_t node_num = 0;
  std::size_t leaf_start = 0;
  recursive_convert(tree, hyperplanes, offsets, children, indices, node_num,
                    leaf_start, tree.children.size() - 1);

  return SearchTree<In, Idx>(hyperplanes, offsets, children, indices,
                             tree.leaf_size);
}

template <typename In, typename Idx>
std::vector<SearchTree<In, Idx>>
convert_rp_forest(const std::vector<RPTree<In, Idx>> &rp_forest,
                  std::size_t n_points, std::size_t ndim) {
  std::vector<SearchTree<In, Idx>> search_forest;
  search_forest.reserve(rp_forest.size());
  for (const auto &rp_tree : rp_forest) {
    search_forest.push_back(convert_tree_format(rp_tree, n_points, ndim));
  }
  return search_forest;
}

template <typename In, typename Idx>
std::pair<std::size_t, std::size_t>
search_leaf_range(const SearchTree<In, Idx> &tree,
                  typename std::vector<In>::const_iterator obs_it,
                  RandomIntGenerator<Idx> &rng) {
  Idx current_node = 0;

  while (true) {
    auto child_pair = tree.children[current_node];
    const auto &hyperplane_offset = tree.offsets[current_node];

    // it's a leaf: child_pair contains pointers into the indices
    if (std::isnan(hyperplane_offset)) {
      return child_pair;
    }

    auto side = select_side(obs_it, tree.hyperplanes[current_node],
                            hyperplane_offset, rng);
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
std::pair<std::vector<Idx>, std::vector<Out>>
search_tree(const SearchTree<In, Idx> &tree,
            const VectorDistance<In, Out, Idx> &distance, Idx i,
            RandomIntGenerator<Idx> &rng) {

  std::vector<Idx> leaf_indices = search_indices(tree, distance.get_y(i), rng);
  std::vector<Out> distances;
  distances.reserve(leaf_indices.size());
  for (auto &idx : leaf_indices) {
    distances.push_back(distance.calculate(idx, i));
  }

  return {leaf_indices, distances};
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
  ExecutionParams exec_params{n_threads};
  dispatch_work(worker, n_queries, n_threads, exec_params, progress, executor);

  return current_graph;
}

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

template <typename Idx, typename In>
double score_tree(const SearchTree<In, Idx> &tree,
                  const std::vector<Idx> &nn_indices, uint32_t n_neighbors) {
  std::size_t overlap_sum = 0;
  for (std::size_t i = 0; i < tree.children.size(); ++i) {
    auto [start, end] = tree.children[i];
    if (std::isnan(tree.offsets[i])) {
      auto leaf_start = tree.indices.begin() + start;
      auto leaf_end = tree.indices.begin() + end;
      std::unordered_set<Idx> indices_set(leaf_start, leaf_end);
      overlap_sum += compute_overlap(indices_set, nn_indices, n_neighbors);
    }
  }
  return overlap_sum / static_cast<double>(nn_indices.size() / n_neighbors);
}

template <typename Idx, typename In>
std::vector<double> score_forest(const std::vector<SearchTree<In, Idx>> &forest,
                                 const std::vector<Idx> &nn_indices,
                                 uint32_t n_neighbors, std::size_t n_threads,
                                 ProgressBase &progress,
                                 const Executor &executor) {
  const auto n_trees = forest.size();

  std::vector<double> scores(n_trees);

  auto worker = [&](std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; ++i) {
      scores[i] = score_tree(forest[i], nn_indices, n_neighbors);
    }
  };

  progress.set_n_iters(1);
  ExecutionParams exec_params{n_threads};
  dispatch_work(worker, n_trees, n_threads, exec_params, progress, executor);

  return scores;
}

// given the overlap scores for a search forest from score_forest, return the
// top n best scoring
template <typename Idx, typename In>
std::vector<SearchTree<Idx, In>>
filter_top_n_trees(const std::vector<SearchTree<Idx, In>> &forest,
                   const std::vector<double> &scores, std::size_t n) {

  // get the order of the scores
  std::vector<std::size_t> order(forest.size());
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + n, order.end(),
                    [&scores](std::size_t a, std::size_t b) {
                      return scores[a] > scores[b];
                    });
  // return the top n trees given by the order
  std::vector<SearchTree<Idx, In>> top_n_trees;
  top_n_trees.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    top_n_trees.push_back(forest[order[i]]);
  }

  return top_n_trees;
}

} // namespace tdoann

#endif // TDOANN_RPTREE_H

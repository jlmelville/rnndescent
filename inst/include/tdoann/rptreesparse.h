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

#ifndef TDOANN_RPTREESPARSE_H
#define TDOANN_RPTREESPARSE_H

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
#include "sparse.h"

namespace tdoann {

// Tree Building

template <typename In, typename Idx> struct SparseRPTree {
  using Index = Idx;

  std::vector<std::vector<std::size_t>> hyperplanes_ind = {};
  std::vector<std::vector<In>> hyperplanes_data = {};
  std::vector<In> offsets = {};
  std::vector<std::pair<std::size_t, std::size_t>> children = {};
  std::vector<std::vector<Idx>> indices = {};
  std::size_t leaf_size = 0;
  std::size_t ndim = 0;

  SparseRPTree() = default;

  SparseRPTree(std::size_t num_indices, std::size_t leaf_size, std::size_t ndim)
      : ndim(ndim) {
    std::size_t min_nodes =
        num_indices <= leaf_size ? 1 : (num_indices / (2 * leaf_size) + 1) - 1;
    hyperplanes_ind.reserve(min_nodes);
    hyperplanes_data.reserve(min_nodes);
    offsets.reserve(min_nodes);
    children.reserve(min_nodes);
    indices.reserve(min_nodes);
  }

  void add_node(const std::vector<std::size_t> &hyperplane_ind,
                const std::vector<In> &hyperplane_data, In offset,
                std::size_t left_node_num, std::size_t right_node_num) {
    static const std::vector<Idx> dummy_indices =
        std::vector<Idx>(0, static_cast<Idx>(-1));
    indices.push_back(dummy_indices);

    hyperplanes_ind.push_back(hyperplane_ind);
    hyperplanes_data.push_back(hyperplane_data);
    offsets.push_back(offset);
    children.emplace_back(left_node_num, right_node_num);
  }

  void add_leaf(const std::vector<Idx> &indices_) {
    static const std::vector<std::size_t> dummy_hyperplane_ind;
    hyperplanes_ind.push_back(dummy_hyperplane_ind);

    static const std::vector<In> dummy_hyperplane_data;
    hyperplanes_data.push_back(dummy_hyperplane_data);

    static const In dummy_offset = std::numeric_limits<In>::quiet_NaN();
    offsets.push_back(dummy_offset);

    static const std::pair<std::size_t, std::size_t> dummy_child = {
        static_cast<std::size_t>(-1), static_cast<std::size_t>(-1)};
    children.push_back(dummy_child);

    indices.push_back(indices_);
    leaf_size = std::max(leaf_size, indices_.size());
  }

  bool is_leaf(std::size_t i) const {
    return children[i].first == static_cast<std::size_t>(-1);
  }
};

// normalization functions needed for sparse angular splits
template <typename In, typename DataIt> In norm(DataIt start, DataIt end) {
  In result = 0.0;
  for (auto it = start; it != end; ++it) {
    result += (*it) * (*it);
  }
  return std::sqrt(result);
}

template <typename In, typename DataIt>
std::vector<In> normalize(DataIt start, DataIt end) {
  constexpr In EPS = 1e-8;
  In norm_val = norm<In>(start, end);
  if (std::abs(norm_val) < EPS) {
    norm_val = 1.0;
  }

  std::vector<In> normalized;
  std::transform(start, end, std::back_inserter(normalized),
                 [norm_val](const In &val) { return val / norm_val; });
  return normalized;
}

template <typename In, typename DataIt>
void normalize_inplace(DataIt start, DataIt end) {
  constexpr In EPS = 1e-8;

  In norm_val = norm<In>(start, end);
  if (std::abs(norm_val) < EPS) {
    norm_val = 1.0;
  }

  std::transform(start, end, start,
                 [norm_val](const In &val) { return val / norm_val; });
}

template <typename In, typename Idx>
uint8_t
select_side_sparse(typename std::vector<std::size_t>::const_iterator ind_start,
                   std::size_t ind_size,
                   typename std::vector<In>::const_iterator data_start,
                   const std::vector<std::size_t> &hyperplane_ind,
                   const std::vector<In> &hyperplane_data, In hyperplane_offset,
                   RandomIntGenerator<Idx> &rng) {
  constexpr In EPS = 1e-8;

  auto [_, mul_data] =
      sparse_mul<In>(hyperplane_ind.begin(), hyperplane_ind.size(),
                     hyperplane_data.begin(), ind_start, ind_size, data_start);

  In margin = hyperplane_offset;
  for (auto val : mul_data) {
    margin += val;
  }

  if (std::abs(margin) < EPS) {
    return rng.rand_int(2);
  }
  return margin > 0 ? 0 : 1;
}

template <typename In, typename Idx>
void split_indices_sparse(const std::vector<std::size_t> &ind,
                          const std::vector<std::size_t> &ptr,
                          const std::vector<In> &data,
                          const std::vector<Idx> &indices,
                          const std::vector<std::size_t> &hyperplane_ind,
                          const std::vector<In> &hyperplane_data,
                          In hyperplane_offset, std::vector<Idx> &indices_left,
                          std::vector<Idx> &indices_right,
                          RandomIntGenerator<Idx> &rng) {
  std::vector<uint8_t> side(indices.size(), 0);
  std::size_t n_left = 0;
  std::size_t n_right = 0;

  for (std::size_t i = 0; i < indices.size(); ++i) {
    Idx idx = indices[i];
    auto range_start = ptr[idx];
    auto range_end = ptr[idx + 1];
    side[i] =
        select_side_sparse(ind.begin() + range_start, range_end - range_start,
                           data.begin() + range_start, hyperplane_ind,
                           hyperplane_data, hyperplane_offset, rng);
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
std::tuple<std::vector<Idx>, std::vector<Idx>, std::vector<std::size_t>,
           std::vector<In>, In>
sparse_angular_random_projection_split(const std::vector<std::size_t> &ind,
                                       const std::vector<std::size_t> &ptr,
                                       const std::vector<In> &data,
                                       const std::vector<Idx> &indices,
                                       RandomIntGenerator<Idx> &rng) {
  auto [left_index, right_index] = select_random_points(indices, rng);

  Idx left = indices[left_index];
  Idx right = indices[right_index];

  const auto left_range_start = ptr[left];
  const auto left_range_end = ptr[left + 1];
  const auto left_size = left_range_end - left_range_start;
  const auto right_range_start = ptr[right];
  const auto right_range_end = ptr[right + 1];
  const auto right_size = right_range_end - right_range_start;

  const auto left_ind = ind.begin() + left_range_start;
  const auto right_ind = ind.begin() + right_range_start;

  const auto left_data = data.begin() + left_range_start;
  std::vector<In> normalized_left_data =
      normalize<In>(left_data, left_data + left_size);

  const auto right_data = data.begin() + right_range_start;
  std::vector<In> normalized_right_data =
      normalize<In>(right_data, right_data + right_size);

  auto [hyperplane_ind, hyperplane_data] =
      sparse_diff<In>(left_ind, left_size, normalized_left_data.begin(),
                      right_ind, right_size, normalized_right_data.begin());
  normalize_inplace<In>(hyperplane_data.begin(), hyperplane_data.end());

  In hyperplane_offset = 0.0;

  std::vector<Idx> indices_left;
  std::vector<Idx> indices_right;
  split_indices_sparse(ind, ptr, data, indices, hyperplane_ind, hyperplane_data,
                       hyperplane_offset, indices_left, indices_right, rng);

  return std::make_tuple(std::move(indices_left), std::move(indices_right),
                         std::move(hyperplane_ind), std::move(hyperplane_data),
                         std::move(hyperplane_offset));
}

template <typename In, typename Idx>
std::tuple<std::vector<Idx>, std::vector<Idx>, std::vector<std::size_t>,
           std::vector<In>, In>
sparse_euclidean_random_projection_split(const std::vector<std::size_t> &ind,
                                         const std::vector<std::size_t> &ptr,
                                         const std::vector<In> &data,
                                         const std::vector<Idx> &indices,
                                         RandomIntGenerator<Idx> &rng) {
  auto [left_index, right_index] = select_random_points(indices, rng);

  Idx left = indices[left_index];
  Idx right = indices[right_index];

  const auto left_range_start = ptr[left];
  const auto left_range_end = ptr[left + 1];
  const auto left_size = left_range_end - left_range_start;
  const auto right_range_start = ptr[right];
  const auto right_range_end = ptr[right + 1];
  const auto right_size = right_range_end - right_range_start;

  const auto left_ind = ind.begin() + left_range_start;
  const auto left_data = data.begin() + left_range_start;
  const auto right_ind = ind.begin() + right_range_start;
  const auto right_data = data.begin() + right_range_start;

  auto [hyperplane_ind, hyperplane_data] = sparse_diff<In>(
      left_ind, left_size, left_data, right_ind, right_size, right_data);

  auto [offset_ind, offset_data] = sparse_sum<In>(
      left_ind, left_size, left_data, right_ind, right_size, right_data);

  for (auto &val : offset_data) {
    val /= 2.0;
  }

  auto [_, mul_data] = sparse_mul<In>(
      hyperplane_ind.begin(), hyperplane_ind.size(), hyperplane_data.begin(),
      offset_ind.begin(), offset_ind.size(), offset_data.begin());

  In hyperplane_offset = 0.0;
  for (auto val : mul_data) {
    hyperplane_offset -= val;
  }

  std::vector<Idx> indices_left;
  std::vector<Idx> indices_right;
  split_indices_sparse(ind, ptr, data, indices, hyperplane_ind, hyperplane_data,
                       hyperplane_offset, indices_left, indices_right, rng);

  return std::make_tuple(std::move(indices_left), std::move(indices_right),
                         std::move(hyperplane_ind), std::move(hyperplane_data),
                         std::move(hyperplane_offset));
}

template <typename In, typename Idx, typename SplitFunc>
void make_sparse_tree_recursive(
    const std::vector<std::size_t> &ind, const std::vector<std::size_t> &ptr,
    const std::vector<In> &data, const std::vector<Idx> &indices,
    SparseRPTree<In, Idx> &tree, RandomIntGenerator<Idx> &rng,
    SplitFunc split_function, uint32_t leaf_size, uint32_t max_depth) {

  if (indices.size() > leaf_size && max_depth > 0) {
    auto [left_indices, right_indices, hyperplane_ind, hyperplane_data,
          offset] = split_function(ind, ptr, data, indices, rng);

    make_sparse_tree_recursive(ind, ptr, data, left_indices, tree, rng,
                               split_function, leaf_size, max_depth - 1);

    std::size_t left_node_num = tree.indices.size() - 1;

    make_sparse_tree_recursive(ind, ptr, data, right_indices, tree, rng,
                               split_function, leaf_size, max_depth - 1);

    std::size_t right_node_num = tree.indices.size() - 1;

    tree.add_node(hyperplane_ind, hyperplane_data, offset, left_node_num,
                  right_node_num);
  } else {
    // leaf node
    tree.add_leaf(indices);
  }
}

template <typename In, typename Idx>
SparseRPTree<In, Idx> make_sparse_tree(const std::vector<std::size_t> &ind,
                                       const std::vector<std::size_t> &ptr,
                                       const std::vector<In> &data,
                                       std::size_t ndim,
                                       RandomIntGenerator<Idx> &rng,
                                       uint32_t leaf_size, bool angular) {
  std::vector<Idx> indices(ptr.size() - 1);
  std::iota(indices.begin(), indices.end(), 0);

  SparseRPTree<In, Idx> tree(indices.size(), leaf_size, ndim);
  constexpr uint32_t max_depth = 100;

  if (angular) {
    auto splitter = [](const auto &ind, const auto &ptr, const auto &data,
                       auto &indices, auto &rng) {
      return sparse_angular_random_projection_split(ind, ptr, data, indices,
                                                    rng);
    };

    make_sparse_tree_recursive(ind, ptr, data, indices, tree, rng, splitter,
                               leaf_size, max_depth);
  } else {
    auto splitter = [](const auto &ind, const auto &ptr, const auto &data,
                       auto &indices, auto &rng) {
      return sparse_euclidean_random_projection_split(ind, ptr, data, indices,
                                                      rng);
    };

    make_sparse_tree_recursive(ind, ptr, data, indices, tree, rng, splitter,
                               leaf_size, max_depth);
  }
  return tree;
}

template <typename In, typename Idx>
std::vector<SparseRPTree<In, Idx>> make_sparse_forest(
    const std::vector<std::size_t> &inds,
    const std::vector<std::size_t> &indptr, const std::vector<In> &data,
    std::size_t ndim, uint32_t n_trees, uint32_t leaf_size,
    ParallelRandomIntProvider<Idx> &parallel_rand, bool angular,
    std::size_t n_threads, ProgressBase &progress, const Executor &executor) {
  std::vector<SparseRPTree<In, Idx>> rp_forest(n_trees);

  parallel_rand.initialize();

  auto worker = [&](std::size_t begin, std::size_t end) {
    auto rng = parallel_rand.get_parallel_instance(end);
    for (auto i = begin; i < end; ++i) {
      rp_forest[i] =
          make_sparse_tree(inds, indptr, data, ndim, *rng, leaf_size, angular);
    }
  };

  progress.set_n_iters(1);
  ExecutionParams exec_params{};
  dispatch_work(worker, n_trees, n_threads, exec_params, progress, executor);

  return rp_forest;
}

// Index Building/Searching

template <typename In, typename Idx> struct SparseSearchTree {
  using Index = Idx;

  std::vector<std::vector<std::size_t>> hyperplanes_ind = {};
  std::vector<std::vector<In>> hyperplanes_data = {};
  std::vector<In> offsets;
  std::vector<std::pair<std::size_t, std::size_t>> children;
  std::vector<Idx> indices;
  Idx leaf_size;

  SparseSearchTree() = default;

  SparseSearchTree(std::size_t n_nodes, std::size_t n_points, std::size_t ndim,
                   Idx lsize)
      : hyperplanes_ind(n_nodes, std::vector<std::size_t>(ndim)),
        hyperplanes_data(n_nodes, std::vector<In>(ndim)),
        offsets(n_nodes, std::numeric_limits<In>::quiet_NaN()),
        children(n_nodes, std::make_pair(static_cast<std::size_t>(-1),
                                         static_cast<std::size_t>(-1))),
        indices(n_points, static_cast<Idx>(-1)), leaf_size(lsize) {}

  // transfer in data from e.g. R
  SparseSearchTree(std::vector<std::vector<std::size_t>> hplanes_ind,
                   std::vector<std::vector<In>> hplanes_data,
                   std::vector<In> offs,
                   std::vector<std::pair<std::size_t, std::size_t>> chldrn,
                   std::vector<Idx> inds, Idx lsize)
      : hyperplanes_ind(std::move(hplanes_ind)),
        hyperplanes_data(std::move(hplanes_data)), offsets(std::move(offs)),
        children(std::move(chldrn)), indices(std::move(inds)),
        leaf_size(lsize) {}

  bool is_leaf(std::size_t i) const { return std::isnan(offsets[i]); }
};

template <typename In, typename Idx>
std::pair<std::size_t, std::size_t>
recursive_convert(SparseRPTree<In, Idx> &tree,
                  SparseSearchTree<In, Idx> &search_tree, std::size_t node_num,
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
    search_tree.hyperplanes_ind[node_num] =
        std::move(tree.hyperplanes_ind[tree_node]);
    search_tree.hyperplanes_data[node_num] =
        std::move(tree.hyperplanes_data[tree_node]);
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
void convert_tree(SparseRPTree<In, Idx> tree,
                  SparseSearchTree<In, Idx> &search_tree, std::size_t node_num,
                  std::size_t leaf_start, std::size_t tree_node) {
  // purposely ignore return value here
  recursive_convert(tree, search_tree, node_num, leaf_start, tree_node);
}

template <typename In, typename Idx>
SparseSearchTree<In, Idx> convert_tree_format(SparseRPTree<In, Idx> &&tree,
                                              std::size_t n_points,
                                              std::size_t ndim) {
  const auto n_nodes = tree.children.size();
  SparseSearchTree<In, Idx> search_tree(n_nodes, n_points, ndim,
                                        tree.leaf_size);

  std::size_t node_num = 0;
  std::size_t leaf_start = 0;
  convert_tree(std::move(tree), search_tree, node_num, leaf_start, n_nodes - 1);

  return search_tree;
}

template <typename In, typename Idx>
std::vector<SparseSearchTree<In, Idx>>
convert_rp_forest(std::vector<SparseRPTree<In, Idx>> &rp_forest,
                  std::size_t n_points, std::size_t ndim) {
  std::vector<SparseSearchTree<In, Idx>> search_forest;
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
search_leaf_range(const SparseSearchTree<In, Idx> &tree,
                  typename std::vector<std::size_t>::const_iterator ind_start,
                  std::size_t ind_size,
                  typename std::vector<In>::const_iterator data_start,
                  RandomIntGenerator<Idx> &rng) {
  Idx current_node = 0;

  while (true) {
    auto child_pair = tree.children[current_node];

    // it's a leaf: child_pair contains pointers into the indices
    if (tree.is_leaf(current_node)) {
      return child_pair;
    }

    // it's a node, find the child to go to
    auto side = select_side_sparse(
        ind_start, ind_size, data_start, tree.hyperplanes_ind[current_node],
        tree.hyperplanes_data[current_node], tree.offsets[current_node], rng);
    if (side == 0) {
      current_node = child_pair.first; // go left
    } else {
      current_node = child_pair.second; // go right
    }
  }
}

template <typename In, typename Idx>
std::vector<Idx>
search_indices(const SparseSearchTree<In, Idx> &tree,
               typename std::vector<std::size_t>::const_iterator ind_start,
               std::size_t ind_size,
               typename std::vector<In>::const_iterator data_start,
               RandomIntGenerator<Idx> &rng) {
  std::pair<std::size_t, std::size_t> range =
      search_leaf_range(tree, ind_start, ind_size, data_start, rng);
  std::vector<Idx> leaf_indices(tree.indices.begin() + range.first,
                                tree.indices.begin() + range.second);
  return leaf_indices;
}

template <typename In, typename Out, typename Idx>
void search_tree_heap_cache(const SparseSearchTree<In, Idx> &tree,
                            const SparseVectorDistance<In, Out, Idx> &distance,
                            Idx i, RandomIntGenerator<Idx> &rng,
                            NNHeap<Out, Idx> &current_graph,
                            std::unordered_set<Idx> &seen) {
  auto [ind_start, ind_size, data_start] = distance.get_y(i);

  std::vector<Idx> leaf_indices =
      search_indices(tree, ind_start, ind_size, data_start, rng);

  for (auto &idx : leaf_indices) {
    if (seen.find(idx) == seen.end()) { // not-contains
      const auto d = distance.calculate(idx, i);
      current_graph.checked_push(i, d, idx);
      seen.insert(idx);
    }
  }
}

template <typename In, typename Out, typename Idx>
void search_tree_heap(const SparseSearchTree<In, Idx> &tree,
                      const SparseVectorDistance<In, Out, Idx> &distance, Idx i,
                      RandomIntGenerator<Idx> &rng,
                      NNHeap<Out, Idx> &current_graph) {
  auto [ind_start, ind_size, data_start] = distance.get_y(i);

  std::vector<Idx> leaf_indices =
      search_indices(tree, ind_start, ind_size, data_start, rng);

  for (auto &idx : leaf_indices) {
    const auto d = distance.calculate(idx, i);
    current_graph.checked_push(i, d, idx);
  }
}

template <typename In, typename Out, typename Idx>
void search_forest_cache(const std::vector<SparseSearchTree<In, Idx>> &forest,
                         const SparseVectorDistance<In, Out, Idx> &distance,
                         Idx i, RandomIntGenerator<Idx> &rng,
                         NNHeap<Out, Idx> &current_graph) {
  std::unordered_set<Idx> seen;

  for (const auto &tree : forest) {
    search_tree_heap_cache(tree, distance, i, rng, current_graph, seen);
  }
}

template <typename In, typename Out, typename Idx>
void search_forest(const std::vector<SparseSearchTree<In, Idx>> &forest,
                   const SparseVectorDistance<In, Out, Idx> &distance, Idx i,
                   RandomIntGenerator<Idx> &rng,
                   NNHeap<Out, Idx> &current_graph) {
  for (const auto &tree : forest) {
    search_tree_heap(tree, distance, i, rng, current_graph);
  }
}

template <typename In, typename Out, typename Idx>
NNHeap<Out, Idx>
search_forest(const std::vector<SparseSearchTree<In, Idx>> &forest,
              const SparseVectorDistance<In, Out, Idx> &distance,
              uint32_t n_nbrs, ParallelRandomIntProvider<Idx> &rng_provider,
              bool cache, std::size_t n_threads, ProgressBase &progress,
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

#endif // TDOANN_RPTREESPARSE_H

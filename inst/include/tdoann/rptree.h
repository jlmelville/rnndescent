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

#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#include "distancebase.h"
#include "heap.h"
#include "parallel.h"
#include "random.h"

namespace tdoann {

template <typename Idx, typename In> struct RPTree {
  std::vector<std::vector<In>> hyperplanes;
  std::vector<In> offsets;
  std::vector<std::pair<Idx, Idx>> children;
  std::vector<std::vector<Idx>> point_indices;
  Idx max_leaf_size;

  RPTree(std::vector<std::vector<In>> hplanes, std::vector<In> offs,
         std::vector<std::pair<Idx, Idx>> chldrn,
         std::vector<std::vector<Idx>> p_indices, Idx max_size)
      : hyperplanes(std::move(hplanes)), offsets(std::move(offs)),
        children(std::move(chldrn)), point_indices(std::move(p_indices)),
        max_leaf_size(max_size) {}
};

template <typename Idx, typename In>
std::tuple<std::vector<Idx>, std::vector<Idx>, std::vector<In>, In>
euclidean_random_projection_split(const std::vector<In> &data, size_t ndim,
                                  const std::vector<Idx> &indices,
                                  tdoann::RandomIntGenerator<Idx> &rng) {
  constexpr In EPS = 1e-8;

  const std::size_t n_points = indices.size();

  // pick two random (distinct) points
  Idx left_index = rng.rand_int(n_points);
  Idx right_index = rng.rand_int(n_points - 1);
  if (left_index == right_index) {
    ++right_index;
  }

  Idx left = indices[left_index] * ndim;
  Idx right = indices[right_index] * ndim;

  std::vector<In> hyperplane_vector(ndim);
  In hyperplane_offset = 0.0;
  for (size_t d = 0; d < ndim; ++d) {
    hyperplane_vector[d] = data[left + d] - data[right + d];
    hyperplane_offset -=
        hyperplane_vector[d] * (data[left + d] + data[right + d]) / 2.0;
  }

  std::size_t n_left = 0;
  std::size_t n_right = 0;
  std::vector<Idx> indices_left;
  std::vector<Idx> indices_right;

  // Calculate margins and assign sides
  for (auto index : indices) {
    In margin =
        std::inner_product(hyperplane_vector.begin(), hyperplane_vector.end(),
                           data.begin() + index * ndim, hyperplane_offset);

    // if effectively on the hyperplane, pick a side at random
    if (std::abs(margin) < EPS) {
      if (rng.rand_int(2) == 0) {
        indices_left.push_back(index);
        ++n_left;
      } else {
        indices_right.push_back(index);
        ++n_right;
      }
    } else if (margin > 0) {
      // +ve on the left
      indices_left.push_back(index);
      ++n_left;
    } else {
      // -ve on the right
      indices_right.push_back(index);
      ++n_right;
    }
  }

  // If one side or the other is empty then assign to the sides randomly
  if (indices_left.empty() || indices_right.empty()) {
    indices_left.clear();
    indices_right.clear();
    for (auto index : indices) {
      if (rng.rand_int(2) == 0) {
        indices_left.push_back(index);
      } else {
        indices_right.push_back(index);
      }
    }
  }

  return std::make_tuple(indices_left, indices_right, hyperplane_vector,
                         hyperplane_offset);
}

template <typename Idx, typename In>
void make_euclidean_tree(const std::vector<In> &data, std::size_t ndim,
                         const std::vector<Idx> &indices,
                         std::vector<std::vector<In>> &hyperplanes,
                         std::vector<In> &offsets,
                         std::vector<std::pair<Idx, Idx>> &children,
                         std::vector<std::vector<Idx>> &point_indices,
                         tdoann::RandomIntGenerator<Idx> &rng,
                         unsigned int leaf_size = 30,
                         unsigned int max_depth = 100) {
  constexpr Idx idx_sentinel = static_cast<Idx>(-1);
  constexpr In minus_one = static_cast<In>(-1);
  constexpr In minus_inf = -std::numeric_limits<In>::infinity();

  if (indices.size() > leaf_size && max_depth > 0) {
    auto [left_indices, right_indices, hyperplane, offset] =
        euclidean_random_projection_split(data, ndim, indices, rng);

    make_euclidean_tree(data, ndim, left_indices, hyperplanes, offsets,
                        children, point_indices, rng, leaf_size, max_depth - 1);

    Idx left_node_num = point_indices.size() - 1;

    make_euclidean_tree(data, ndim, right_indices, hyperplanes, offsets,
                        children, point_indices, rng, leaf_size, max_depth - 1);

    Idx right_node_num = point_indices.size() - 1;

    hyperplanes.push_back(hyperplane);
    offsets.push_back(offset);
    children.emplace_back(left_node_num, right_node_num);
    point_indices.emplace_back(std::vector<Idx>(1, idx_sentinel));
  } else {
    hyperplanes.emplace_back(std::vector<In>(1, minus_one));
    offsets.push_back(minus_inf);
    children.emplace_back(idx_sentinel, idx_sentinel);
    point_indices.push_back(indices);
  }
}

template <typename Idx, typename In>
RPTree<Idx, In> make_dense_tree(const std::vector<In> &data, std::size_t ndim,
                                tdoann::RandomIntGenerator<Idx> &rng,
                                Idx leaf_size, bool angular) {

  std::vector<Idx> indices(data.size() / ndim);
  std::iota(indices.begin(), indices.end(), 0);

  std::vector<std::vector<In>> hyperplanes;
  std::vector<In> offsets;
  std::vector<std::pair<Idx, Idx>> children;
  std::vector<std::vector<Idx>> point_indices;

  if (angular) {
    make_euclidean_tree(data, ndim, indices, hyperplanes, offsets, children,
                        point_indices, rng, leaf_size);
  } else {
    make_euclidean_tree(data, ndim, indices, hyperplanes, offsets, children,
                        point_indices, rng, leaf_size);
  }

  Idx max_leaf_size = leaf_size;
  for (const auto &points : point_indices) {
    max_leaf_size = std::max(max_leaf_size, static_cast<Idx>(points.size()));
  }

  return RPTree<Idx, In>(std::move(hyperplanes), std::move(offsets),
                         std::move(children), std::move(point_indices),
                         max_leaf_size);
}

template <typename Idx, typename In>
std::vector<Idx> get_leaves_from_tree(const RPTree<Idx, In> &tree) {
  constexpr Idx sentinel = static_cast<Idx>(-1);

  Idx n_leaves = 0;
  for (const auto &child : tree.children) {
    if (child.first == sentinel && child.second == sentinel) {
      ++n_leaves;
    }
  }

  std::vector<Idx> leaf_indices;
  leaf_indices.reserve(n_leaves * tree.max_leaf_size);

  for (size_t i = 0; i < tree.point_indices.size(); ++i) {
    if (tree.children[i].first == sentinel ||
        tree.children[i].second == sentinel) {
      const auto &indices = tree.point_indices[i];
      leaf_indices.insert(leaf_indices.end(), indices.begin(), indices.end());
      const std::size_t padding = tree.max_leaf_size - indices.size();
      leaf_indices.insert(leaf_indices.end(), padding, sentinel);
    }
  }

  return leaf_indices;
}

template <typename Idx, typename Out>
void generate_leaf_updates(
    const BaseDistance<Out, Idx> &distance,
    const NNHeap<Out, Idx> &current_graph, const std::vector<Idx> &leaves,
    std::size_t max_leaf_size,
    std::vector<std::vector<std::tuple<Idx, Idx, Out>>> &updates,
    std::size_t begin, std::size_t end) {
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

      for (auto j = i + 1; j != leaf_end; ++j) {
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

template <typename Idx, typename Out>
void init_rp_tree(const BaseDistance<Out, Idx> &distance,
                  NNHeap<Out, Idx> &current_graph,
                  const std::vector<Idx> &leaves, std::size_t max_leaf_size,
                  std::size_t n_threads, ProgressBase &progress,
                  const Executor &executor) {

  const std::size_t n_leaves = leaves.size() / max_leaf_size;
  std::vector<std::vector<std::tuple<Idx, Idx, Out>>> updates(n_leaves);

  auto worker = [&distance, &current_graph, &leaves, &updates,
                 max_leaf_size](std::size_t begin, std::size_t end) {
    generate_leaf_updates(distance, current_graph, leaves, max_leaf_size,
                          updates, begin, end);
  };
  auto after_worker = [&current_graph, &updates](std::size_t, std::size_t) {
    for (const auto &block_updates : updates) {
      for (const auto &[p, q, d] : block_updates) {
        current_graph.checked_push_pair(p, d, q);
      }
    }
  };
  ExecutionParams exec_params{65536};
  dispatch_work(worker, after_worker, n_leaves, n_threads, exec_params,
                progress, executor);
}

template <typename Idx, typename Out>
auto init_rp_tree(const BaseDistance<Out, Idx> &distance,
                  const std::vector<Idx> &leaves, std::size_t max_leaf_size,
                  unsigned int n_nbrs, std::size_t n_threads,
                  ProgressBase &progress, const Executor &executor)
    -> NNHeap<Out, Idx> {
  progress.set_n_iters(1);

  NNHeap<Out, Idx> current_graph(distance.get_ny(), n_nbrs);

  init_rp_tree(distance, current_graph, leaves, max_leaf_size, n_threads,
               progress, executor);

  return current_graph;
}

} // namespace tdoann

#endif // TDOANN_RPTREE_H

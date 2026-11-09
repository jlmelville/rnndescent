//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#ifndef RNN_HUB_TREE_H
#define RNN_HUB_TREE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "tdoann/rptree.h"

namespace rnndescent {

// Use the three highest-in-degree points at each node as in PyNNDescent's
// balanced hub tree. Store the result in the existing explicit tree format.
template <typename In, typename Idx> class HubTreeBuilder {
  static constexpr In tie_margin = 1e-8;
  static constexpr Idx missing = std::numeric_limits<Idx>::max();

  struct Plane {
    std::vector<In> normal;
    In offset = 0;
  };

  const std::vector<In> &data;
  const std::vector<int> &degree;
  std::size_t ndim;
  std::size_t leaf_size;
  std::size_t max_depth;
  std::vector<Idx> order;
  tdoann::RPTree<In, Idx> tree;

  auto point(Idx row) const { return data.cbegin() + row * ndim; }

  auto hubs(std::size_t begin, std::size_t end) const -> std::array<Idx, 3> {
    std::array<Idx, 3> top{missing, missing, missing};
    for (auto i = begin; i < end; ++i) {
      auto candidate = order[i];
      for (auto &hub : top) {
        if (hub == missing || degree[candidate] > degree[hub] ||
            (degree[candidate] == degree[hub] && candidate < hub)) {
          std::swap(candidate, hub);
          if (candidate == missing) {
            break;
          }
        }
      }
    }
    return top;
  }

  auto bisector(Idx left, Idx right) const -> Plane {
    Plane plane{std::vector<In>(ndim)};
    const auto x = point(left);
    const auto y = point(right);
    In sum = 0;
    for (std::size_t j = 0; j < ndim; ++j) {
      const auto normal = x[j] - y[j];
      plane.normal[j] = normal;
      sum += normal * (x[j] + y[j]);
    }
    plane.offset = -sum / 2;
    return plane;
  }

  bool margins(const Plane &plane, std::size_t begin,
               std::vector<In> &result) const {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = std::inner_product(plane.normal.begin(), plane.normal.end(),
                                     point(order[begin + i]), plane.offset);
      if (!std::isfinite(result[i])) {
        return false;
      }
    }
    return true;
  }

  static bool tied(In margin) { return std::abs(margin) < tie_margin; }

  static bool left_at(In margin, std::size_t rank) {
    return tied(margin) ? rank % 2 == 0 : margin > 0;
  }

  auto split(std::size_t begin, std::size_t end)
      -> std::pair<Plane, std::size_t> {
    const auto size = end - begin;
    const auto top = hubs(begin, end);
    Plane best;
    std::vector<In> best_margins(size);
    std::vector<In> scratch(size);
    std::size_t best_smaller = 0;
    const auto n_hubs = std::min(size, top.size());
    for (std::size_t a = 0; a < n_hubs; ++a) {
      for (auto b = a + 1; b < n_hubs; ++b) {
        auto plane = bisector(top[a], top[b]);
        if (!margins(plane, begin, scratch)) {
          continue;
        }
        std::size_t n_left = 0;
        for (std::size_t i = 0; i < size; ++i) {
          n_left += left_at(scratch[i], i);
        }
        const auto smaller = std::min(n_left, size - n_left);
        if (best.normal.empty() || smaller > best_smaller) {
          best = std::move(plane);
          best_margins.swap(scratch);
          best_smaller = smaller;
        }
      }
    }

    // Use a balanced tie split if float overflow makes all margins unusable.
    const auto tie_split = [&] {
      return std::make_pair(Plane{std::vector<In>(ndim)}, begin + size / 2);
    };
    if (best.normal.empty()) {
      return tie_split();
    }

    const bool use_median = best_smaller * 10 < size;
    std::size_t left_count = 0;
    std::size_t right_count = 0;
    if (use_median) {
      std::copy(best_margins.begin(), best_margins.end(), scratch.begin());
      const auto middle = scratch.begin() + size / 2;
      std::nth_element(scratch.begin(), middle, scratch.end());
      auto shift = *middle;
      if (size % 2 == 0) {
        const auto lower = *std::max_element(scratch.begin(), middle);
        shift = lower + (shift - lower) / 2;
      }
      best.offset -= shift;
      if (!margins(best, begin, best_margins)) {
        return tie_split();
      }
      for (auto margin : best_margins) {
        if (!tied(margin)) {
          left_count += margin > 0;
          right_count += margin < 0;
        }
      }
    }

    std::vector<Idx> left;
    std::vector<Idx> right;
    left.reserve(size / 2);
    right.reserve(size / 2);
    for (std::size_t i = 0; i < size; ++i) {
      const auto margin = best_margins[i];
      bool go_left = left_at(margin, i);
      if (use_median && tied(margin)) {
        go_left = left_count <= right_count;
        left_count += go_left;
        right_count += !go_left;
      }
      (go_left ? left : right).push_back(order[begin + i]);
    }
    // Recomputing margins after shifting can lose precision near the median.
    if (std::min(left.size(), right.size()) * 10 < size) {
      return tie_split();
    }
    std::copy(left.begin(), left.end(), order.begin() + begin);
    std::copy(right.begin(), right.end(), order.begin() + begin + left.size());
    return {std::move(best), begin + left.size()};
  }

  auto build(std::size_t begin, std::size_t end, std::size_t depth)
      -> std::size_t {
    if (end - begin <= leaf_size || depth == max_depth) {
      tree.add_leaf(
          std::vector<Idx>(order.begin() + begin, order.begin() + end));
    } else {
      auto [plane, middle] = split(begin, end);
      const auto left = build(begin, middle, depth + 1);
      const auto right = build(middle, end, depth + 1);
      tree.add_node(plane.normal, plane.offset, left, right);
    }
    return tree.children.size() - 1;
  }

public:
  HubTreeBuilder(const std::vector<In> &data, const std::vector<int> &degree,
                 std::size_t ndim, std::size_t leaf_size, std::size_t max_depth)
      : data(data), degree(degree), ndim(ndim), leaf_size(leaf_size),
        max_depth(max_depth), order(degree.size()),
        tree(order.size(), leaf_size, ndim) {}

  auto make_tree() -> tdoann::SearchTree<In, Idx> {
    std::iota(order.begin(), order.end(), Idx{0});
    build(0, order.size(), 0);
    return tdoann::convert_tree_format(std::move(tree), order.size(), ndim);
  }
};

} // namespace rnndescent

#endif // RNN_HUB_TREE_H

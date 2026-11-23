//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#ifndef RNN_RPTREE_ANGULAR_H
#define RNN_RPTREE_ANGULAR_H

#include <cstdint>
#include <limits>

#include "rnn_distance_batch.h"
#include "rnn_rptree_shared.h"

namespace rnndescent::shared_forest {

inline std::size_t angular_local_rows(std::size_t dimension,
                                      std::size_t leaf_size) {
  constexpr std::size_t cache_bytes = 1024 * 1024;
  return std::max(leaf_size, cache_bytes / sizeof(float) / dimension);
}

constexpr uint32_t no_plane = std::numeric_limits<uint32_t>::max();

// A row owns its plane IDs during projection. Replace each active ID with its
// split decision in place so partitioning needs no second row-by-tree buffer.
template <typename Idx>
void project_angular_row(const float *row, std::size_t dimension,
                         const std::vector<Plane<float, Idx>> &planes,
                         uint32_t *slots, std::size_t n_trees) {
  std::array<const float *, 4> normals;
  std::array<uint32_t *, 4> sides;
  std::size_t used = 0;
  for (std::size_t tree = 0; tree < n_trees; ++tree) {
    if (slots[tree] == no_plane) {
      continue;
    }
    normals[used] = planes[slots[tree]].normal.data();
    sides[used++] = &slots[tree];
    if (used == normals.size()) {
      const auto margins = DenseDistanceFour<float, Idx>::template blocked_four<
          DenseBatchMetric::raw_dot>(row, normals, dimension);
      for (std::size_t j = 0; j < used; ++j) {
        *sides[j] = margin_side(margins[j]);
      }
      used = 0;
    }
  }
  for (std::size_t j = 0; j < used; ++j) {
    *sides[j] = margin_side(tdoann::sum_feature_products(
        normals[j], normals[j] + dimension, row, 0.0f, std::multiplies<>{}));
  }
}

template <typename Idx>
auto build_angular(const std::vector<float> &data, std::size_t ndim,
                   std::size_t n_trees, uint32_t leaf_size, uint32_t max_depth,
                   tdoann::ParallelRandomIntProvider<Idx> &provider,
                   std::size_t threads, RPProgress &progress,
                   const tdoann::Executor &executor) {
  using Members = std::vector<Idx>;
  using Tree = tdoann::RPLeafTree<Idx>;
  struct Node {
    std::size_t tree;
    uint32_t depth = 0;
    Members rows;
  };
  const auto n = data.size() / ndim;
  const auto local_rows = angular_local_rows(ndim, leaf_size);
  provider.initialize();
  progress.set_n_iters(1);
  progress.set_n_batches(2);

  std::vector<Node> frontier(n_trees);
  for (std::size_t t = 0; t < n_trees; ++t) {
    frontier[t].tree = t;
    frontier[t].rows.resize(n);
    std::iota(frontier[t].rows.begin(), frontier[t].rows.end(), Idx{0});
  }

  uint64_t first_stream = 1;
  {
    std::vector<uint32_t> slots(n * n_trees);
    for (uint32_t level = 0; level < max_depth; ++level) {
      std::vector<std::size_t> active;
      for (std::size_t i = 0; i < frontier.size(); ++i) {
        if (frontier[i].rows.size() > local_rows) {
          active.push_back(i);
        }
      }
      if (active.empty()) {
        break;
      }
      std::fill(slots.begin(), slots.end(), no_plane);
      std::vector<Plane<float, Idx>> planes(active.size());
      run_jobs(
          active.size(), threads,
          [&](std::size_t i) {
            const auto &node = frontier[active[i]];
            auto &plane = planes[i];
            plane.rng = provider.get_parallel_instance(first_stream + i);
            plane.normal =
                tdoann::angular_split_plane(data, ndim, node.rows, *plane.rng);
            for (const auto p : node.rows) {
              slots[p * n_trees + node.tree] = static_cast<uint32_t>(i);
            }
          },
          progress, executor);
      first_stream += active.size();

      constexpr std::size_t tile = 64;
      run_jobs((n + tile - 1) / tile, threads,
               [&](std::size_t block) {
                 for (auto p = block * tile;
                      p < std::min(n, (block + 1) * tile); ++p) {
                   project_angular_row(data.data() + p * ndim, ndim, planes,
                                       slots.data() + p * n_trees, n_trees);
                 }
               },
               progress, executor);

      std::vector<std::pair<Members, Members>> children(active.size());
      run_jobs(
          active.size(), threads,
          [&](std::size_t i) {
            const auto &node = frontier[active[i]];
            auto &[left, right] = children[i];
            partition_projected_rows(
                node.rows,
                [&](Idx p) { return slots[p * n_trees + node.tree]; }, left,
                right, *planes[i].rng);
          },
          progress, executor);

      // Replacing a split node by its left and right children preserves the
      // depth-first leaf order, including small nodes deferred at earlier
      // levels.
      std::vector<Node> next;
      next.reserve(frontier.size() + active.size());
      std::size_t split = 0;
      for (std::size_t i = 0; i < frontier.size(); ++i) {
        auto &node = frontier[i];
        if (split == active.size() || i != active[split]) {
          next.push_back(std::move(node));
          continue;
        }
        auto &[left, right] = children[split++];
        next.push_back({node.tree, level + 1, std::move(left)});
        next.push_back({node.tree, level + 1, std::move(right)});
      }
      frontier = std::move(next);
    }
  }
  progress.batch_finished();

  std::vector<Tree> subtrees(frontier.size());
  run_jobs(
      frontier.size(), threads,
      [&](std::size_t i) {
        const auto &node = frontier[i];
        auto rng = provider.get_parallel_instance(first_stream + i);
        tdoann::make_tree_recursive(
            data, ndim, node.rows, subtrees[i], *rng,
            tdoann::angular_random_projection_split<float, Idx>, leaf_size,
            max_depth - node.depth);
      },
      progress, executor);
  progress.batch_finished();
  std::vector<Tree> forest(n_trees);
  for (std::size_t i = 0; i < frontier.size(); ++i) {
    auto &tree = forest[frontier[i].tree];
    auto &subtree = subtrees[i];
    tree.leaf_size = std::max(tree.leaf_size, subtree.leaf_size);
    tree.indices.insert(tree.indices.end(),
                        std::make_move_iterator(subtree.indices.begin()),
                        std::make_move_iterator(subtree.indices.end()));
  }
  return forest;
}

} // namespace rnndescent::shared_forest

#endif

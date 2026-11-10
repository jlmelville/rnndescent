#ifndef RNN_RPTREE_SHARED_H
#define RNN_RPTREE_SHARED_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "rnn_progress.h"
#include "tdoann/rptree.h"

namespace rnndescent {
namespace shared_forest {

template <typename Job>
void run_jobs(std::size_t n_jobs, std::size_t n_threads, Job job,
              RPProgress &progress, const tdoann::Executor &executor) {
  std::atomic<std::size_t> next{0};
  executor.parallel_for(
      0, std::min(n_jobs, n_threads),
      [&](std::size_t, std::size_t) {
        for (auto i = next.fetch_add(1, std::memory_order_relaxed); i < n_jobs;
             i = next.fetch_add(1, std::memory_order_relaxed)) {
          job(i);
        }
      },
      n_threads, 1);
  progress.check_interrupt();
  progress.batch_finished();
}

template <typename In, typename Idx> struct Plane {
  std::vector<In> normal;
  In offset{};
  std::unique_ptr<tdoann::RandomIntGenerator<Idx>> rng;
};

constexpr uint8_t tied_side = 2;

template <typename In> uint8_t margin_side(In margin) {
  constexpr In epsilon = 1e-8;
  return std::abs(margin) < epsilon ? tied_side : (margin > 0 ? 0 : 1);
}

template <typename In, typename Idx>
void project_four(const In *row, std::size_t ndim,
                  const std::array<const Plane<In, Idx> *, 4> &planes,
                  const std::array<uint8_t *, 4> &output) {
  const auto *a = planes[0]->normal.data();
  const auto *b = planes[1]->normal.data();
  const auto *c = planes[2]->normal.data();
  const auto *d = planes[3]->normal.data();
  In ma = planes[0]->offset;
  In mb = planes[1]->offset;
  In mc = planes[2]->offset;
  In md = planes[3]->offset;
  for (std::size_t j = 0; j < ndim; ++j) {
    const In value = row[j];
    ma += a[j] * value;
    mb += b[j] * value;
    mc += c[j] * value;
    md += d[j] * value;
  }
  *output[0] = margin_side(ma);
  *output[1] = margin_side(mb);
  *output[2] = margin_side(mc);
  *output[3] = margin_side(md);
}

template <typename In, typename Idx>
auto split_level(const std::vector<In> &data, std::size_t ndim,
                 std::size_t n_trees, std::size_t width, uint32_t leaf_size,
                 std::vector<std::optional<std::vector<Idx>>> &frontier,
                 tdoann::ParallelRandomIntProvider<Idx> &provider,
                 std::size_t n_threads, RPProgress &progress,
                 const tdoann::Executor &executor) {
  using Members = std::vector<Idx>;
  const auto n_points = data.size() / ndim;
  const bool roots = width == 1;
  Members root_members;
  if (roots) {
    root_members.resize(n_points);
    std::iota(root_members.begin(), root_members.end(), Idx{0});
  }

  auto members_at = [&](std::size_t i) -> const Members * {
    return roots ? &root_members : (frontier[i] ? &*frontier[i] : nullptr);
  };
  std::vector<Plane<In, Idx>> planes(frontier.size());
  Plane<In, Idx> inactive;
  inactive.normal.resize(ndim);
  std::vector<std::vector<uint8_t>> node(n_trees);
  std::vector<std::vector<uint8_t>> decisions(n_trees);
  for (std::size_t tree = 0; tree < n_trees; ++tree) {
    node[tree].resize(n_points);
    decisions[tree].resize(n_points);
    for (std::size_t position = 0; position < width; ++position) {
      const auto i = tree * width + position;
      const auto *members = members_at(i);
      if (!members) {
        continue;
      }
      if (position != 0) {
        for (const auto point : *members) {
          node[tree][point] = static_cast<uint8_t>(position);
        }
      }
      if (members->size() > leaf_size) {
        auto &plane = planes[i];
        plane.rng =
            provider.get_parallel_instance(n_trees * (width - 1) + i + 1);
        auto [normal, offset] =
            tdoann::euclidean_split_plane(data, ndim, *members, *plane.rng);
        plane.normal = std::move(normal);
        plane.offset = offset;
      }
    }
  }

  constexpr std::size_t tile = 64;
  run_jobs((n_points + tile - 1) / tile, n_threads,
           [&](std::size_t block) {
             const auto end = std::min(n_points, (block + 1) * tile);
             for (std::size_t tree = 0; tree < n_trees; tree += 4) {
               const auto count = std::min(n_trees - tree, std::size_t{4});
               for (auto point = block * tile; point < end; ++point) {
                 std::array<const Plane<In, Idx> *, 4> selected;
                 selected.fill(&inactive);
                 std::array<uint8_t, 4> unused;
                 std::array<uint8_t *, 4> output;
                 for (std::size_t lane = 0; lane < 4; ++lane) {
                   output[lane] = &unused[lane];
                 }
                 for (std::size_t lane = 0; lane < count; ++lane) {
                   const auto t = tree + lane;
                   const auto &plane = planes[t * width + node[t][point]];
                   if (plane.rng) {
                     selected[lane] = &plane;
                     output[lane] = &decisions[t][point];
                   }
                 }
                 project_four(data.data() + point * ndim, ndim, selected,
                              output);
               }
             }
           },
           progress, executor);

  std::vector<std::optional<Members>> children(frontier.size() * 2);
  run_jobs(
      frontier.size(), n_threads,
      [&](std::size_t i) {
        const auto *members = members_at(i);
        if (!members) {
          return;
        }
        auto &plane = planes[i];
        if (!plane.rng) {
          children[2 * i] = std::move(frontier[i]);
          return;
        }

        // Resolve ties in member order to match the scalar split's RNG draws.
        std::vector<uint8_t> sides;
        sides.reserve(members->size());
        std::size_t n_left = 0;
        for (const auto point : *members) {
          const auto decision = decisions[i / width][point];
          const auto side =
              decision == tied_side ? plane.rng->rand_int(2) : decision;
          sides.push_back(side);
          n_left += side == 0;
        }
        tdoann::partition_indices(*members, sides, n_left,
                                  members->size() - n_left,
                                  children[2 * i].emplace(),
                                  children[2 * i + 1].emplace(), *plane.rng);
      },
      progress, executor);
  return children;
}

template <typename In, typename Idx>
auto build(const std::vector<In> &data, std::size_t ndim, std::size_t n_trees,
           uint32_t leaf_size, uint32_t max_depth,
           tdoann::ParallelRandomIntProvider<Idx> &provider,
           std::size_t n_threads, RPProgress &progress,
           const tdoann::Executor &executor) {
  using Tree = tdoann::RPLeafTree<Idx>;
  using Members = std::vector<Idx>;
  constexpr uint32_t shared_levels = 8;
  const auto depth = std::min(shared_levels, max_depth);
  provider.initialize();
  progress.set_n_iters(1);
  progress.set_n_batches(2 * depth + 1);

  // The first level uses all points for each root. Later levels keep slots
  // in tree and path order. An empty leaf differs from an unused slot.
  std::vector<std::optional<Members>> frontier(n_trees);
  std::size_t width = 1;
  uint32_t level = 0;
  for (; level < depth; ++level) {
    frontier = split_level(data, ndim, n_trees, width, leaf_size, frontier,
                           provider, n_threads, progress, executor);
    width *= 2;
    if (std::none_of(frontier.begin(), frontier.end(), [&](const auto &node) {
          return node && node->size() > leaf_size;
        })) {
      ++level;
      break;
    }
  }

  std::vector<Tree> subtrees(frontier.size());
  progress.set_n_batches(1);
  run_jobs(
      frontier.size(), n_threads,
      [&](std::size_t i) {
        if (!frontier[i]) {
          return;
        }
        const auto &members = *frontier[i];
        auto rng =
            provider.get_parallel_instance(n_trees * (width - 1) + i + 1);
        Tree subtree(members.size(), leaf_size, ndim);
        tdoann::make_tree_recursive(
            data, ndim, members, subtree, *rng,
            tdoann::euclidean_random_projection_split<In, Idx>, leaf_size,
            max_depth - level);
        subtrees[i] = std::move(subtree);
      },
      progress, executor);

  std::vector<Tree> forest(n_trees);
  for (std::size_t i = 0; i < n_trees; ++i) {
    auto &tree = forest[i];
    std::size_t n_leaves = 0;
    for (std::size_t j = 0; j < width; ++j) {
      const auto &subtree = subtrees[i * width + j];
      n_leaves += subtree.indices.size();
      tree.leaf_size = std::max(tree.leaf_size, subtree.leaf_size);
    }
    tree.indices.reserve(n_leaves);
    for (std::size_t j = 0; j < width; ++j) {
      auto &indices = subtrees[i * width + j].indices;
      tree.indices.insert(tree.indices.end(),
                          std::make_move_iterator(indices.begin()),
                          std::make_move_iterator(indices.end()));
    }
  }
  return forest;
}

} // namespace shared_forest
} // namespace rnndescent

#endif

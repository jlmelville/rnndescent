//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#ifndef RNN_NND_ROUTE_H
#define RNN_NND_ROUTE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <vector>

#include "tdoann/nndparallel.h"

namespace rnndescent {

// The matrix importer visits sorted source rows and proposes each edge in both
// directions. Replay that order for each destination so cutoff ties and flags
// match serial import. Limit the number of scans as in parallel leaf seeding.
template <typename Out, typename Idx>
auto import_leaf_heap(tdoann::NNHeap<Out, Idx> &leaves, std::size_t n_threads,
                      tdoann::ProgressBase &progress,
                      const tdoann::Executor &executor)
    -> tdoann::NNDHeap<Out, Idx> {
  tdoann::sort_heap(leaves, n_threads, progress, executor);
  tdoann::NNDHeap<Out, Idx> graph(leaves.n_points, leaves.n_nbrs);
  auto insert = [&](std::size_t first, std::size_t last) {
    for (std::size_t row = 0; row < leaves.n_points; ++row) {
      const auto begin = row * leaves.n_nbrs;
      const auto end = begin + leaves.n_nbrs;
      if (row >= first && row < last) {
        for (auto slot = begin; slot < end; ++slot) {
          const auto neighbor = leaves.idx[slot];
          if (neighbor == leaves.npos()) {
            continue;
          }
          graph.checked_push(row, leaves.dist[slot], neighbor);
          if (neighbor >= first && neighbor < last && neighbor != row) {
            graph.checked_push(neighbor, leaves.dist[slot], row);
          }
        }
      } else {
        for (auto slot = begin; slot < end; ++slot) {
          const auto neighbor = leaves.idx[slot];
          if (neighbor >= first && neighbor < last) {
            graph.checked_push(neighbor, leaves.dist[slot], row);
          }
        }
      }
    }
  };
  const auto owners = std::min(n_threads, std::size_t{8});
  tdoann::dispatch_work(insert, leaves.n_points, owners, progress, executor);
  return graph;
}

template <typename Out, typename Idx> class RowUpdateQueues {
  using Update = typename tdoann::ParallelLocalJoin<Out, Idx>::Update;

  // Keep the number of source-by-destination buffers bounded.
  static constexpr std::size_t max_partitions = 32;
  std::size_t partitions = 0;
  std::vector<std::vector<Update>> routed;

public:
  template <typename Generate>
  void collect(const tdoann::NNDHeap<Out, Idx> &graph, std::size_t begin,
               std::size_t end, std::size_t n_threads,
               const tdoann::Executor &executor, Generate generate) {
    const auto size = end - begin;
    partitions = std::min({size, n_threads, max_partitions});
    routed.resize(partitions * partitions);
    executor.parallel_for(
        0, partitions,
        [&](std::size_t first, std::size_t last) {
          for (auto source = first; source < last; ++source) {
            const auto from = begin + size * source / partitions;
            const auto to = begin + size * (source + 1) / partitions;
            auto *buffers = routed.data() + source * partitions;
            for (std::size_t owner = 0; owner < partitions; ++owner) {
              buffers[owner].clear();
            }
            generate(from, to, [&](Idx p, Idx q, Out distance) {
              // A destination's cutoff can only decrease during application.
              if (graph.accepts(p, distance)) {
                buffers[p % partitions].emplace_back(p, q, distance);
              }
              if (q != p && graph.accepts(q, distance)) {
                buffers[q % partitions].emplace_back(q, p, distance);
              }
            });
          }
        },
        partitions, 1);
  }

  auto apply(tdoann::NNDHeap<Out, Idx> &graph,
             const tdoann::Executor &executor) const -> std::size_t {
    std::array<std::size_t, max_partitions> accepted{};
    executor.parallel_for(
        0, partitions,
        [&](std::size_t first, std::size_t last) {
          for (auto owner = first; owner < last; ++owner) {
            std::size_t count = 0;
            // Visiting source ranges in order preserves each heap's insertions.
            for (std::size_t source = 0; source < partitions; ++source) {
              const auto &updates = routed[source * partitions + owner];
              for (const auto &[row, neighbor, distance] : updates) {
                count += graph.checked_push(row, distance, neighbor);
              }
            }
            accepted[owner] = count;
          }
        },
        partitions, 1);
    return std::accumulate(accepted.begin(), accepted.end(), std::size_t{0});
  }
};

template <typename Out, typename Idx>
class RoutedLowMemParallelLocalJoin
    : public tdoann::LowMemParallelLocalJoin<Out, Idx> {
  RowUpdateQueues<Out, Idx> queues;

public:
  using tdoann::LowMemParallelLocalJoin<Out, Idx>::LowMemParallelLocalJoin;

  auto apply_routed(tdoann::NNDHeap<Out, Idx> &graph, std::size_t begin,
                    std::size_t end, std::size_t n_threads,
                    const tdoann::Executor &executor) -> std::size_t {
    queues.collect(graph, begin, end, n_threads, executor,
                   [&](std::size_t from, std::size_t to, auto enqueue) {
                     for (auto row = from; row < to; ++row) {
                       auto &edges = this->edge_updates[row];
                       for (const auto &[p, q, distance] : edges) {
                         enqueue(p, q, distance);
                       }
                       edges.clear();
                     }
                   });
    return queues.apply(graph, executor);
  }
};

template <typename Out, typename Idx> class BinaryDirectJoin {
  const tdoann::BaseDistance<Out, Idx> &distance;
  RowUpdateQueues<Out, Idx> queues;

public:
  explicit BinaryDirectJoin(const tdoann::BaseDistance<Out, Idx> &distance)
      : distance(distance) {}

  auto execute(tdoann::NNDHeap<Out, Idx> &graph,
               const tdoann::NNHeap<Out, Idx> &fresh,
               const tdoann::NNHeap<Out, Idx> &old,
               tdoann::NNDProgressBase &progress, std::size_t n_threads,
               const tdoann::Executor &executor) -> std::size_t {
    // Score into the owners' queues while the graph is read-only. Applying
    // each window in source order preserves serial tie and duplicate handling.
    constexpr std::size_t window = 2048;
    auto score = [&](std::size_t begin, std::size_t end) {
      queues.collect(graph, begin, end, n_threads, executor,
                     [&](std::size_t from, std::size_t to, auto enqueue) {
                       tdoann::ParallelLocalJoin<Out, Idx>::visit_candidates(
                           fresh, old, fresh.n_nbrs, from, to,
                           [&](std::size_t, Idx p, auto visit) {
                             visit([&](Idx q) {
                               enqueue(p, q, distance.calculate(p, q));
                             });
                           });
                     });
    };
    std::size_t changes = 0;
    auto apply = [&](std::size_t, std::size_t) {
      changes += queues.apply(graph, executor);
    };
    tdoann::dispatch_work(score, apply, graph.n_points,
                          tdoann::ExecutionParams{window},
                          progress.get_base_progress());
    return changes;
  }
};

} // namespace rnndescent

#endif // RNN_NND_ROUTE_H

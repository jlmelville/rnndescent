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

template <typename Out, typename Idx>
class RoutedLowMemParallelLocalJoin
    : public tdoann::LowMemParallelLocalJoin<Out, Idx> {
  using Join = tdoann::LowMemParallelLocalJoin<Out, Idx>;
  using Update = typename Join::Update;

  // Keep the number of source-by-destination buffers bounded.
  static constexpr std::size_t max_partitions = 32;
  std::vector<std::vector<Update>> routed;

public:
  using Join::Join;

  auto apply_routed(tdoann::NNDHeap<Out, Idx> &graph, std::size_t begin,
                    std::size_t end, std::size_t n_threads,
                    const tdoann::Executor &executor) -> std::size_t {
    const auto size = end - begin;
    const auto partitions = std::min({size, n_threads, max_partitions});
    routed.resize(partitions * partitions);
    executor.parallel_for(
        0, partitions,
        [&](std::size_t first, std::size_t last) {
          for (auto source = first; source < last; ++source) {
            const auto from = begin + size * source / partitions;
            const auto to = begin + size * (source + 1) / partitions;
            auto *buffers = routed.data() + source * partitions;
            for (auto row = from; row < to; ++row) {
              auto &edges = this->edge_updates[row];
              for (const auto &[p, q, distance] : edges) {
                // A destination's cutoff can only decrease during application.
                if (graph.accepts(p, distance)) {
                  buffers[p % partitions].emplace_back(p, q, distance);
                }
                if (q != p && graph.accepts(q, distance)) {
                  buffers[q % partitions].emplace_back(q, p, distance);
                }
              }
              edges.clear();
            }
          }
        },
        partitions, 1);

    std::array<std::size_t, max_partitions> accepted{};
    executor.parallel_for(
        0, partitions,
        [&](std::size_t first, std::size_t last) {
          for (auto owner = first; owner < last; ++owner) {
            std::size_t count = 0;
            // Visiting source ranges in order preserves each heap's insertions.
            for (std::size_t source = 0; source < partitions; ++source) {
              auto &updates = routed[source * partitions + owner];
              for (const auto &[row, neighbor, distance] : updates) {
                count += graph.checked_push(row, distance, neighbor);
              }
              updates.clear();
            }
            accepted[owner] = count;
          }
        },
        partitions, 1);
    return std::accumulate(accepted.begin(), accepted.end(), std::size_t{0});
  }
};

} // namespace rnndescent

#endif // RNN_NND_ROUTE_H

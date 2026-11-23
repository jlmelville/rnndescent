//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#ifndef RNN_CANDIDATE_ROUTING_H
#define RNN_CANDIDATE_ROUTING_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "pforr.h"
#include "tdoann/nndcommon.h"
#include "tdoann/parallel.h"
#include "tdoann/random.h"

namespace rnndescent {

template <typename Out, typename Idx> class CompactCandidates {
  using Graph = tdoann::NNDHeap<Out, Idx>;
  using Heap = tdoann::NNHeap<Out, Idx>;

  // Keep capacities across descent rounds. Only one priority vector is used.
  std::vector<Out> priorities;
  std::vector<double> weighted_priorities;
  std::vector<std::vector<std::uint32_t>> remote_slots;

public:
  static bool supported(std::size_t n_threads, std::size_t slots) {
    // Bound the source-by-owner queue table and the width of a saved offset.
    return n_threads > 1 && n_threads <= 32 &&
           slots <= std::numeric_limits<std::uint32_t>::max();
  }

  void build(const Graph &graph, Heap &fresh, Heap &old, bool weight_by_degree,
             tdoann::ParallelRandomProvider &random, std::size_t n_threads,
             const tdoann::Executor &executor) {
    if (weight_by_degree) {
      const auto degrees = tdoann::count_reverse_neighbors(graph);
      // Multiply the original double draw before rounding to the heap type.
      populate(graph, fresh, old, random, n_threads, executor,
               weighted_priorities,
               [&](double draw, Idx other) { return draw * degrees[other]; });
    } else {
      populate(graph, fresh, old, random, n_threads, executor, priorities,
               [](Out draw, Idx) { return draw; });
    }
  }

private:
  template <typename Draw, typename Weight>
  void populate(const Graph &graph, Heap &fresh, Heap &old,
                tdoann::ParallelRandomProvider &random, std::size_t n_threads,
                const tdoann::Executor &executor, std::vector<Draw> &draws,
                Weight weight) {
    random.initialize();
    const auto ranges =
        pforr::split_input_range({0, graph.n_points}, n_threads, 1);
    if (ranges.empty()) {
      return;
    }
    const auto count = ranges.size();
    const auto width = ranges.front().second;
    const std::size_t k = graph.n_nbrs;
    draws.resize(graph.idx.size());
    remote_slots.resize(count * count);

    // Use the same source ranges, stream keys and valid-edge draws as locking.
    auto route = [&](std::size_t first, std::size_t last) {
      for (auto source = first; source < last; ++source) {
        const auto [begin, end] = ranges[source];
        auto rng = random.get_parallel_instance(end);
        auto *queues = remote_slots.data() + source * count;
        for (auto slot = begin * k; slot < end * k; ++slot) {
          const auto neighbor = graph.idx[slot];
          if (neighbor == graph.npos()) {
            continue;
          }
          draws[slot] = rng->unif();
          if (neighbor < begin || neighbor >= end) {
            queues[neighbor / width].push_back(
                static_cast<std::uint32_t>(slot));
          }
        }
      }
    };
    executor.parallel_for(0, count, route, count, 1);

    auto insert = [&](std::size_t first, std::size_t last) {
      auto push = [&](std::size_t slot, Idx row, Idx neighbor) {
        auto &heap = graph.flags[slot] == 1 ? fresh : old;
        heap.checked_push(row, weight(draws[slot], neighbor), neighbor);
      };
      for (auto owner = first; owner < last; ++owner) {
        const auto [begin, end] = ranges[owner];
        // Merge local and remote proposals in source order for each heap row.
        for (std::size_t source = 0; source < count; ++source) {
          if (source != owner) {
            auto &queue = remote_slots[source * count + owner];
            for (const auto slot : queue) {
              push(slot, graph.idx[slot], static_cast<Idx>(slot / k));
            }
            queue.clear();
            continue;
          }
          for (auto row = begin; row < end; ++row) {
            for (auto slot = row * k; slot < (row + 1) * k; ++slot) {
              const auto neighbor = graph.idx[slot];
              if (neighbor == graph.npos()) {
                continue;
              }
              push(slot, static_cast<Idx>(row), neighbor);
              if (neighbor >= begin && neighbor < end && neighbor != row) {
                push(slot, neighbor, static_cast<Idx>(row));
              }
            }
          }
        }
      }
    };
    executor.parallel_for(0, count, insert, count, 1);
  }
};

} // namespace rnndescent

#endif // RNN_CANDIDATE_ROUTING_H

//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#ifndef RNN_NND_BATCH_H
#define RNN_NND_BATCH_H

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "rnn_distance_batch.h"
#include "rnn_nnd_route.h"
#include "tdoann/nndescent.h"
#include "tdoann/nndparallel.h"

namespace rnndescent {

template <typename Out, typename Idx, typename Visit, typename Skip,
          typename Apply>
void score_candidates(const DenseDistanceFour<Out, Idx> &kernel,
                      const tdoann::BaseDistance<Out, Idx> &distance, Idx p,
                      Visit visit, Skip skip, Apply apply) {
  typename DenseDistanceFour<Out, Idx>::Targets targets;
  std::size_t size = 0;
  visit([&](Idx q) {
    if (skip(q)) {
      return;
    }
    targets[size++] = q;
    if (size == targets.size()) {
      const auto scores = kernel.evaluate(p, targets);
      for (std::size_t j = 0; j < size; ++j) {
        apply(targets[j], scores[j]);
      }
      size = 0;
    }
  });
  for (std::size_t j = 0; j < size; ++j) {
    apply(targets[j], distance.calculate(p, targets[j]));
  }
}

template <typename Out, typename Idx>
class DenseLowMemSerialLocalJoin
    : public tdoann::LowMemSerialLocalJoin<Out, Idx> {
  using Join = tdoann::LowMemSerialLocalJoin<Out, Idx>;
  DenseDistanceFour<Out, Idx> kernel;

public:
  DenseLowMemSerialLocalJoin(DenseDistanceFour<Out, Idx> kernel,
                             const tdoann::BaseDistance<Out, Idx> &distance)
      : Join(distance), kernel(kernel) {}

  auto execute(tdoann::NNDHeap<Out, Idx> &graph,
               const tdoann::NNHeap<Out, Idx> &new_nbrs,
               const tdoann::NNHeap<Out, Idx> &old_nbrs,
               tdoann::NNDProgressBase &progress) -> unsigned long override {
    return this->execute_anchors(
        new_nbrs, old_nbrs, progress, [&](Idx p, auto visit) {
          unsigned long updates = 0;
          score_candidates(
              kernel, this->distance, p, visit, [](Idx) { return false; },
              [&](Idx q, Out score) {
                updates += graph.checked_push_pair(p, score, q);
              });
          return updates;
        });
  }
};

template <typename Out, typename Idx, bool Cached>
class DenseParallelLocalJoin
    : public std::conditional_t<Cached,
                                tdoann::CacheParallelLocalJoin<Out, Idx>,
                                RoutedLowMemParallelLocalJoin<Out, Idx>> {
  using Join =
      std::conditional_t<Cached, tdoann::CacheParallelLocalJoin<Out, Idx>,
                         RoutedLowMemParallelLocalJoin<Out, Idx>>;
  using Heap = tdoann::NNDHeap<Out, Idx>;
  using Candidates = tdoann::NNHeap<Out, Idx>;
  DenseDistanceFour<Out, Idx> kernel;

public:
  template <typename... Args>
  DenseParallelLocalJoin(DenseDistanceFour<Out, Idx> kernel, Args &&...args)
      : Join(std::forward<Args>(args)...), kernel(kernel) {}

  auto execute(Heap &graph, const Candidates &new_nbrs,
               const Candidates &old_nbrs, tdoann::NNDProgressBase &progress,
               std::size_t n_threads, const tdoann::Executor &executor)
      -> std::size_t {
    auto generate = [&](std::size_t begin, std::size_t end) {
      if (kernel.evaluate) {
        generate_range(graph, new_nbrs, old_nbrs, begin, end);
      } else {
        Join::execute(graph, new_nbrs, old_nbrs, new_nbrs.n_nbrs, begin, end);
      }
    };
    if constexpr (!Cached) {
      if (n_threads > 1) {
        return this->dispatch_candidates(
            graph, progress, n_threads, executor, generate,
            [&](std::size_t begin, std::size_t end) {
              return this->apply_routed(graph, begin, end, n_threads, executor);
            });
      }
    }
    return this->dispatch_candidates(graph, progress, n_threads, executor,
                                     generate);
  }

private:
  void generate_range(const Heap &graph, const Candidates &new_nbrs,
                      const Candidates &old_nbrs, std::size_t begin,
                      std::size_t end) {
    this->visit_candidates(
        new_nbrs, old_nbrs, new_nbrs.n_nbrs, begin, end,
        [&](std::size_t row, Idx p, auto visit) {
          auto queue = [&](Idx q, Out distance) {
            if constexpr (Cached) {
              auto [first, second] = std::minmax(p, q);
              this->queue_update(graph, this->edge_updates[row], first, second,
                                 distance);
            } else {
              this->queue_update(graph, this->edge_updates[row], p, q,
                                 distance);
            }
          };

          auto skip = [&](Idx q) {
            if constexpr (Cached) {
              auto [first, second] = std::minmax(p, q);
              return this->cache.contains(first, second);
            }
            return false;
          };
          score_candidates(kernel, this->distance, p, visit, skip, queue);
        });
  }
};

} // namespace rnndescent

#endif // RNN_NND_BATCH_H

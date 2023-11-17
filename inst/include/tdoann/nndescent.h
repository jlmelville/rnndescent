// BSD 2-Clause License
//
// Copyright 2019 James Melville
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

#ifndef TDOANN_NNDESCENT_H
#define TDOANN_NNDESCENT_H

#include <sstream>

#include "distancebase.h"
#include "heap.h"
#include "nndcommon.h"
#include "random.h"

namespace tdoann {

template <typename Out, typename Idx> class SerialLocalJoin {
  static constexpr auto npos = static_cast<Idx>(-1);

public:
  virtual ~SerialLocalJoin() = default;

  virtual auto update(NNDHeap<Out, Idx> &current_graph, Idx idx_p, Idx idx_q)
      -> std::size_t = 0;

  // Local join update: instead of updating item i with the neighbors of the
  // candidates of i, explore pairs (p, q) of candidates and treat q as a
  // candidate for p, and vice versa.
  auto execute(NNDHeap<Out, Idx> &current_graph,
               const NNHeap<Out, Idx> &new_nbrs, decltype(new_nbrs) &old_nbrs,
               NNDProgressBase &progress) -> unsigned long {
    const auto n_points = new_nbrs.n_points;
    const auto max_candidates = new_nbrs.n_nbrs;
    progress.set_n_batches(n_points);
    unsigned long num_updates = 0UL;
    for (Idx i = 0; i < n_points; i++) {
      for (Idx j = 0; j < max_candidates; j++) {
        // (new, new) pairs from j -> max_candidates
        auto new_j = new_nbrs.index(i, j);
        if (new_j == npos) {
          continue;
        }
        for (auto k = j; k < max_candidates; k++) {
          auto new_k = new_nbrs.index(i, k);
          if (new_k == npos) {
            continue;
          }
          num_updates += this->update(current_graph, new_j, new_k);
        }

        // (new, old) pairs from 0 -> max_candidates
        for (Idx k = 0; k < max_candidates; k++) {
          auto old_k = old_nbrs.index(i, k);
          if (old_k == npos) {
            continue;
          }
          num_updates += this->update(current_graph, new_j, old_k);
        }
      }
      if (progress.check_interrupt()) {
        break;
      }
      progress.batch_finished();
    }
    return num_updates;
  }
};

template <typename Out, typename Idx>
class LowMemSerialLocalJoin : public SerialLocalJoin<Out, Idx> {

public:
  const BaseDistance<Out, Idx> &distance;

  LowMemSerialLocalJoin(const BaseDistance<Out, Idx> &dist) : distance(dist) {}

  std::size_t update(NNDHeap<Out, Idx> &current_graph, Idx idx_p,
                     Idx idx_q) override {
    const auto dist_pq = distance.calculate(idx_p, idx_q);
    if (current_graph.accepts_either(idx_p, idx_q, dist_pq)) {
      return current_graph.checked_push_pair(idx_p, dist_pq, idx_q);
    }
    return 0; // No updates were made.
  }
};

template <typename Out, typename Idx>
class CacheSerialLocalJoin : public SerialLocalJoin<Out, Idx> {

public:
  const BaseDistance<Out, Idx> &distance;
  EdgeCache<Idx> cache;

  CacheSerialLocalJoin(const NNDHeap<Out, Idx> &graph,
                       const BaseDistance<Out, Idx> &dist)
      : distance(dist), cache(EdgeCache<Idx>::from_graph(graph)) {}

  std::size_t update(NNDHeap<Out, Idx> &current_graph, Idx idx_p,
                     Idx idx_q) override {
    Idx upd_p, upd_q;
    std::tie(upd_p, upd_q) = std::minmax(idx_p, idx_q);

    if (cache.contains(upd_p, upd_q)) {
      return 0; // No updates made
    }

    const auto dist = distance.calculate(upd_p, upd_q);
    std::size_t updates = 0;

    if (current_graph.accepts(upd_p, dist)) {
      current_graph.unchecked_push(upd_p, dist, upd_q);
      updates++;
    }

    if (upd_p != upd_q && current_graph.accepts(upd_q, dist)) {
      current_graph.unchecked_push(upd_q, dist, upd_p);
      updates++;
    }

    if (updates > 0) {
      cache.insert(upd_p, upd_q);
    }

    return updates;
  }
};

// This corresponds to the construction of new, old, new' and old' in
// Algorithm 2, with some minor differences:
// 1. old' and new' (the reverse candidates) are built at the same time as old
// and new respectively, based on the fact that if j is a candidate of new[i],
// then i is a reverse candidate of new[j]. This saves on building the entire
// reverse candidates list and then down-sampling.
// 2. Not all old members of current KNN are retained in the old candidates
// list, nor are rho * K new candidates sampled. Instead, the current members
// of the KNN are assigned into old and new based on their flag value, with the
// size of the final candidate list controlled by the maximum size of
// the candidates neighbors lists.
template <typename Out, typename Idx>
void build_candidates(const NNDHeap<Out, Idx> &current_graph,
                      NNHeap<Out, Idx> &new_nbrs, decltype(new_nbrs) &old_nbrs,
                      RandomGenerator &rand) {
  constexpr auto npos = static_cast<Idx>(-1);
  const std::size_t n_points = current_graph.n_points;
  const std::size_t n_nbrs = current_graph.n_nbrs;

  for (std::size_t i = 0, idx_offset = 0; i < n_points;
       i++, idx_offset += n_nbrs) {
    for (auto idx_ij = idx_offset; idx_ij < idx_offset + n_nbrs; idx_ij++) {
      const auto nbr = current_graph.idx[idx_ij];
      if (nbr == npos) {
        continue;
      }
      auto &nbrs = current_graph.flags[idx_ij] == 1 ? new_nbrs : old_nbrs;
      auto rand_weight = rand.unif(); // pairs will be processed in random order
      nbrs.checked_push_pair(i, rand_weight, nbr);
    }
  }
}

template <typename Out, typename Idx>
void flag_retained_new_candidates(NNDHeap<Out, Idx> &current_graph,
                                  const NNHeap<Out, Idx> &new_nbrs) {
  // shared with parallel code path
  flag_retained_new_candidates(current_graph, new_nbrs, 0,
                               current_graph.n_points);
}

// Pretty close to the NNDescentFull algorithm (#2 in the paper)
template <typename Out, typename Idx>
void nnd_build(NNDHeap<Out, Idx> &nn_heap,
               SerialLocalJoin<Out, Idx> &local_join,
               std::size_t max_candidates, uint32_t n_iters, double delta,
               RandomGenerator &rand, NNDProgressBase &progress) {
  const std::size_t n_points = nn_heap.n_points;
  for (auto iter = 0U; iter < n_iters; iter++) {
    NNHeap<Out, Idx> new_nbrs(n_points, max_candidates);
    decltype(new_nbrs) old_nbrs(n_points, max_candidates);

    build_candidates(nn_heap, new_nbrs, old_nbrs, rand);

    flag_retained_new_candidates(nn_heap, new_nbrs);

    auto num_updates =
        local_join.execute(nn_heap, new_nbrs, old_nbrs, progress);

    if (nnd_should_stop(progress, nn_heap, num_updates, delta)) {
      break;
    }
  }
}
} // namespace tdoann
#endif // TDOANN_NNDESCENT_H

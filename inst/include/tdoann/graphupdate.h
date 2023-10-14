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

#ifndef TDOANN_GRAPHUPDATE_H
#define TDOANN_GRAPHUPDATE_H

#include <numeric>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "heap.h"
#include "nndprogress.h"

namespace tdoann {

// These classes are used in the "local join" step of nearest neighbor descent.
// This is done in two separate steps to allow for parallelizing the (expensive)
// distance calculation: distance calculations are carried out in batches
// and stored, then updating the neighbor heaps is carried out in a serial
// step (this is like a map-reduce procedure where the reduce step is
// single-threaded).
// There is also a "HiMem" variant, where distances are cached.

namespace upd {

template <typename Idx> struct GraphCache {
private:
  std::vector<std::unordered_set<Idx>> seen;

public:
  GraphCache(std::size_t n_points, std::size_t n_nbrs,
             const std::vector<Idx> &idx_data)
      : seen(n_points) {
    for (Idx i = 0, innbrs = 0; i < n_points; i++, innbrs += n_nbrs) {
      for (std::size_t j = 0, idx_ij = innbrs; j < n_nbrs; j++, idx_ij++) {
        auto idx_p = idx_data[idx_ij];
        if (i > idx_p) {
          seen[idx_p].emplace(i);
        } else {
          seen[i].emplace(idx_p);
        }
      }
    }
  }

  // Static factory function
  template <typename DistOut>
  static GraphCache<Idx> from_heap(const NNDHeap<DistOut, Idx> &heap) {
    return GraphCache<Idx>(heap.n_points, heap.n_nbrs, heap.idx);
  }

  auto contains(const Idx &idx_p, const Idx &idx_q) const -> bool {
    return seen[idx_p].find(idx_q) != seen[idx_p].end();
  }

  auto insert(Idx idx_p, Idx idx_q) -> bool {
    return !seen[idx_p].emplace(idx_q).second;
  }
};

template <typename Distance> class GraphBatchUpdater {
public:
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  using Update = std::tuple<Idx, Idx, DistOut>;

  virtual ~GraphBatchUpdater() = default;

  virtual void generate(Idx idx_p, Idx idx_q, std::size_t key) = 0;
  virtual auto apply() -> std::size_t = 0;
};

template <typename Distance>
class BatchLowMem : public GraphBatchUpdater<Distance> {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  using Update = std::tuple<Idx, Idx, DistOut>;

public:
  NNDHeap<DistOut, Idx> &current_graph;
  const Distance &distance;
  std::vector<std::vector<Update>> updates;

  BatchLowMem(NNDHeap<DistOut, Idx> &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance),
        updates(current_graph.n_points) {}

  void generate(Idx idx_p, Idx idx_q, std::size_t key) {
    const auto dist_pq = distance(idx_p, idx_q);
    if (current_graph.accepts_either(idx_p, idx_q, dist_pq)) {
      updates[key].emplace_back(idx_p, idx_q, dist_pq);
    }
  }

  auto apply() -> std::size_t {
    std::size_t num_updates = 0;
    for (auto &update_set : updates) {
      for (auto &[upd_p, upd_q, upd_d] : update_set) {
        num_updates += current_graph.checked_push_pair(upd_p, upd_d, upd_q);
      }
      update_set.clear();
    }
    return num_updates;
  }
};

template <typename Distance>
class BatchHiMem : public GraphBatchUpdater<Distance> {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  using Update = std::tuple<Idx, Idx, DistOut>;

public:
  NNDHeap<DistOut, Idx> &current_graph;
  const Distance &distance;
  GraphCache<Idx> seen;
  std::vector<std::vector<Update>> updates;

  BatchHiMem(NNDHeap<DistOut, Idx> &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance),
        seen(GraphCache<Idx>::from_heap(current_graph)),
        updates(current_graph.n_points) {}

  void generate(Idx idx_p, Idx idx_q, std::size_t key) {
    auto [idx_pp, idx_qq] = std::minmax(idx_p, idx_q);

    if (seen.contains(idx_pp, idx_qq)) {
      return;
    }

    const auto dist_pq = distance(idx_pp, idx_qq);
    if (current_graph.accepts_either(idx_pp, idx_qq, dist_pq)) {
      updates[key].emplace_back(idx_pp, idx_qq, dist_pq);
    }
  }

  auto apply() -> std::size_t {
    std::size_t num_updates = 0;

    for (auto &update_set : updates) {
      for (const auto &[idx_p, idx_q, dist_pq] : update_set) {

        if (seen.contains(idx_p, idx_q)) {
          continue;
        }

        const bool bad_pd = !current_graph.accepts(idx_p, dist_pq);
        const bool bad_qd = !current_graph.accepts(idx_q, dist_pq);

        if (bad_pd && bad_qd) {
          continue;
        }

        std::size_t local_c = 0;
        if (!bad_pd) {
          current_graph.unchecked_push(idx_p, dist_pq, idx_q);
          local_c++;
        }

        if (idx_p != idx_q && !bad_qd) {
          current_graph.unchecked_push(idx_q, dist_pq, idx_p);
          local_c++;
        }

        if (local_c > 0) {
          seen.insert(idx_p, idx_q);
          num_updates += local_c;
        }
      }
      update_set.clear();
    }
    return num_updates;
  }
};
} // namespace upd
} // namespace tdoann

#endif // TDOANN_GRAPHUPDATE_H

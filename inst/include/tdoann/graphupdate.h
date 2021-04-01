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

#include <unordered_set>
#include <vector>

#include "heap.h"
#include "typedefs.h"

namespace tdoann {

namespace upd {

template <typename DistOut, typename Idx> struct Update {
  Idx p{0};
  Idx q{0};
  DistOut d{0};

  Update() = default;

  Update(Idx p, Idx q, DistOut d) : p(p), q(q), d(d) {}

  Update(Update &&) = default;
};

template <typename DistOut, typename Idx> struct GraphCacheConstructionInit {
  using DistanceOut = DistOut;
  using Index = Idx;
  static void init(const NNDHeap<DistOut, Idx> &neighbor_heap,
                   std::vector<std::unordered_set<Idx>> &seen) {
    const auto n_points = neighbor_heap.n_points;
    const auto n_nbrs = neighbor_heap.n_nbrs;
    for (Idx i = 0; i < n_points; i++) {
      std::size_t innbrs = i * n_nbrs;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        auto p = neighbor_heap.idx[innbrs + j];
        if (i > p) {
          seen[p].emplace(i);
        } else {
          seen[i].emplace(p);
        }
      }
    }
  }
};

template <typename DistOut, typename Idx> struct GraphCacheQueryInit {
  using DistanceOut = DistOut;
  using Index = Idx;
  static void init(const NNDHeap<DistOut, Idx> &neighbor_heap,
                   std::vector<std::unordered_set<Idx>> &seen) {
    const auto n_points = neighbor_heap.n_points;
    const auto n_nbrs = neighbor_heap.n_nbrs;
    for (std::size_t q = 0; q < n_points; q++) {
      std::size_t qnnbrs = q * n_nbrs;
      for (std::size_t k = 0; k < n_nbrs; k++) {
        std::size_t r = neighbor_heap.idx[qnnbrs + k];
        seen[q].emplace(r);
      }
    }
  }
};

template <typename DistOut, typename Idx,
          template <typename D, typename I> class GraphCacheInit =
              GraphCacheConstructionInit>
struct GraphCache {
  std::vector<std::unordered_set<Idx>> seen;

  GraphCache(const NNDHeap<DistOut, Idx> &neighbor_heap)
      : seen(neighbor_heap.n_points) {
    GraphCacheInit<DistOut, Idx>::init(neighbor_heap, seen);
  }

  auto contains(Idx &p, Idx &q) const -> bool {
    return seen[p].find(q) != seen[p].end();
  }

  auto insert(Idx p, Idx q) -> bool { return !seen[p].emplace(q).second; }

  auto size() const -> std::size_t {
    std::size_t sum = 0;
    for (std::size_t i = 0; i < seen.size(); i++) {
      sum += seen[i].size();
    }
    return sum;
  }
};

template <typename Distance> struct Batch {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;

  NNDHeap<DistOut, Idx> &current_graph;
  const Distance &distance;
  std::vector<std::vector<Update<DistOut, Idx>>> updates;

  Batch(NNDHeap<DistOut, Idx> &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance),
        updates(current_graph.n_points) {}

  void generate(Idx p, DistOut q, std::size_t key) {
    auto d = distance(p, q);
    if (current_graph.accepts_either(p, q, d)) {
      updates[key].emplace_back(p, q, d);
    }
  }

  auto apply() -> std::size_t {
    std::size_t c = 0;
    const auto n_points = updates.size();
    for (std::size_t i = 0; i < n_points; i++) {
      const auto n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        auto &update = updates[i][j];
        c += current_graph.checked_push_pair(update.p, update.d, update.q);
      }
      updates[i].clear();
    }
    return c;
  }
};

template <typename Distance> struct BatchHiMem {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;

  NNDHeap<DistOut, Idx> &current_graph;
  const Distance &distance;
  GraphCache<DistOut, Idx, GraphCacheConstructionInit> seen;
  std::vector<std::vector<Update<DistOut, Idx>>> updates;

  BatchHiMem(NNDHeap<DistOut, Idx> &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance), seen(current_graph),
        updates(current_graph.n_points) {}

  void generate(Idx p, Idx q, std::size_t key) {
    // canonicalize the order of (p, q) so that qq >= pp
    auto pp = p > q ? q : p;
    auto qq = pp == p ? q : p;

    if (seen.contains(pp, qq)) {
      return;
    }
    auto d = distance(p, q);
    if (current_graph.accepts_either(p, q, d)) {
      updates[key].emplace_back(pp, qq, d);
    }
  }

  auto apply() -> std::size_t {
    std::size_t c = 0;
    const auto n_points = updates.size();
    for (std::size_t i = 0; i < n_points; i++) {
      const auto n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        auto &update = updates[i][j];
        auto &p = update.p;
        auto &q = update.q;
        auto &d = update.d;

        bool bad_pd = !current_graph.accepts(p, d);
        bool bad_qd = !current_graph.accepts(q, d);
        if ((bad_pd && bad_qd) || seen.contains(p, q)) {
          continue;
        }

        std::size_t local_c = 0;
        if (!bad_pd) {
          current_graph.unchecked_push(p, d, q);
          local_c += 1;
        }

        if (p != q && !bad_qd) {
          current_graph.unchecked_push(q, d, p);
          local_c += 1;
        }

        if (local_c > 0) {
          seen.insert(p, q);
          c += local_c;
        }
      }
      updates[i].clear();
    }
    return c;
  }
};

template <typename Distance> struct Serial {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;

  NNDHeap<DistOut, Idx> &current_graph;
  const Distance &distance;
  Idx upd_p;
  Idx upd_q;
  DistOut upd_d;

  Serial(NNDHeap<DistOut, Idx> &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance),
        upd_p(current_graph.npos()), upd_q(current_graph.npos()), upd_d(0) {}

  auto generate_and_apply(Idx p, Idx q) -> std::size_t {
    generate(p, q, p);
    return apply();
  }

  void generate(Idx p, Idx q, std::size_t) {
    auto d = distance(p, q);
    if (current_graph.accepts_either(p, q, d)) {
      upd_p = p;
      upd_q = q;
      upd_d = d;
    } else {
      upd_p = current_graph.npos();
    }
  }

  auto apply() -> std::size_t {
    if (upd_p == current_graph.npos()) {
      return 0;
    }
    return current_graph.checked_push_pair(upd_p, upd_d, upd_q);
  }
};

template <typename Distance> struct SerialHiMem {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;

  NNDHeap<DistOut, Idx> &current_graph;
  const Distance &distance;
  GraphCache<DistOut, Idx, GraphCacheConstructionInit> seen;
  Idx upd_p;
  Idx upd_q;

  SerialHiMem(NNDHeap<DistOut, Idx> &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance), seen(current_graph),
        upd_p(current_graph.npos()), upd_q(current_graph.npos()) {}

  auto generate_and_apply(Idx p, Idx q) -> std::size_t {
    generate(p, q);
    return apply();
  }

  void generate(Idx p, Idx q) {
    // canonicalize the order of (p, q) so that qq >= pp
    auto pp = p > q ? q : p;
    auto qq = pp == p ? q : p;

    upd_p = pp;
    upd_q = qq;
  }

  auto apply() -> std::size_t {
    std::size_t c = 0;

    if (seen.contains(upd_p, upd_q)) {
      return c;
    }

    auto d = distance(upd_p, upd_q);

    if (current_graph.accepts(upd_p, d)) {
      current_graph.unchecked_push(upd_p, d, upd_q);
      c += 1;
    }

    if (upd_p != upd_q && current_graph.accepts(upd_q, d)) {
      current_graph.unchecked_push(upd_q, d, upd_p);
      c += 1;
    }

    if (c > 0) {
      seen.insert(upd_p, upd_q);
    }
    return c;
  }
};

template <typename Idx> struct NullNeighborSet {
  NullNeighborSet(std::size_t) {}
  auto contains(Idx) -> bool { return false; }
  void clear() {}
};

template <typename Idx> struct UnorderedNeighborSet {
  std::unordered_set<Idx> seen;

  UnorderedNeighborSet(std::size_t n_nbrs) : seen(n_nbrs) {}
  auto contains(Idx idx) -> bool { return !seen.emplace(idx).second; }
  void clear() { seen.clear(); }
};

template <typename Distance> struct QuerySerial {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;

  NNDHeap<DistOut, Idx> &current_graph;
  const Distance &distance;
  Idx ref;
  Idx query;
  DistOut dist;

  QuerySerial(NNDHeap<DistOut, Idx> &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance),
        ref(current_graph.npos()), query(current_graph.npos()), dist(0) {}

  auto generate_and_apply(Idx query_idx, Idx ref_idx) -> std::size_t {
    generate(query_idx, ref_idx, 0);
    return apply();
  }

  void generate(Idx query_idx, Idx ref_idx, std::size_t) {
    auto d = distance(ref_idx, query_idx);
    if (current_graph.accepts(query_idx, d)) {
      ref = ref_idx;
      query = query_idx;
      dist = d;
    } else {
      ref = current_graph.npos();
    }
  }

  auto apply() -> std::size_t {
    if (ref == current_graph.npos()) {
      return 0;
    }
    return current_graph.checked_push(query, dist, ref);
  }

  using NeighborSet = NullNeighborSet<Idx>;
};

template <typename Distance> struct QuerySerialHiMem {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;

  NNDHeap<DistOut, Idx> &current_graph;
  const Distance &distance;
  GraphCache<DistOut, Idx, GraphCacheQueryInit> seen;
  Idx ref_;
  Idx query_;

  QuerySerialHiMem(NNDHeap<DistOut, Idx> &current_graph,
                   const Distance &distance)
      : current_graph(current_graph), distance(distance), seen(current_graph),
        ref_(current_graph.npos()), query_(current_graph.npos()) {}

  auto generate_and_apply(Idx query_idx, Idx ref_idx) -> std::size_t {
    generate(query_idx, ref_idx, 0);
    return apply();
  }

  void generate(Idx query_idx, Idx ref_idx, std::size_t) {
    ref_ = ref_idx;
    query_ = query_idx;
  }

  auto apply() -> std::size_t {
    std::size_t c = 0;
    if (seen.contains(query_, ref_)) {
      return c;
    }

    auto d = distance(ref_, query_);
    c += current_graph.checked_push(query_, d, ref_);
    if (c > 0) {
      seen.insert(query_, ref_);
    }
    return c;
  }

  using NeighborSet = UnorderedNeighborSet<Idx>;
};

// Template aliases can't be declared inside a function, so this struct is
// necessary to avoid wanting to write e.g.:
// template <typename T>
// using  = Serial<T>;
// which won't compile.
template <template <typename> class Impl> struct Factory {
  template <typename Distance>
  static auto create(NNDHeap<typename Distance::Output,
                             typename Distance::Index> &current_graph,
                     Distance &distance) -> Impl<Distance> {
    return Impl<Distance>(current_graph, distance);
  }
};
} // namespace upd
} // namespace tdoann

#endif // TDOANN_GRAPHUPDATE_H

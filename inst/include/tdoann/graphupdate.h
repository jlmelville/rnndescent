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
struct Update {
  std::size_t p;
  std::size_t q;
  double d;

  Update() : p(0), q(0), d(0) {}

  Update(std::size_t p, std::size_t q, double d) : p(p), q(q), d(d) {}

  Update(Update &&) = default;
};

struct GraphCacheConstructionInit {
  static void init(const NeighborHeap &neighbor_heap,
                   std::vector<std::unordered_set<std::size_t>> &seen) {
    std::size_t n_points = neighbor_heap.n_points;
    std::size_t n_nbrs = neighbor_heap.n_nbrs;
    for (std::size_t i = 0; i < n_points; i++) {
      std::size_t innbrs = i * n_nbrs;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        std::size_t p = neighbor_heap.idx[innbrs + j];
        if (i > p) {
          seen[p].emplace(i);
        } else {
          seen[i].emplace(p);
        }
      }
    }
  }
};

struct GraphCacheQueryInit {
  static void init(const NeighborHeap &neighbor_heap,
                   std::vector<std::unordered_set<std::size_t>> &seen) {
    std::size_t n_points = neighbor_heap.n_points;
    std::size_t n_nbrs = neighbor_heap.n_nbrs;
    for (std::size_t q = 0; q < n_points; q++) {
      std::size_t qnnbrs = q * n_nbrs;
      for (std::size_t k = 0; k < n_nbrs; k++) {
        std::size_t r = neighbor_heap.idx[qnnbrs + k];
        seen[q].emplace(r);
      }
    }
  }
};

template <typename GraphCacheInit = GraphCacheConstructionInit>
struct GraphCache {
  std::vector<std::unordered_set<std::size_t>> seen;

  GraphCache(const NeighborHeap &neighbor_heap) : seen(neighbor_heap.n_points) {
    GraphCacheInit::init(neighbor_heap, seen);
  }

  bool contains(std::size_t &p, std::size_t &q) const {
    return seen[p].find(q) != seen[p].end();
  }

  bool insert(std::size_t p, std::size_t q) {
    return !seen[p].emplace(q).second;
  }

  std::size_t size() const {
    std::size_t sum = 0;
    for (std::size_t i = 0; i < seen.size(); i++) {
      sum += seen[i].size();
    }
    return sum;
  }
};

template <typename Distance> struct BatchGraphUpdater {
  NeighborHeap &current_graph;
  const Distance &distance;
  std::size_t n_nbrs;

  std::vector<std::vector<Update>> updates;

  BatchGraphUpdater(NeighborHeap &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance),
        n_nbrs(current_graph.n_nbrs), updates(current_graph.n_points) {}

  void generate(std::size_t p, std::size_t q, std::size_t key) {
    double d = distance(p, q);
    if (current_graph.accepts_either(p, q, d)) {
      updates[key].emplace_back(p, q, d);
    }
  }

  std::size_t apply() {
    std::size_t c = 0;
    std::size_t n_points = updates.size();
    for (std::size_t i = 0; i < n_points; i++) {
      std::size_t n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        auto &update = updates[i][j];
        c += current_graph.checked_push_pair(update.p, update.d, update.q);
      }
      updates[i].clear();
    }
    return c;
  }
};

template <typename Distance> struct BatchGraphUpdaterHiMem {
  NeighborHeap &current_graph;
  const Distance &distance;
  std::size_t n_nbrs;

  GraphCache<> seen;
  std::vector<std::vector<Update>> updates;

  BatchGraphUpdaterHiMem(NeighborHeap &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance),
        n_nbrs(current_graph.n_nbrs), seen(current_graph),
        updates(current_graph.n_points) {}

  void generate(std::size_t p, std::size_t q, std::size_t key) {
    // canonicalize the order of (p, q) so that qq >= pp
    std::size_t pp = p > q ? q : p;
    std::size_t qq = pp == p ? q : p;

    if (seen.contains(pp, qq)) {
      return;
    }
    double d = distance(p, q);
    if (current_graph.accepts_either(p, q, d)) {
      updates[key].emplace_back(pp, qq, d);
    }
  }

  std::size_t apply() {
    std::size_t c = 0;
    std::size_t n_points = updates.size();
    for (std::size_t i = 0; i < n_points; i++) {
      std::size_t n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        auto &update = updates[i][j];
        auto &p = update.p;
        auto &q = update.q;
        auto &d = update.d;

        const bool bad_pd = !current_graph.accepts(p, d);
        const bool bad_qd = !current_graph.accepts(q, d);
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

template <typename Distance> struct SerialGraphUpdater {
  NeighborHeap &current_graph;
  const Distance &distance;
  std::size_t n_nbrs;

  std::size_t upd_p;
  std::size_t upd_q;
  double upd_d;

  SerialGraphUpdater(NeighborHeap &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance),
        n_nbrs(current_graph.n_nbrs), upd_p(NeighborHeap::npos()),
        upd_q(NeighborHeap::npos()), upd_d(0) {}

  std::size_t generate_and_apply(std::size_t p, std::size_t q) {
    generate(p, q, p);
    return apply();
  }

  void generate(std::size_t p, std::size_t q, std::size_t) {
    double d = distance(p, q);
    if (current_graph.accepts_either(p, q, d)) {
      upd_p = p;
      upd_q = q;
      upd_d = d;
    } else {
      upd_p = NeighborHeap::npos();
    }
  }

  std::size_t apply() {
    if (upd_p == NeighborHeap::npos()) {
      return 0;
    }
    return current_graph.checked_push_pair(upd_p, upd_d, upd_q);
  }
};

template <typename Distance> struct SerialGraphUpdaterHiMem {
  NeighborHeap &current_graph;
  const Distance &distance;
  std::size_t n_nbrs;

  GraphCache<> seen;
  std::size_t upd_p;
  std::size_t upd_q;

  SerialGraphUpdaterHiMem(NeighborHeap &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance),
        n_nbrs(current_graph.n_nbrs), seen(current_graph),
        upd_p(NeighborHeap::npos()), upd_q(NeighborHeap::npos()) {}

  std::size_t generate_and_apply(std::size_t p, std::size_t q) {
    generate(p, q);
    return apply();
  }

  void generate(std::size_t p, std::size_t q) {
    // canonicalize the order of (p, q) so that qq >= pp
    std::size_t pp = p > q ? q : p;
    std::size_t qq = pp == p ? q : p;

    upd_p = pp;
    upd_q = qq;
  }

  std::size_t apply() {
    std::size_t c = 0;

    if (seen.contains(upd_p, upd_q)) {
      return c;
    }

    double d = distance(upd_p, upd_q);

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

// Caches all seen pairs. Compared to HiMem, it's a bit faster (~25%) but stores
// 6-7 times the number of items when tested on MNIST (70,000 items), k = 15,
// max_candidates = 20
template <typename Distance> struct SerialGraphUpdaterVeryHiMem {
  NeighborHeap &current_graph;
  const Distance &distance;
  std::size_t n_nbrs;

  GraphCache<> seen;
  std::size_t upd_p;
  std::size_t upd_q;

  SerialGraphUpdaterVeryHiMem(NeighborHeap &current_graph,
                              const Distance &distance)
      : current_graph(current_graph), distance(distance),
        n_nbrs(current_graph.n_nbrs), seen(current_graph),
        upd_p(NeighborHeap::npos()), upd_q(NeighborHeap::npos()) {}

  std::size_t generate_and_apply(std::size_t p, std::size_t q) {
    generate(p, q);
    return apply();
  }

  void generate(std::size_t p, std::size_t q) {
    // canonicalize the order of (p, q) so that qq >= pp
    std::size_t pp = p > q ? q : p;
    std::size_t qq = pp == p ? q : p;

    upd_p = pp;
    upd_q = qq;
  }

  std::size_t apply() {
    std::size_t c = 0;

    if (seen.insert(upd_p, upd_q)) {
      return c;
    }

    double d = distance(upd_p, upd_q);
    if (current_graph.accepts(upd_p, d)) {
      current_graph.unchecked_push(upd_p, d, upd_q);
      c += 1;
    }

    if (upd_p != upd_q && current_graph.accepts(upd_q, d)) {
      current_graph.unchecked_push(upd_q, d, upd_p);
      c += 1;
    }
    return c;
  }
};

// Not quite as memory hungry as SerialGraphUpdaterVeryHiMem, only storing
// ~30% more items, but shows negligible performance improvement.
template <typename Distance> struct BatchGraphUpdaterVeryHiMem {
  NeighborHeap &current_graph;
  const Distance &distance;
  std::size_t n_nbrs;

  GraphCache<> seen;
  std::vector<std::vector<Update>> updates;

  BatchGraphUpdaterVeryHiMem(NeighborHeap &current_graph,
                             const Distance &distance)
      : current_graph(current_graph), distance(distance),
        n_nbrs(current_graph.n_nbrs), seen(current_graph),
        updates(current_graph.n_points) {}

  void generate(std::size_t p, std::size_t q, std::size_t key) {
    // canonicalize the order of (p, q) so that qq >= pp
    std::size_t pp = p > q ? q : p;
    std::size_t qq = pp == p ? q : p;

    if (seen.contains(pp, qq)) {
      return;
    }
    double d = distance(p, q);
    if (current_graph.accepts_either(p, q, d)) {
      updates[key].emplace_back(pp, qq, d);
    }
  }

  std::size_t apply() {
    std::size_t c = 0;
    std::size_t n_points = updates.size();
    for (std::size_t i = 0; i < n_points; i++) {
      std::size_t n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        auto &update = updates[i][j];
        auto &p = update.p;
        auto &q = update.q;
        auto &d = update.d;
        if (seen.insert(p, q)) {
          continue;
        }

        if (current_graph.accepts(p, d)) {
          current_graph.unchecked_push(p, d, q);
          c += 1;
        }

        if (p != q && current_graph.accepts(q, d)) {
          current_graph.unchecked_push(q, d, p);
          c += 1;
        }
      }
      updates[i].clear();
    }
    return c;
  }
};

// For use in queries: whether to cache previously seen points
struct NullNeighborSet {
  NullNeighborSet(std::size_t n_nbrs) {}
  bool contains(std::size_t) { return false; }
  void clear() {}
};

struct UnorderedNeighborSet {
  std::unordered_set<std::size_t> seen;

  UnorderedNeighborSet(std::size_t n_nbrs) : seen(n_nbrs) {}
  bool contains(std::size_t idx) { return !seen.emplace(idx).second; }
  void clear() { seen.clear(); }
};

template <typename Distance> struct QuerySerialGraphUpdater {
  NeighborHeap &current_graph;
  const Distance &distance;
  std::size_t n_nbrs;

  std::size_t ref;
  std::size_t query;
  double dist;

  QuerySerialGraphUpdater(NeighborHeap &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance),
        n_nbrs(current_graph.n_nbrs), ref(NeighborHeap::npos()),
        query(NeighborHeap::npos()), dist(0) {}

  std::size_t generate_and_apply(std::size_t query_idx, std::size_t ref_idx) {
    generate(query_idx, ref_idx, 0);
    return apply();
  }

  void generate(std::size_t query_idx, std::size_t ref_idx, std::size_t) {
    double d = distance(ref_idx, query_idx);
    if (current_graph.accepts(query_idx, d)) {
      ref = ref_idx;
      query = query_idx;
      dist = d;
    } else {
      ref = NeighborHeap::npos();
    }
  }

  std::size_t apply() {
    if (ref == NeighborHeap::npos()) {
      return 0;
    }
    return current_graph.checked_push(query, dist, ref);
  }

  using NeighborSet = NullNeighborSet;
};

template <typename Distance> struct QuerySerialGraphUpdaterHiMem {
  NeighborHeap &current_graph;
  const Distance &distance;
  std::size_t n_nbrs;

  GraphCache<GraphCacheQueryInit> seen;
  std::size_t ref_;
  std::size_t query_;

  QuerySerialGraphUpdaterHiMem(NeighborHeap &current_graph,
                               const Distance &distance)
      : current_graph(current_graph), distance(distance),
        n_nbrs(current_graph.n_nbrs), seen(current_graph),
        ref_(NeighborHeap::npos()), query_(NeighborHeap::npos()) {}

  std::size_t generate_and_apply(std::size_t query_idx, std::size_t ref_idx) {
    generate(query_idx, ref_idx, 0);
    return apply();
  }

  void generate(std::size_t query_idx, std::size_t ref_idx, std::size_t) {
    ref_ = ref_idx;
    query_ = query_idx;
  }

  std::size_t apply() {
    std::size_t c = 0;
    if (seen.contains(query_, ref_)) {
      return c;
    }

    double d = distance(ref_, query_);
    c += current_graph.checked_push(query_, d, ref_);
    if (c > 0) {
      seen.insert(query_, ref_);
    }
    return c;
  }

  using NeighborSet = UnorderedNeighborSet;
};

template <typename Distance> struct QueryBatchGraphUpdater {
  NeighborHeap &current_graph;
  const Distance &distance;
  std::size_t n_nbrs;

  std::vector<std::vector<Update>> updates;

  QueryBatchGraphUpdater(NeighborHeap &current_graph, const Distance &distance)
      : current_graph(current_graph), distance(distance),
        n_nbrs(current_graph.n_nbrs), updates(current_graph.n_points) {}

  void generate(std::size_t query_idx, std::size_t ref_idx, std::size_t) {
    double d = distance(ref_idx, query_idx);
    if (current_graph.accepts(query_idx, d)) {
      updates[query_idx].emplace_back(query_idx, ref_idx, d);
    }
  }

  std::size_t apply() {
    std::size_t c = 0;
    std::size_t n_points = updates.size();
    for (std::size_t i = 0; i < n_points; i++) {
      std::size_t n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        auto &update = updates[i][j];
        c += current_graph.checked_push(update.p, update.d, update.q);
      }
      updates[i].clear();
    }
    return c;
  }

  using NeighborSet = NullNeighborSet;
};

template <typename Distance> struct QueryBatchGraphUpdaterHiMem {
  NeighborHeap &current_graph;
  const Distance &distance;
  std::size_t n_nbrs;

  GraphCache<GraphCacheQueryInit> seen;
  std::vector<std::vector<Update>> updates;

  QueryBatchGraphUpdaterHiMem(NeighborHeap &current_graph,
                              const Distance &distance)
      : current_graph(current_graph), distance(distance),
        n_nbrs(current_graph.n_nbrs), seen(current_graph),
        updates(current_graph.n_points) {}

  void generate(std::size_t query_idx, std::size_t ref_idx, std::size_t) {
    if (seen.contains(query_idx, ref_idx)) {
      return;
    }
    double d = distance(ref_idx, query_idx);
    if (current_graph.accepts(query_idx, d)) {
      updates[query_idx].emplace_back(query_idx, ref_idx, d);
    }
  }

  std::size_t apply() {
    std::size_t c = 0;
    std::size_t n_points = updates.size();
    for (std::size_t i = 0; i < n_points; i++) {
      std::size_t n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        auto &update = updates[i][j];
        auto &query_idx = update.p;
        auto &ref_idx = update.q;
        auto &d = update.d;
        bool bad_queryd = !current_graph.accepts(query_idx, d);
        if (bad_queryd || seen.contains(query_idx, ref_idx)) {
          continue;
        }
        if (!bad_queryd) {
          current_graph.unchecked_push(query_idx, d, ref_idx);
          seen.insert(query_idx, ref_idx);
          c += 1;
        }
      }
      updates[i].clear();
    }
    return c;
  }
  using NeighborSet = UnorderedNeighborSet;
};

// Template aliases can't be declared inside a function, so this struct is
// necessary to avoid wanting to write e.g.:
// template <typename T>
// using GraphUpdater = SerialGraphUpdater<T>;
// which won't compile.
template <template <typename> class GraphUpdater> struct GUFactory {
  template <typename Distance>
  static GraphUpdater<Distance> create(NeighborHeap &current_graph,
                                       Distance &distance) {
    return GraphUpdater<Distance>(current_graph, distance);
  }
};

} // namespace tdoann

#endif // TDOANN_GRAPHUPDATE_H

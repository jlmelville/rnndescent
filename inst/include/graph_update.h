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

#ifndef NND_UPDATE_H
#define NND_UPDATE_H

#include <unordered_set>
#include <vector>

#include "heap.h"

struct Update {
  const std::size_t p;
  const std::size_t q;
  const double d;

  Update() :
    p(0),
    q(0),
    d(0) {}

  Update(
    const std::size_t p,
    const std::size_t q,
    const double d
  ) :
    p(p),
    q(q),
    d(d)
  {}

  Update(Update&&) = default;
};

struct GraphCache {
  std::vector<std::unordered_set<std::size_t>> seen;

  GraphCache(const NeighborHeap& neighbor_heap) :
    seen(neighbor_heap.n_points)
  {
    const std::size_t n_points = neighbor_heap.n_points;
    const std::size_t n_nbrs = neighbor_heap.n_nbrs;
    for (std::size_t i = 0; i < n_points; i++) {
      const std::size_t innbrs = i * n_nbrs;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        std::size_t p = neighbor_heap.idx[innbrs + j];
        if (i > p) {
          seen[p].emplace(i);
        }
        else {
          seen[i].emplace(p);
        }
      }
    }
  }

  bool contains(const std::size_t& p,
                const std::size_t& q) const
  {
    return seen[p].find(q) != seen[p].end();
  }

  bool insert(std::size_t p, std::size_t q)
  {
    return !seen[p].emplace(q).second;
  }
};

template <typename Distance>
struct GraphUpdaterHiMem {
  GraphCache seen;
  const Distance& distance;
  std::vector<std::vector<Update>> updates;

  GraphUpdaterHiMem(
    const NeighborHeap& neighbor_heap,
    const Distance& distance
  ) :
    seen(neighbor_heap),
    distance(distance),
    updates(neighbor_heap.n_points)
  {}

  void generate(
      const NeighborHeap& current_graph,
      const std::size_t i,
      const std::size_t p,
      const std::size_t q,
      const std::size_t n_nbrs,
      const std::size_t pnnbrs
  )
  {
    // canonicalize the order of (p, q) so that qq >= pp
    std::size_t pp = p > q ? q : p;
    std::size_t qq = pp == p ? q : p;

    if (seen.contains(p, q)) {
      return;
    }
    double d = distance(p, q);
    if (d < current_graph.dist[pnnbrs] || d < current_graph.dist[q * n_nbrs]) {
      updates[i].emplace_back(pp, qq, d);
    }
  }

  size_t apply(
      NeighborHeap& current_graph
  )
  {
    std::size_t c = 0;
    const std::size_t n_points = updates.size();
    for (std::size_t i = 0; i < n_points; i++) {
      const std::size_t n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        const auto& update = updates[i][j];
        const auto& p = update.p;
        const auto& q = update.q;
        const auto& d = update.d;
        if (seen.insert(p, q)) {
          continue;
        }

        if (d < current_graph.distance(p, 0)) {
          current_graph.unchecked_push(p, d, q, true);
          c += 1;
        }

        if (p != q && d < current_graph.distance(q, 0)) {
          current_graph.unchecked_push(q, d, p, true);
          c += 1;
        }
      }
      updates[i].clear();
    }
    return c;
  }
};

template <typename Distance>
struct GraphUpdater {
  const Distance& distance;
  std::vector<std::vector<Update>> updates;

  GraphUpdater(
    const NeighborHeap& neighbor_heap,
    const Distance& distance
  ) :
    distance(distance),
    updates(neighbor_heap.n_points)
  {}

  void generate(
      const NeighborHeap& current_graph,
      const std::size_t i,
      const std::size_t p,
      const std::size_t q,
      const std::size_t n_nbrs,
      const std::size_t pnnbrs
  )
  {
    double d = distance(p, q);
    if (d < current_graph.dist[pnnbrs] || d < current_graph.dist[q * n_nbrs]) {
      updates[i].emplace_back(p, q, d);
    }
  }

  size_t apply(
      NeighborHeap& current_graph
  )
  {
    std::size_t c = 0;
    const std::size_t n_points = updates.size();
    for (std::size_t i = 0; i < n_points; i++) {
      const std::size_t n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        const auto& update = updates[i][j];
        c += current_graph.checked_push_pair(update.p, update.d, update.q, true);
      }
      updates[i].clear();
    }
    return c;
  }
};


#endif // NND_UPDATE_H

// BSD 2-Clause License
//
// Copyright 2021 James Melville
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

#ifndef TDOANN_HUB_H
#define TDOANN_HUB_H

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "heap.h"

namespace tdoann {

template <typename T>
auto reverse_nbr_counts_impl(const std::vector<T> &forward_nbrs,
                             std::size_t n_points) -> std::vector<std::size_t> {
  const std::size_t n_nbrs = forward_nbrs.size() / n_points;

  std::vector<std::size_t> counts(n_points);
  std::size_t innbrs;
  std::size_t inbr;

  for (std::size_t i = 0; i < n_points; i++) {
    innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      inbr = forward_nbrs[innbrs + j];
      if (inbr == i) {
        continue;
      }
      ++counts[inbr];
    }
  }
  return counts;
}

// treat the knn graph as directed where each point is the "head" of the
// directed edge, and each neighbor is the "tail". The head and tail nodes may
// be entirely disjoint, e.g. coming from query and reference nodes,
// respectively.
template <typename T>
auto reverse_nbr_counts_impl(const std::vector<T> &forward_nbrs,
                             std::size_t n_head_points,
                             std::size_t n_tail_points)
    -> std::vector<std::size_t> {
  const std::size_t n_nbrs = forward_nbrs.size() / n_head_points;

  std::vector<std::size_t> counts(n_tail_points);
  std::size_t innbrs;

  for (std::size_t i = 0; i < n_head_points; i++) {
    innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      ++counts[forward_nbrs[innbrs + j]];
    }
  }
  return counts;
}

template <typename T>
auto reverse_nbr_counts(const std::vector<T> &forward_nbrs,
                        std::size_t n_points, bool include_self)
    -> std::vector<std::size_t> {

  // include_self: either the index into the nbrs and the nbrs are from two
  // disjoint set of nodes or it's the same nodes and we don't mind counting
  // loops
  if (include_self) {
    auto n_tail_points =
        *std::max_element(forward_nbrs.begin(), forward_nbrs.end()) + 1;
    return reverse_nbr_counts_impl(forward_nbrs, n_points, n_tail_points);
  } else {
    return reverse_nbr_counts_impl(forward_nbrs, n_points);
  }
}

template <typename DistOut, typename Value> struct DistPair {
  DistOut dist;
  Value value;
  DistPair()
      : dist((std::numeric_limits<DistOut>::max)()),
        value(static_cast<Value>(-1)) {}
  DistPair(const DistOut &d, const Value &v) : dist(d), value(v) {}
};

template <typename D, typename V>
bool operator<(const DistPair<D, V> &x, const DistPair<D, V> &y) {
  return std::tie(x.value, x.dist) < std::tie(y.value, y.dist);
}

template <typename D, typename V>
bool operator>=(const DistPair<D, V> &x, const DistPair<D, V> &y) {
  return std::tie(x.value, x.dist) >= std::tie(y.value, y.dist);
}

template <typename NbrHeap>
auto kograph(const NbrHeap &heap)
    -> NNHeap<DistPair<typename NbrHeap::DistanceOut, typename NbrHeap::Index>,
              typename NbrHeap::Index> {
  using KDist =
      DistPair<typename NbrHeap::DistanceOut, typename NbrHeap::Index>;
  auto kos = reverse_nbr_counts(heap.idx, heap.n_points, true);

  NNHeap<KDist, typename NbrHeap::Index> kog(heap.n_points, heap.n_nbrs);
  for (typename NbrHeap::Index i = 0; i < heap.n_points; i++) {
    for (std::size_t j = 0; j < heap.n_nbrs; j++) {
      auto nbr = heap.index(i, j);
      auto d = heap.distance(i, j);
      kog.checked_push(i, KDist(d, kos[nbr]), nbr);
    }
  }

  return kog;
}

// Create a degree-adjusted graph, similar to the "Static Degree Adjustment with
// Constraints" algorithm described by Iwazaki and Miyasaki (2018).
// Basically this builds the reverse neighbor graph and then adds back in
// forward neighbors to fill in any gaps.
// n_rev_nbrs controls the maximum number of reverse neighbors to keep in the
// reverse neighbor heap.
// n_adj_nbrs controls both the number of neighbors in the returned graph, and
// also the number of number of forward neighbors used to create the reverse
// neighbor heap. e.g. if you pass in a 40NN heap but set n_adj_nbrs to 20, you
// are effectively using the 20NN to construct the reverse neighbors (and you
// will also only get a 20NN heap back).
// NOTE: this sorts the heap.
template <typename NbrHeap>
auto deg_adj_graph(NbrHeap &heap, std::size_t n_rev_nbrs,
                   std::size_t n_adj_nbrs) -> NbrHeap {
  auto rev_heap = reverse_heap(heap, n_rev_nbrs, n_adj_nbrs);
  NbrHeap adj_heap(heap.n_points, n_adj_nbrs);

  heap.deheap_sort();

  for (typename NbrHeap::Index i = 0; i < rev_heap.n_points; i++) {
    for (std::size_t j = 0; j < rev_heap.n_nbrs; j++) {
      if (adj_heap.is_full(i)) {
        break;
      }
      auto nbr = rev_heap.index(i, j);
      if (nbr == rev_heap.npos()) {
        continue;
      }
      adj_heap.checked_push(i, rev_heap.distance(i, j), nbr);
    }
    for (std::size_t j = 0; j < heap.n_nbrs; j++) {
      if (adj_heap.is_full(i)) {
        break;
      }
      auto nbr = heap.index(i, j);
      if (nbr == heap.npos()) {
        continue;
      }
      adj_heap.checked_push(i, heap.distance(i, j), nbr);
    }
  }
  return adj_heap;
}

template <typename NbrHeap> auto deg_adj_graph(const NbrHeap &heap) -> NbrHeap {
  return deg_adj_heap(heap, heap.n_nbrs, heap.n_nbrs);
}

// Similar to deg_adj_graph, but forward neighbors are added according to
// increasing k-occurrence
template <typename NbrHeap>
auto ko_adj_graph(const NbrHeap &heap, std::size_t n_rev_nbrs,
                  std::size_t n_adj_nbrs) -> NbrHeap {
  auto rev_heap = reverse_heap(heap, n_rev_nbrs, n_adj_nbrs);
  NbrHeap adj_heap(heap.n_points, n_adj_nbrs);

  auto ko_heap = kograph(heap);
  ko_heap.deheap_sort();

  for (typename NbrHeap::Index i = 0; i < rev_heap.n_points; i++) {
    for (std::size_t j = 0; j < rev_heap.n_nbrs; j++) {
      if (adj_heap.is_full(i)) {
        break;
      }
      auto nbr = rev_heap.index(i, j);
      if (nbr == rev_heap.npos()) {
        continue;
      }
      adj_heap.checked_push(i, rev_heap.distance(i, j), nbr);
    }
    for (std::size_t j = 0; j < heap.n_nbrs; j++) {
      if (adj_heap.is_full(i)) {
        break;
      }
      auto nbr = ko_heap.index(i, j);
      if (nbr == ko_heap.npos()) {
        continue;
      }
      adj_heap.checked_push(i, ko_heap.distance(i, j).dist, nbr);
    }
  }
  return adj_heap;
}

template <typename NbrHeap> auto ko_adj_graph(const NbrHeap &heap) -> NbrHeap {
  return ko_adj_graph(heap, heap.n_nbrs, heap.n_nbrs);
}

} // namespace tdoann
#endif // TDOANN_HUB_H

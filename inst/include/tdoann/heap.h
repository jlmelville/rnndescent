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

#ifndef TDOANN_HEAP_H
#define TDOANN_HEAP_H

#include <algorithm>
#include <limits>
#include <vector>

#include "parallel.h"

namespace tdoann {

// Base class storing neighbor data as a series of heaps
template <typename DistOut = float, typename Idx = uint32_t> struct NNDHeap {
  using DistanceOut = DistOut;
  using Index = Idx;

  static constexpr auto npos() -> Idx { return static_cast<Idx>(-1); }

  Idx n_points;
  Idx n_nbrs;
  std::vector<Idx> idx;
  std::vector<DistOut> dist;
  Idx n_nbrs1;
  std::vector<char> flags;

  NNDHeap(std::size_t n_points, std::size_t n_nbrs)
      : n_points(n_points), n_nbrs(n_nbrs), idx(n_points * n_nbrs, npos()),
        dist(n_points * n_nbrs, (std::numeric_limits<DistOut>::max)()),
        n_nbrs1(n_nbrs - 1), flags(n_points * n_nbrs, 0) {}

  NNDHeap(const NNDHeap &) = default;
  ~NNDHeap() = default;
  auto operator=(const NNDHeap &) -> NNDHeap & = default;

  auto contains(Idx row, Idx index) const -> bool {
    std::size_t rnnbrs = row * n_nbrs;
    for (std::size_t i = 0; i < n_nbrs; i++) {
      if (index == idx[rnnbrs + i]) {
        return true;
      }
    }
    return false;
  }

  // returns true if either p or q would accept a neighbor with distance d
  auto accepts_either(Idx p, Idx q, DistOut d) const -> bool {
    return (p < n_points && d < dist[p * n_nbrs]) ||
           (p != q && q < n_points && d < dist[q * n_nbrs]);
  }

  // returns true if p would accept a neighbor with distance d
  auto accepts(Idx p, DistOut d) const -> bool {
    return p < n_points && d < dist[p * n_nbrs];
  }

  auto checked_push_pair(Idx row, DistOut weight, Idx idx, char flag = 1)
      -> std::size_t {
    std::size_t c = checked_push(row, weight, idx, flag);
    if (row != idx) {
      c += checked_push(idx, weight, row, flag);
    }
    return c;
  }

  auto checked_push(Idx row, DistOut weight, Idx idx, char flag = 1)
      -> std::size_t {
    if (!accepts(row, weight) || contains(row, idx)) {
      return 0;
    }

    return unchecked_push(row, weight, idx, flag);
  }

  // This differs from the pynndescent version as it is truly unchecked
  auto unchecked_push(Idx row, DistOut weight, Idx index, char flag = 1)
      -> std::size_t {
    std::size_t r0 = row * n_nbrs;

    // insert val at position zero
    dist[r0] = weight;
    idx[r0] = index;
    flags[r0] = flag;

    // descend the heap, swapping values until the max heap criterion is met
    std::size_t i = 0;
    std::size_t i_swap = 0;
    while (true) {
      std::size_t ic1 = 2 * i + 1;
      std::size_t ic2 = ic1 + 1;

      if (ic1 >= n_nbrs) {
        break;
      } else if (ic2 >= n_nbrs) {
        if (dist[r0 + ic1] >= weight) {
          i_swap = ic1;
        } else {
          break;
        }
      } else if (dist[r0 + ic1] >= dist[r0 + ic2]) {
        if (weight < dist[r0 + ic1]) {
          i_swap = ic1;
        } else {
          break;
        }
      } else {
        if (weight < dist[r0 + ic2]) {
          i_swap = ic2;
        } else {
          break;
        }
      }

      std::size_t r0i = r0 + i;
      std::size_t r0is = r0 + i_swap;
      dist[r0i] = dist[r0is];
      idx[r0i] = idx[r0is];
      flags[r0i] = flags[r0is];

      i = i_swap;
    }

    std::size_t r0i = r0 + i;
    dist[r0i] = weight;
    idx[r0i] = index;
    flags[r0i] = flag;

    return 1;
  }

  void deheap_sort() {
    for (Idx i = 0; i < n_points; i++) {
      deheap_sort(i);
    }
  }

  void deheap_sort(Idx i) {
    std::size_t r0 = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs1; j++) {
      std::size_t n1j = n_nbrs1 - j;
      std::size_t r0nn1 = r0 + n1j;
      std::swap(idx[r0], idx[r0nn1]);
      std::swap(dist[r0], dist[r0nn1]);
      siftdown(r0, n1j);
    }
  }

  void siftdown(std::size_t r0, std::size_t len) {
    std::size_t elt = 0;
    std::size_t e21 = elt * 2 + 1;

    while (e21 < len) {
      std::size_t left_child = e21;
      std::size_t right_child = left_child + 1;
      std::size_t swap = elt;

      if (dist[r0 + swap] < dist[r0 + left_child]) {
        swap = left_child;
      }

      if (right_child < len && dist[r0 + swap] < dist[r0 + right_child]) {
        swap = right_child;
      }

      if (swap == elt) {
        break;
      } else {
        std::swap(dist[r0 + elt], dist[r0 + swap]);
        std::swap(idx[r0 + elt], idx[r0 + swap]);
        elt = swap;
      }
      e21 = elt * 2 + 1;
    }
  }

  auto index(Idx i, Idx j) const -> Idx { return idx[i * n_nbrs + j]; }
  auto index(Idx i, Idx j) -> Idx & { return idx[i * n_nbrs + j]; }

  auto distance(Idx i, Idx j) const -> DistOut { return dist[i * n_nbrs + j]; }
  auto distance(Idx i, Idx j) -> DistOut & { return dist[i * n_nbrs + j]; }

  auto flag(Idx i, Idx j) const -> char { return flags[i * n_nbrs + j]; }
  auto flag(Idx i, Idx j) -> char & { return flags[i * n_nbrs + j]; }

  auto max_distance(Idx i) const -> DistOut { return dist[i * n_nbrs]; }

  auto is_full(Idx i) const -> bool { return idx[i * n_nbrs] != npos(); }
};

// Like NNDHeap, but no flag vector
template <typename DistOut = float, typename Idx = uint32_t> struct NNHeap {
  using DistanceOut = DistOut;
  using Index = Idx;

  static constexpr auto npos() -> Idx { return static_cast<Idx>(-1); }

  Idx n_points;
  Idx n_nbrs;
  std::vector<Idx> idx;
  std::vector<DistOut> dist;
  Idx n_nbrs1;

  NNHeap(Idx n_points, Idx n_nbrs)
      : n_points(n_points), n_nbrs(n_nbrs), idx(n_points * n_nbrs, npos()),
        dist(n_points * n_nbrs, (std::numeric_limits<DistOut>::max)()),
        n_nbrs1(n_nbrs - 1) {}

  NNHeap(const NNHeap &) = default;
  ~NNHeap() = default;
  auto operator=(const NNHeap &) -> NNHeap & = default;

  auto contains(Idx row, Idx index) const -> bool {
    std::size_t rnnbrs = row * n_nbrs;
    for (std::size_t i = 0; i < n_nbrs; i++) {
      if (index == idx[rnnbrs + i]) {
        return true;
      }
    }
    return false;
  }

  // returns true if either p or q would accept a neighbor with distance d
  auto accepts_either(Idx p, Idx q, DistOut d) const -> bool {
    return (p < n_points && d < dist[p * n_nbrs]) ||
           (p != q && q < n_points && d < dist[q * n_nbrs]);
  }

  // returns true if p would accept a neighbor with distance d
  auto accepts(Idx p, DistOut d) const -> bool {
    return p < n_points && d < dist[p * n_nbrs];
  }

  auto checked_push_pair(std::size_t row, DistOut weight, Idx idx)
      -> std::size_t {
    std::size_t c = checked_push(row, weight, idx);
    if (row != idx) {
      c += checked_push(idx, weight, row);
    }
    return c;
  }

  auto checked_push(Idx row, DistOut weight, Idx idx) -> std::size_t {
    if (!accepts(row, weight) || contains(row, idx)) {
      return 0;
    }

    return unchecked_push(row, weight, idx);
  }

  auto unchecked_push(Idx row, DistOut weight, Idx index) -> std::size_t {
    std::size_t r0 = row * n_nbrs;

    // insert val at position zero
    dist[r0] = weight;
    idx[r0] = index;

    // descend the heap, swapping values until the max heap criterion is met
    std::size_t i = 0;
    std::size_t i_swap = 0;
    while (true) {
      std::size_t ic1 = 2 * i + 1;
      std::size_t ic2 = ic1 + 1;

      if (ic1 >= n_nbrs) {
        break;
      } else if (ic2 >= n_nbrs) {
        if (dist[r0 + ic1] >= weight) {
          i_swap = ic1;
        } else {
          break;
        }
      } else if (dist[r0 + ic1] >= dist[r0 + ic2]) {
        if (weight < dist[r0 + ic1]) {
          i_swap = ic1;
        } else {
          break;
        }
      } else {
        if (weight < dist[r0 + ic2]) {
          i_swap = ic2;
        } else {
          break;
        }
      }

      std::size_t r0i = r0 + i;
      std::size_t r0is = r0 + i_swap;
      dist[r0i] = dist[r0is];
      idx[r0i] = idx[r0is];

      i = i_swap;
    }

    std::size_t r0i = r0 + i;
    dist[r0i] = weight;
    idx[r0i] = index;

    return 1;
  }

  void deheap_sort() {
    for (Idx i = 0; i < n_points; i++) {
      deheap_sort(i);
    }
  }

  void deheap_sort(Idx i) {
    std::size_t r0 = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs1; j++) {
      std::size_t n1j = n_nbrs1 - j;
      std::size_t r0nn1 = r0 + n1j;
      std::swap(idx[r0], idx[r0nn1]);
      std::swap(dist[r0], dist[r0nn1]);
      siftdown(r0, n1j);
    }
  }

  void siftdown(std::size_t r0, std::size_t len) {
    std::size_t elt = 0;
    std::size_t e21 = elt * 2 + 1;

    while (e21 < len) {
      std::size_t left_child = e21;
      std::size_t right_child = left_child + 1;
      std::size_t swap = elt;

      if (dist[r0 + swap] < dist[r0 + left_child]) {
        swap = left_child;
      }

      if (right_child < len && dist[r0 + swap] < dist[r0 + right_child]) {
        swap = right_child;
      }

      if (swap == elt) {
        break;
      } else {
        std::swap(dist[r0 + elt], dist[r0 + swap]);
        std::swap(idx[r0 + elt], idx[r0 + swap]);
        elt = swap;
      }
      e21 = elt * 2 + 1;
    }
  }

  auto index(Idx i, Idx j) const -> Idx { return idx[i * n_nbrs + j]; }

  auto distance(Idx i, Idx j) const -> DistOut { return dist[i * n_nbrs + j]; }

  auto max_distance(Idx i) const -> DistOut { return dist[i * n_nbrs]; }

  auto is_full(Idx i) const -> bool { return idx[i * n_nbrs] != npos(); }
};

template <typename NbrHeap, typename Parallel = NoParallel>
void sort_heap(NbrHeap &heap, std::size_t block_size, std::size_t n_threads,
               std::size_t grain_size) {
  NullProgress progress;
  auto sort_worker = [&](std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; i++) {
      heap.deheap_sort(i);
    }
  };
  batch_parallel_for<Parallel>(sort_worker, progress, heap.n_points, block_size,
                               n_threads, grain_size);
}

template <typename NbrHeap> void sort_heap(NbrHeap &neighbor_heap) {
  neighbor_heap.deheap_sort();
}

// Construct a heap containing the reverse neighbors of the input neighbor heap.
// n_reverse_nbrs is the number of neighbors to retain in the returned heap
// (note that the full  reverse neighbor list for a vertex could be as large as
// N).
// n_forward_nbrs restricts the search to the specified number of nearest
// neighbors (i.e. effectively passes heap[, 1:n_forward_nbrs]).
template <typename NbrHeap>
auto reverse_heap(const NbrHeap &heap, typename NbrHeap::Index n_reverse_nbrs,
                  typename NbrHeap::Index n_forward_nbrs) -> NbrHeap {
  NbrHeap reversed(heap.n_points, n_reverse_nbrs);
  const auto n_fwd_nbrs = std::min(n_forward_nbrs, heap.n_nbrs);

  for (typename NbrHeap::Index i = 0; i < heap.n_points; i++) {
    for (std::size_t j = 0; j < n_fwd_nbrs; j++) {
      reversed.checked_push(heap.index(i, j), heap.distance(i, j), i);
    }
  }
  return reversed;
}

template <typename NbrHeap> auto reverse_heap(const NbrHeap &heap) -> NbrHeap {
  return reverse_heap(heap, heap.n_nbrs, heap.n_nbrs);
}

} // namespace tdoann
#endif // TDOANN_HEAP_H

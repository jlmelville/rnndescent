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

template <typename DistOut>
auto should_swap(std::size_t root, std::size_t len,
                 const std::vector<DistOut> &weights, const DistOut &weight,
                 std::size_t parent_idx, std::size_t &swap_idx) -> bool {
  constexpr std::size_t left_offset = 1;
  constexpr std::size_t right_offset = 2;

  std::size_t left_child = 2 * parent_idx + left_offset;
  if (left_child >= len) {
    return true;
  }

  std::size_t right_child = left_child + right_offset - left_offset;

  // Find the child with the maximum weight
  std::size_t max_child_idx =
      (right_child >= len ||
       weights[root + left_child] >= weights[root + right_child])
          ? left_child
          : right_child;

  if (weight >= weights[root + max_child_idx]) {
    return true;
  }

  swap_idx = max_child_idx;
  return false;
}

// Ensure max-heap property by moving the root element downwards until it is
// in the correct position in the heap.
template <typename DistOut, typename Idx>
void siftdown(std::size_t root, std::size_t len, std::vector<Idx> &idx,
              std::vector<DistOut> &weights) {

  std::size_t parent = 0;
  std::size_t swap;
  std::size_t left_child;
  std::size_t right_child;

  while (true) {
    left_child = 2 * parent + 1;
    if (left_child >= len) {
      break;
    }

    right_child = left_child + 1;

    // By default, set swap to parent
    swap = parent;

    // Should left child be swapped?
    if (weights[root + left_child] > weights[root + swap]) {
      swap = left_child;
    }

    // Should right child be swapped?
    if (right_child < len &&
        weights[root + right_child] > weights[root + swap]) {
      swap = right_child;
    }

    // If no swap is needed, we can break out of the loop
    if (swap == parent) {
      break;
    }

    // Swap the elements
    std::swap(weights[root + parent], weights[root + swap]);
    std::swap(idx[root + parent], idx[root + swap]);

    // Update parent to the swap position for the next iteration
    parent = swap;
  }
}

// Base class storing neighbor data as a series of heaps
template <typename DistOut = float, typename Idx = uint32_t> class NNDHeap {
public:
  using DistanceOut = DistOut;
  using Index = Idx;

  static constexpr auto npos() -> Idx { return static_cast<Idx>(-1); }

  Idx n_points;
  Idx n_nbrs;
  std::vector<Idx> idx;
  std::vector<DistOut> dist;
  Idx n_nbrs1;
  std::vector<uint8_t> flags;

  NNDHeap(std::size_t n_points, std::size_t n_nbrs)
      : n_points(n_points), n_nbrs(n_nbrs), idx(n_points * n_nbrs, npos()),
        dist(n_points * n_nbrs, (std::numeric_limits<DistOut>::max)()),
        n_nbrs1(n_nbrs - 1), flags(n_points * n_nbrs, 0) {}

  NNDHeap(const NNDHeap &) = default;
  auto operator=(const NNDHeap &) -> NNDHeap & = default;
  NNDHeap(NNDHeap &&) noexcept = default;
  auto operator=(NNDHeap &&) noexcept -> NNDHeap & = default;
  ~NNDHeap() = default;

  auto contains(Idx row, Idx index) const -> bool {
    std::size_t rnnbrs = row * n_nbrs;
    for (std::size_t i = 0; i < n_nbrs; i++) {
      if (index == idx[rnnbrs + i]) {
        return true;
      }
    }
    return false;
  }

  // returns true if either p or q would accept a neighbor with distance dist
  auto accepts_either(Idx idx_p, Idx idx_q, const DistOut &d_pq) const -> bool {
    return (idx_p < n_points && d_pq < dist[idx_p * n_nbrs]) ||
           (idx_p != idx_q && idx_q < n_points && d_pq < dist[idx_q * n_nbrs]);
  }

  // returns true if p would accept a neighbor with distance d
  auto accepts(Idx idx_p, const DistOut &d_pq) const -> bool {
    return idx_p < n_points && d_pq < dist[idx_p * n_nbrs];
  }

  auto checked_push_pair(Idx row, const DistOut &weight, Idx idx,
                         uint8_t flag = 1) -> std::size_t {
    std::size_t num_updates = checked_push(row, weight, idx, flag);
    if (row != idx) {
      // NOLINTNEXTLINE(readability-suspicious-call-argument)
      num_updates += checked_push(idx, weight, row, flag);
    }
    return num_updates;
  }

  auto checked_push(Idx row, const DistOut &weight, Idx idx, uint8_t flag = 1)
      -> std::size_t {
    if (!accepts(row, weight) || contains(row, idx)) {
      return 0;
    }

    return unchecked_push(row, weight, idx, flag);
  }

  // This differs from the pynndescent version as it is truly unchecked
  auto unchecked_push(Idx row, const DistOut &weight, Idx index,
                      uint8_t flag = 1) -> std::size_t {
    std::size_t root = row * n_nbrs;

    // insert val at position zero
    dist[root] = weight;
    idx[root] = index;
    flags[root] = flag;

    // descend the heap, swapping values until the max heap criterion is met
    std::size_t rel_parent = 0;
    std::size_t rel_swap = 0;

    while (true) {
      if (should_swap(root, n_nbrs, dist, weight, rel_parent, rel_swap)) {
        break;
      }

      std::size_t parent = root + rel_parent;
      std::size_t swap = root + rel_swap;
      dist[parent] = dist[swap];
      idx[parent] = idx[swap];
      flags[parent] = flags[swap];

      rel_parent = rel_swap;
    }

    std::size_t parent = root + rel_parent;
    dist[parent] = weight;
    idx[parent] = index;
    flags[parent] = flag;

    return 1;
  }

  void deheap_sort() {
    for (Idx i = 0; i < n_points; i++) {
      deheap_sort(i);
    }
  }
  // NOLINTBEGIN(readability-identifier-length)
  void deheap_sort(Idx i) {
    std::size_t r0 = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs1; j++) {
      std::size_t n1j = n_nbrs1 - j;
      std::size_t r0nn1 = r0 + n1j;
      std::swap(idx[r0], idx[r0nn1]);
      std::swap(dist[r0], dist[r0nn1]);
      siftdown(r0, n1j, idx, dist);
    }
  }

  auto index(Idx i, Idx j) const -> Idx { return idx[i * n_nbrs + j]; }
  auto index(Idx i, Idx j) -> Idx & { return idx[i * n_nbrs + j]; }

  auto distance(Idx i, Idx j) const -> DistOut { return dist[i * n_nbrs + j]; }
  auto distance(Idx i, Idx j) -> DistOut & { return dist[i * n_nbrs + j]; }

  auto flag(Idx i, Idx j) const -> uint8_t { return flags[i * n_nbrs + j]; }
  auto flag(Idx i, Idx j) -> uint8_t & { return flags[i * n_nbrs + j]; }

  auto max_distance(Idx i) const -> DistOut { return dist[i * n_nbrs]; }

  auto is_full(Idx i) const -> bool { return idx[i * n_nbrs] != npos(); }
  // NOLINTEND(readability-identifier-length)
};

template <typename T> auto limit_max() -> T {
  return (std::numeric_limits<T>::max)();
}

// Like NNDHeap, but no flag vector
template <typename DistOut = float, typename Idx = uint32_t,
          DistOut (*max_dist_func)() = limit_max>
struct NNHeap {
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
        dist(n_points * n_nbrs, max_dist_func()), n_nbrs1(n_nbrs - 1) {}

  NNHeap(const NNHeap &) = default;
  auto operator=(const NNHeap &) -> NNHeap & = default;
  NNHeap(NNHeap &&) noexcept = default;
  auto operator=(NNHeap &&) noexcept -> NNHeap & = default;
  ~NNHeap() = default;

  auto contains(Idx row, Idx index) const -> bool {
    auto start = idx.begin() + row * n_nbrs;
    auto end = start + n_nbrs;

    return std::find(start, end, index) != end;
  }

  // returns true if idx_p would accept a neighbor with distance d_pq
  auto accepts(Idx idx_p, const DistOut &d_pq) const -> bool {
    return idx_p < n_points && d_pq < dist[idx_p * n_nbrs];
  }

  // returns true if either idx_p or idx_q would accept a neighbor with distance
  // d_pq
  auto accepts_either(Idx idx_p, Idx idx_q, const DistOut &d_pq) const -> bool {
    return accepts(idx_p, d_pq) || (idx_p != idx_q && accepts(idx_q, d_pq));
  }

  auto checked_push_pair(std::size_t row, const DistOut &weight, Idx idx)
      -> std::size_t {
    std::size_t n_updates = checked_push(row, weight, idx);
    if (row != idx) {
      // NOLINTNEXTLINE(readability-suspicious-call-argument)
      n_updates += checked_push(idx, weight, row);
    }
    return n_updates;
  }

  auto checked_push(Idx row, const DistOut &weight, Idx idx) -> std::size_t {
    if (!accepts(row, weight) || contains(row, idx)) {
      return 0;
    }

    unchecked_push(row, weight, idx);
    return 1;
  }

  auto unchecked_push(Idx row, const DistOut &weight, Idx index) -> void {
    std::size_t root = row * n_nbrs;

    // insert val at position zero
    dist[root] = weight;
    idx[root] = index;

    // descend the heap, swapping values until the max heap criterion is met
    std::size_t rel_parent = 0;
    std::size_t rel_swap = 0;

    // Continue until the heap property is satisfied or we reach a leaf node
    while (!should_swap(root, n_nbrs, dist, weight, rel_parent, rel_swap)) {
      std::size_t parent = root + rel_parent;
      std::size_t swap = root + rel_swap;

      dist[parent] = dist[swap];
      idx[parent] = idx[swap];

      rel_parent = rel_swap;
    }

    std::size_t parent = root + rel_parent;
    dist[parent] = weight;
    idx[parent] = index;
  }

  void deheap_sort() {
    for (Idx i = 0; i < n_points; i++) {
      deheap_sort(i);
    }
  }

  // NOLINTBEGIN(readability-identifier-length)
  void deheap_sort(Idx i) {
    std::size_t root_offset = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs1; j++) {
      std::size_t remaining_size = n_nbrs1 - j;
      std::size_t last_elem_offset = root_offset + remaining_size;

      std::swap(idx[root_offset], idx[last_elem_offset]);
      std::swap(dist[root_offset], dist[last_elem_offset]);

      siftdown(root_offset, remaining_size, idx, dist);
    }
  }

  auto index(Idx i, Idx j) const -> Idx { return idx[i * n_nbrs + j]; }

  auto distance(Idx i, Idx j) const -> DistOut { return dist[i * n_nbrs + j]; }

  auto max_distance(Idx i) const -> DistOut { return dist[i * n_nbrs]; }

  auto is_full(Idx i) const -> bool { return idx[i * n_nbrs] != npos(); }
  // NOLINTEND(readability-identifier-length)
};

template <typename Parallel = NoParallel, typename NbrHeap>
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

template <typename Parallel = NoParallel, typename NbrHeap>
void sort_heap(NbrHeap &heap, std::size_t n_threads) {
  NullProgress progress;
  auto sort_worker = [&](std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; i++) {
      heap.deheap_sort(i);
    }
  };
  const std::size_t grain_size = 1;
  batch_parallel_for<Parallel>(sort_worker, progress, heap.n_points, n_threads,
                               grain_size);
}

template <typename NbrHeap> void sort_heap(NbrHeap &neighbor_heap) {
  neighbor_heap.deheap_sort();
}

// Construct a heap containing the reverse neighbors of the input neighbor heap.
// n_reverse_nbrs is the number of neighbors to retain in the returned heap
// (note that the full reverse neighbor list for a vertex could be as large as
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

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

#ifndef TDOANN_NNGRAPH_H
#define TDOANN_NNGRAPH_H

#include <mutex>
#include <vector>

#include "heap.h"
#include "parallel.h"

namespace tdoann {

template <typename DistOut = float, typename Idx = uint32_t>
struct SparseNNGraph {
  std::vector<std::size_t> row_ptr;
  std::vector<Idx> col_idx;
  std::vector<DistOut> dist;
  std::size_t n_points;

  SparseNNGraph(const std::vector<std::size_t> &row_ptr,
                const std::vector<Idx> &col_idx,
                const std::vector<DistOut> &dist)
      : row_ptr(row_ptr), col_idx(col_idx), dist(dist),
        n_points(row_ptr.size() - 1) {}

  using DistanceOut = DistOut;
  using Index = Idx;

  static constexpr auto npos() -> Idx { return static_cast<Idx>(-1); }

  static constexpr auto zero = static_cast<DistOut>(0);

  auto n_nbrs(Idx i) const -> std::size_t {
    return row_ptr[i + 1] - row_ptr[i];
  }

  auto index(Idx i, Idx j) const -> Idx {
    return col_idx[row_ptr[i] + static_cast<std::size_t>(j)];
  }

  auto distance(Idx i, Idx j) const -> DistOut {
    return dist[row_ptr[i] + static_cast<std::size_t>(j)];
  }

  auto distance(Idx i, Idx j) -> DistOut & {
    return dist[row_ptr[i] + static_cast<std::size_t>(j)];
  }

  void mark_for_deletion(Idx i, Idx j) { distance(i, j) = zero; }

  auto is_marked_for_deletion(Idx i, Idx j) -> bool {
    return distance(i, j) == zero;
  }
};

template <typename DistOut = float, typename Idx = uint32_t> struct NNGraph {
  std::vector<Idx> idx;
  std::vector<DistOut> dist;

  std::size_t n_points;
  std::size_t n_nbrs;

  static constexpr auto npos() -> Idx { return static_cast<Idx>(-1); }

  NNGraph(const std::vector<Idx> &idx, const std::vector<DistOut> &dist,
          std::size_t n_points)
      : idx(idx), dist(dist), n_points(n_points),
        n_nbrs(idx.size() / n_points) {}

  NNGraph(std::size_t n_points, std::size_t n_nbrs)
      : idx(std::vector<Idx>(n_points * n_nbrs, npos())),
        dist(std::vector<DistOut>(n_points * n_nbrs,
                                  (std::numeric_limits<DistOut>::max)())),
        n_points(n_points), n_nbrs(n_nbrs) {}

  using DistanceOut = DistOut;
  using Index = Idx;
};

template <typename NbrHeap>
void heap_to_graph(
    const NbrHeap &heap,
    NNGraph<typename NbrHeap::DistanceOut, typename NbrHeap::Index> &nn_graph) {
  nn_graph.idx = heap.idx;
  nn_graph.dist = heap.dist;
}

template <typename NbrHeap>
auto heap_to_graph(const NbrHeap &heap)
    -> NNGraph<typename NbrHeap::DistanceOut, typename NbrHeap::Index> {
  NNGraph<typename NbrHeap::DistanceOut, typename NbrHeap::Index> nn_graph(
      heap.n_points, heap.n_nbrs);
  heap_to_graph(heap, nn_graph);

  return nn_graph;
}

struct HeapAddSymmetric {
  template <typename NbrHeap>
  void push(NbrHeap &heap, std::size_t ref, std::size_t query,
            typename NbrHeap::DistanceOut d) {
    heap.checked_push_pair(ref, d, query);
  }
};

struct HeapAddQuery {
  template <typename NbrHeap>
  void push(NbrHeap &heap, std::size_t ref, std::size_t query,
            typename NbrHeap::DistanceOut d) {
    heap.checked_push(ref, d, query);
  }
};

struct LockingHeapAddSymmetric {
  static const constexpr std::size_t n_mutexes = 10;
  std::mutex mutexes[n_mutexes];

  template <typename NbrHeap>
  void push(NbrHeap &heap, std::size_t ref, std::size_t query,
            typename NbrHeap::DistanceOut d) {
    {
      std::lock_guard<std::mutex> guard(mutexes[ref % n_mutexes]);
      heap.checked_push(ref, d, query);
    }
    {
      std::lock_guard<std::mutex> guard(mutexes[query % n_mutexes]);
      heap.checked_push(query, d, ref);
    }
  }
};

// input idx vector is 0-indexed and transposed
// output heap index is 0-indexed
template <typename HeapAdd, typename NbrHeap>
void vec_to_heap(NbrHeap &current_graph,
                 const std::vector<typename NbrHeap::Index> &nn_idx,
                 std::size_t nrow,
                 const std::vector<typename NbrHeap::DistanceOut> &nn_dist,
                 std::size_t begin, std::size_t end, HeapAdd &heap_add,
                 bool transpose = true) {
  std::size_t n_nbrs = nn_idx.size() / nrow;
  for (auto i = begin; i < end; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      std::size_t ij = transpose ? i + j * nrow : j + i * n_nbrs;
      heap_add.push(current_graph, i, nn_idx[ij], nn_dist[ij]);
    }
  }
}

template <typename HeapAdd, typename Progress, typename Parallel,
          typename NbrHeap>
void vec_to_heap(NbrHeap &heap,
                 const std::vector<typename NbrHeap::Index> &nn_idx,
                 std::size_t n_points,
                 const std::vector<typename NbrHeap::DistanceOut> &nn_dist,
                 std::size_t block_size, std::size_t n_threads,
                 std::size_t grain_size, bool transpose) {
  HeapAdd heap_add;
  auto worker = [&](std::size_t begin, std::size_t end) {
    vec_to_heap<HeapAdd>(heap, nn_idx, n_points, nn_dist, begin, end, heap_add,
                         transpose);
  };
  Progress progress;
  batch_parallel_for<Parallel>(worker, progress, n_points, block_size,
                               n_threads, grain_size);
}

template <typename HeapAdd, typename Progress, typename NbrHeap>
void vec_to_heap(NbrHeap &heap,
                 const std::vector<typename NbrHeap::Index> &nn_idx,
                 std::size_t n_points,
                 const std::vector<typename NbrHeap::DistanceOut> &nn_dist,
                 std::size_t block_size, bool transpose) {
  HeapAdd heap_add;
  auto worker = [&](std::size_t begin, std::size_t end) {
    vec_to_heap<HeapAdd, NbrHeap>(heap, nn_idx, n_points, nn_dist, begin, end,
                                  heap_add, transpose);
  };
  Progress progress;
  batch_serial_for(worker, progress, n_points, block_size);
}

template <typename HeapAdd, typename Progress, typename Parallel,
          typename NbrHeap>
void graph_to_heap(NbrHeap &heap,
                   const NNGraph<typename NbrHeap::DistanceOut,
                                 typename NbrHeap::Index> &nn_graph,
                   std::size_t block_size, std::size_t n_threads,
                   std::size_t grain_size, bool transpose = false) {
  vec_to_heap<HeapAdd, Progress, Parallel>(
      heap, nn_graph.idx, nn_graph.n_points, nn_graph.dist, block_size,
      n_threads, grain_size, transpose);
}

template <typename HeapAdd, typename Progress, typename NbrHeap>
void graph_to_heap(NbrHeap &heap,
                   const NNGraph<typename NbrHeap::DistanceOut,
                                 typename NbrHeap::Index> &nn_graph,
                   std::size_t block_size, bool transpose = false) {
  vec_to_heap<HeapAdd, Progress>(heap, nn_graph.idx, nn_graph.n_points,
                                 nn_graph.dist, block_size, transpose);
}

template <typename HeapAdd, typename Parallel = NoParallel,
          typename Progress = NullProgress, typename NbrGraph>
void sort_knn_graph(NbrGraph &nn_graph, std::size_t block_size,
                    std::size_t n_threads = 0, std::size_t grain_size = 1) {
  NNHeap<typename NbrGraph::DistanceOut, typename NbrGraph::Index> heap(
      nn_graph.n_points, nn_graph.n_nbrs);
  graph_to_heap<HeapAdd, Progress, Parallel>(heap, nn_graph, block_size,
                                             n_threads, grain_size);
  sort_heap(heap, block_size, n_threads, grain_size);
  heap_to_graph(heap, nn_graph);
}

template <typename Distance>
void idx_to_graph(const Distance &distance,
                  const std::vector<typename Distance::Index> &idx,
                  std::vector<typename Distance::Output> &dist,
                  std::size_t n_nbrs, std::size_t begin, std::size_t end) {
  std::size_t innbrs = 0;
  std::size_t ij = 0;
  for (std::size_t i = begin; i < end; i++) {
    innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      ij = innbrs + j;
      dist[ij] = distance(idx[ij], i);
    }
  }
}

template <typename Distance, typename Progress>
auto idx_to_graph(const Distance &distance,
                  const std::vector<typename Distance::Index> &idx,
                  bool verbose)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  using Out = typename Distance::Output;
  using Index = typename Distance::Index;

  Progress progress(1, verbose);
  const std::size_t n_points = distance.ny;
  const std::size_t n_nbrs = idx.size() / n_points;
  std::vector<Out> dist(idx.size());

  auto worker = [&](std::size_t begin, std::size_t end) {
    idx_to_graph(distance, idx, dist, n_nbrs, begin, end);
  };
  batch_serial_for(worker, progress, n_points, 1024);

  return NNGraph<Out, Index>(idx, dist, n_points);
}

template <typename Distance, typename Progress, typename Parallel>
auto idx_to_graph(const Distance &distance,
                  const std::vector<typename Distance::Index> &idx,
                  std::size_t n_threads, bool verbose)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  using Out = typename Distance::Output;
  using Index = typename Distance::Index;

  Progress progress(1, verbose);
  const std::size_t n_points = distance.ny;
  const std::size_t n_nbrs = idx.size() / n_points;
  std::vector<Out> dist(idx.size());

  auto worker = [&](std::size_t begin, std::size_t end) {
    idx_to_graph(distance, idx, dist, n_nbrs, begin, end);
  };
  const std::size_t grain_size = 1;
  batch_parallel_for<Parallel>(worker, progress, n_points, 1024, n_threads,
                               grain_size);

  return NNGraph<Out, Index>(idx, dist, n_points);
}

} // namespace tdoann

#endif // TDOANN_NNGRAPH_H

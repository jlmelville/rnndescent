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

#include <array>
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

  auto n_nbrs(Idx idx) const -> std::size_t {
    return row_ptr[idx + 1] - row_ptr[idx];
  }

  // index of the ith non-zero nbr of idx
  auto index(Idx idx, Idx i) const -> Idx {
    return col_idx[row_ptr[idx] + static_cast<std::size_t>(i)];
  }

  // distance of the ith non-zero nbr of idx
  auto distance(Idx idx, Idx i) const -> DistOut {
    return dist[row_ptr[idx] + static_cast<std::size_t>(i)];
  }
  auto distance(Idx idx, Idx i) -> DistOut & {
    return dist[row_ptr[idx] + static_cast<std::size_t>(i)];
  }

  void mark_for_deletion(Idx idx, Idx i) { distance(idx, i) = zero; }

  auto is_marked_for_deletion(Idx idx, Idx i) const -> bool {
    return distance(idx, i) == zero;
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
  void push(NbrHeap &heap, typename NbrHeap::Index ref,
            typename NbrHeap::Index query,
            typename NbrHeap::DistanceOut dist_rq) {
    heap.checked_push_pair(ref, dist_rq, query);
  }
};

struct HeapAddQuery {
  template <typename NbrHeap>
  void push(NbrHeap &heap, typename NbrHeap::Index ref,
            typename NbrHeap::Index query,
            typename NbrHeap::DistanceOut dist_rq) {
    heap.checked_push(ref, dist_rq, query);
  }
};

struct LockingHeapAddSymmetric {
  static const constexpr std::size_t n_mutexes = 10;
  std::array<std::mutex, n_mutexes> mutexes;

  template <typename NbrHeap>
  void push(NbrHeap &heap, typename NbrHeap::Index ref,
            typename NbrHeap::Index query,
            typename NbrHeap::DistanceOut dist_rq) {
    {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
      std::lock_guard<std::mutex> guard(mutexes[ref % n_mutexes]);
      heap.checked_push(ref, dist_rq, query);
    }
    {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
      std::lock_guard<std::mutex> guard(mutexes[query % n_mutexes]);
      heap.checked_push(query, dist_rq, ref);
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
      std::size_t ij_1d = transpose ? i + j * nrow : j + i * n_nbrs;
      heap_add.push(current_graph, i, nn_idx[ij_1d], nn_dist[ij_1d]);
    }
  }
}

template <typename HeapAdd, typename NbrHeap>
void vec_to_heap(NbrHeap &heap,
                 const std::vector<typename NbrHeap::Index> &nn_idx,
                 std::size_t n_points,
                 const std::vector<typename NbrHeap::DistanceOut> &nn_dist,
                 std::size_t n_threads, bool transpose, ProgressBase &progress,
                 Executor &executor) {
  HeapAdd heap_add;
  auto worker = [&](std::size_t begin, std::size_t end) {
    vec_to_heap<HeapAdd>(heap, nn_idx, n_points, nn_dist, begin, end, heap_add,
                         transpose);
  };
  batch_parallel_for(worker, n_points, n_threads, progress, executor);
}

template <typename NbrHeap>
void vec_to_knn_heap(NbrHeap &heap,
                     const std::vector<typename NbrHeap::Index> &nn_idx,
                     std::size_t n_points,
                     const std::vector<typename NbrHeap::DistanceOut> &nn_dist,
                     std::size_t n_threads, bool transpose,
                     ProgressBase &progress, Executor &executor) {
  if (n_threads > 0) {
    vec_to_heap<LockingHeapAddSymmetric>(heap, nn_idx, n_points, nn_dist,
                                         n_threads, transpose, progress,
                                         executor);
  } else {
    vec_to_heap<HeapAddSymmetric>(heap, nn_idx, n_points, nn_dist, n_threads,
                                  transpose, progress, executor);
  }
}

template <typename NbrHeap>
void vec_to_query_heap(
    NbrHeap &heap, const std::vector<typename NbrHeap::Index> &nn_idx,
    std::size_t n_points,
    const std::vector<typename NbrHeap::DistanceOut> &nn_dist,
    std::size_t n_threads, bool transpose, ProgressBase &progress,
    Executor &executor) {

  vec_to_heap<HeapAddQuery>(heap, nn_idx, n_points, nn_dist, n_threads,
                            transpose, progress, executor);
}

// allow the use of auto heap = init_graph(nn_graph) and avoid long type in
// sort graph
template <typename NbrGraph>
auto init_heap(const NbrGraph &nn_graph)
    -> NNHeap<typename NbrGraph::DistanceOut, typename NbrGraph::Index> {
  return NNHeap<typename NbrGraph::DistanceOut, typename NbrGraph::Index>(
      nn_graph.n_points, nn_graph.n_nbrs);
}

// In a knn graph sort, it's assumed that the graph is a k-nearest neighbor
// graph, i.e. the neighbors of i are drawn from the same data as i and
// therefore that if the kth neighbor of i is j, then i may also be a neighbor
// of j. This sort will therefore not only modify the order of the neighbors
// but also replace some if it finds i in the neighbor list of any other item.
// If this isn't what you want, use `sort_query_graph`.
template <typename NbrGraph>
void sort_knn_graph(NbrGraph &nn_graph, std::size_t n_threads,
                    ProgressBase &progress, Executor &executor) {
  auto heap = init_heap(nn_graph);
  constexpr bool transpose = false;
  vec_to_knn_heap(heap, nn_graph.idx, nn_graph.n_points, nn_graph.dist,
                  n_threads, transpose, progress, executor);
  sort_heap(heap, n_threads, progress, executor);
  heap_to_graph(heap, nn_graph);
}

// In a query graph sort, it's assumed the graph is bipartite, i.e. the
// neighbors of i are not drawn from the same data as i. It's safe to run this
// on a knn graph (where the neighbors of i *are* from the same data as i).
template <typename NbrGraph>
void sort_query_graph(NbrGraph &nn_graph, std::size_t n_threads,
                      ProgressBase &progress, Executor &executor) {
  auto heap = init_heap(nn_graph);
  bool transpose = false;
  vec_to_query_heap(heap, nn_graph.idx, nn_graph.n_points, nn_graph.dist,
                    n_threads, transpose, progress, executor);
  sort_heap(heap, n_threads, progress, executor);
  heap_to_graph(heap, nn_graph);
}

template <typename Distance>
void idx_to_graph(const Distance &distance,
                  const std::vector<typename Distance::Index> &idx,
                  std::vector<typename Distance::Output> &dist,
                  std::size_t n_nbrs, std::size_t begin, std::size_t end) {
  std::size_t innbrs = 0;
  std::size_t ij_1d = 0;
  for (std::size_t i = begin; i < end; i++) {
    innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      ij_1d = innbrs + j;
      dist[ij_1d] = distance(idx[ij_1d], i);
    }
  }
}

template <typename Distance>
auto idx_to_graph(const Distance &distance,
                  const std::vector<typename Distance::Index> &idx,
                  std::size_t n_threads, ProgressBase &progress,
                  Executor &executor)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  using Out = typename Distance::Output;
  using Index = typename Distance::Index;

  const std::size_t n_points = distance.ny;
  const std::size_t n_nbrs = idx.size() / n_points;
  std::vector<Out> dist(idx.size());

  auto worker = [&](std::size_t begin, std::size_t end) {
    idx_to_graph(distance, idx, dist, n_nbrs, begin, end);
  };
  progress.set_n_iters(1);
  ExecutionParams exec_params{1024};
  batch_parallel_for(worker, n_points, n_threads, exec_params, progress,
                     executor);
  return NNGraph<Out, Index>(idx, dist, n_points);
}

} // namespace tdoann

#endif // TDOANN_NNGRAPH_H

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

#include "typedefs.h"

namespace tdoann {

struct NNGraph {
  std::vector<int> idx;
  std::vector<double> dist;

  std::size_t n_points;
  std::size_t n_nbrs;

  NNGraph(const std::vector<int> &idx, const std::vector<double> &dist,
          std::size_t n_points)
      : idx(idx), dist(dist), n_points(n_points),
        n_nbrs(idx.size() / n_points) {}

  NNGraph(std::size_t n_points, std::size_t n_nbrs)
      : idx(std::vector<int>(n_points * n_nbrs)),
        dist(std::vector<double>(n_points * n_nbrs)), n_points(n_points),
        n_nbrs(n_nbrs) {}
};

template <typename NbrHeap = SimpleNeighborHeap>
void heap_to_graph(const NbrHeap &heap, NNGraph &nn_graph) {
  for (std::size_t c = 0; c < nn_graph.n_points; c++) {
    std::size_t cnnbrs = c * nn_graph.n_nbrs;
    for (std::size_t r = 0; r < nn_graph.n_nbrs; r++) {
      std::size_t rc = cnnbrs + r;
      nn_graph.idx[rc] = static_cast<int>(heap.idx[rc]);
      nn_graph.dist[rc] = static_cast<double>(heap.dist[rc]);
    }
  }
}

template <typename NbrHeap = SimpleNeighborHeap>
NNGraph heap_to_graph(const NbrHeap &heap) {
  NNGraph nn_graph(heap.n_points, heap.n_nbrs);

  for (std::size_t c = 0; c < nn_graph.n_points; c++) {
    std::size_t cnnbrs = c * nn_graph.n_nbrs;
    for (std::size_t r = 0; r < nn_graph.n_nbrs; r++) {
      std::size_t rc = cnnbrs + r;
      nn_graph.idx[rc] = static_cast<int>(heap.idx[rc]);
      nn_graph.dist[rc] = static_cast<double>(heap.dist[rc]);
    }
  }

  return nn_graph;
}

struct HeapAddSymmetric {
  template <typename NbrHeap>
  void push(NbrHeap &heap, std::size_t ref, std::size_t query, double d) {
    heap.checked_push_pair(ref, d, query);
  }
};

struct HeapAddQuery {
  template <typename NbrHeap>
  void push(NbrHeap &heap, std::size_t ref, std::size_t query, double d) {
    heap.checked_push(ref, d, query);
  }
};

struct LockingHeapAddSymmetric {
  static const constexpr std::size_t n_mutexes = 10;
  std::mutex mutexes[n_mutexes];

  template <typename NbrHeap>
  void push(NbrHeap &heap, std::size_t ref, std::size_t query, double d) {
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
void vec_to_heap(NbrHeap &current_graph, const std::vector<int> &nn_idx,
                 std::size_t nrow, const std::vector<double> &nn_dist,
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

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
struct VecToHeapWorker : public BatchParallelWorker {
  NbrHeap &heap;
  const std::vector<int> &nn_idx;
  std::size_t nrow;
  const std::vector<double> &nn_dist;
  HeapAdd heap_add;
  bool transpose;

  VecToHeapWorker(NbrHeap &heap, const std::vector<int> &nn_idx,
                  std::size_t nrow, const std::vector<double> &nn_dist,
                  bool transpose = true)
      : heap(heap), nn_idx(nn_idx), nrow(nrow), nn_dist(nn_dist), heap_add(),
        transpose(transpose) {}

  void operator()(std::size_t begin, std::size_t end) {
    vec_to_heap<HeapAdd, NbrHeap>(heap, nn_idx, nrow, nn_dist, begin, end,
                                  heap_add, transpose);
  }
};

template <typename HeapAdd, typename Progress = NullProgress,
          typename NbrHeap = SimpleNeighborHeap, typename Parallel = NoParallel>
void vec_to_heap_parallel(NbrHeap &heap, std::vector<int> &nn_idx,
                          std::size_t n_points, std::vector<double> &nn_dist,
                          std::size_t n_threads, std::size_t block_size,
                          std::size_t grain_size) {
  VecToHeapWorker<HeapAdd, NbrHeap> worker(heap, nn_idx, n_points, nn_dist);
  Progress progress;
  batch_parallel_for<Parallel>(worker, progress, n_points, n_threads,
                               block_size, grain_size);
}

template <typename HeapAdd, typename Progress = NullProgress,
          typename NbrHeap = SimpleNeighborHeap, typename Parallel = NoParallel>
void graph_to_heap_parallel(NbrHeap &heap, const NNGraph &nn_graph,
                            std::size_t n_threads, std::size_t block_size,
                            std::size_t grain_size, bool transpose = false) {
  VecToHeapWorker<HeapAdd, NbrHeap> worker(
      heap, nn_graph.idx, nn_graph.n_points, nn_graph.dist, transpose);
  Progress progress;
  batch_parallel_for<Parallel>(worker, progress, nn_graph.n_points, n_threads,
                               block_size, grain_size);
}

template <typename HeapAdd, typename NbrHeap>
void vec_to_heap(NbrHeap &current_graph, const std::vector<int> &nn_idx,
                 std::size_t nrow, const std::vector<double> &nn_dist,
                 bool transpose = true) {
  HeapAdd heap_add;
  vec_to_heap<HeapAdd>(current_graph, nn_idx, nrow, nn_dist, 0, nrow, heap_add,
                       transpose);
}

template <typename HeapAdd, typename Progress = NullProgress,
          typename NbrHeap = SimpleNeighborHeap>
void vec_to_heap_serial(NbrHeap &heap, std::vector<int> &nn_idx,
                        std::size_t n_points, std::vector<double> &nn_dist,
                        std::size_t block_size) {
  VecToHeapWorker<HeapAdd, NbrHeap> worker(heap, nn_idx, n_points, nn_dist);
  Progress progress;
  batch_serial_for(worker, progress, n_points, block_size);
}

template <typename HeapAdd, typename Progress = NullProgress,
          typename NbrHeap = SimpleNeighborHeap>
void graph_to_heap_serial(NbrHeap &heap, const NNGraph &nn_graph,
                          std::size_t block_size, bool transpose = false) {
  VecToHeapWorker<HeapAdd, NbrHeap> worker(
      heap, nn_graph.idx, nn_graph.n_points, nn_graph.dist, transpose);
  Progress progress;
  batch_serial_for(worker, progress, nn_graph.n_points, block_size);
}

template <typename HeapAdd, typename Progress = NullProgress,
          typename NbrHeap = SimpleNeighborHeap, typename Parallel = NoParallel>
void sort_knn_graph_parallel(NNGraph &nn_graph, std::size_t n_threads,
                             std::size_t block_size, std::size_t grain_size) {
  NbrHeap heap(nn_graph.n_points, nn_graph.n_nbrs);
  graph_to_heap_parallel<HeapAdd, Progress>(heap, nn_graph, n_threads,
                                            block_size, grain_size);
  sort_heap_parallel(heap, n_threads, block_size, grain_size);

  heap_to_graph(heap, nn_graph);
}

template <typename HeapAdd, typename Progress = NullProgress,
          typename NbrHeap = SimpleNeighborHeap>
void sort_knn_graph(NNGraph &nn_graph) {
  NbrHeap heap(nn_graph.n_points, nn_graph.n_nbrs);
  graph_to_heap_serial<HeapAdd, Progress>(heap, nn_graph, 1000);

  heap.deheap_sort();

  heap_to_graph(heap, nn_graph);
}

} // namespace tdoann

#endif // TDOANN_NNGRAPH_H

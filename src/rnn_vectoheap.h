//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2019 James Melville
//
//  This file is part of rnndescent
//
//  rnndescent is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  rnndescent is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with rnndescent.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RNN_VECTOHEAP_H
#define RNN_VECTOHEAP_H

#include <mutex>

#include <Rcpp.h>

#include "RcppPerpendicular.h"
#include "tdoann/heap.h"

#include "rnn_parallel.h"
#include "rnn_progress.h"

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

// input idx R matrix is 1-indexed and transposed
// output heap index is 0-indexed
template <typename HeapAdd, typename NbrHeap>
void vec_to_heap(NbrHeap &current_graph, const std::vector<int> &nn_idx,
                 std::size_t nrow, const std::vector<double> &nn_dist,
                 std::size_t begin, std::size_t end, HeapAdd &heap_add,
                 int max_idx = (std::numeric_limits<int>::max)(),
                 bool transpose = true) {
  std::size_t n_nbrs = nn_idx.size() / nrow;

  for (auto i = begin; i < end; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      std::size_t ij = transpose ? i + j * nrow : j + i * n_nbrs;
      int k = nn_idx[ij] - 1;
      if (k < 0 || k > max_idx) {
        Rcpp::stop("Bad indexes in input: " + std::to_string(k));
      }
      double d = nn_dist[ij];
      heap_add.push(current_graph, i, k, d);
    }
  }
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
struct VecToHeapWorker : public BatchParallelWorker {
  NbrHeap &heap;
  const std::vector<int> &nn_idx;
  std::size_t nrow;
  const std::vector<double> &nn_dist;
  int max_idx;
  HeapAdd heap_add;
  bool transpose;

  VecToHeapWorker(NbrHeap &heap, const std::vector<int> &nn_idx,
                  std::size_t nrow, const std::vector<double> &nn_dist,
                  int max_idx = (std::numeric_limits<int>::max)(),
                  bool transpose = true)
      : heap(heap), nn_idx(nn_idx), nrow(nrow), nn_dist(nn_dist),
        max_idx(max_idx), heap_add(), transpose(transpose) {}

  void operator()(std::size_t begin, std::size_t end) {
    vec_to_heap<HeapAdd, NbrHeap>(heap, nn_idx, nrow, nn_dist, begin, end,
                                  heap_add, max_idx, transpose);
  }
};

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void vec_to_heap_parallel(NbrHeap &heap, std::vector<int> &nn_idx,
                          std::size_t n_points, std::vector<double> &nn_dist,
                          std::size_t n_threads, std::size_t block_size,
                          std::size_t grain_size,
                          int max_idx = (std::numeric_limits<int>::max)()) {
  VecToHeapWorker<HeapAdd, NbrHeap> worker(heap, nn_idx, n_points, nn_dist,
                                           max_idx);
  tdoann::NullProgress progress;
  batch_parallel_for(worker, progress, n_points, n_threads, block_size,
                     grain_size);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void vec_to_heap_parallelt(NbrHeap &heap, std::vector<int> &nn_idx,
                           std::size_t n_points, std::vector<double> &nn_dist,
                           std::size_t n_threads, std::size_t block_size,
                           std::size_t grain_size,
                           int max_idx = (std::numeric_limits<int>::max)()) {
  VecToHeapWorker<HeapAdd, NbrHeap> worker(heap, nn_idx, n_points, nn_dist,
                                           max_idx, false);
  tdoann::NullProgress progress;
  batch_parallel_for(worker, progress, n_points, n_threads, block_size,
                     grain_size);
}

template <typename HeapAdd, typename NbrHeap>
void vec_to_heap(NbrHeap &current_graph, const std::vector<int> &nn_idx,
                 std::size_t nrow, const std::vector<double> &nn_dist,
                 int max_idx = (std::numeric_limits<int>::max)(),
                 bool transpose = true) {
  HeapAdd heap_add;
  vec_to_heap<HeapAdd>(current_graph, nn_idx, nrow, nn_dist, 0, nrow, heap_add,
                       max_idx, transpose);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void vec_to_heap_serial(NbrHeap &heap, std::vector<int> &nn_idx,
                        std::size_t n_points, std::vector<double> &nn_dist,
                        std::size_t block_size,
                        int max_idx = (std::numeric_limits<int>::max)()) {
  VecToHeapWorker<HeapAdd, NbrHeap> worker(heap, nn_idx, n_points, nn_dist,
                                           max_idx);
  RInterruptableProgress progress;
  batch_serial_for(worker, progress, n_points, block_size);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void vec_to_heap_serialt(NbrHeap &heap, std::vector<int> &nn_idx,
                         std::size_t n_points, std::vector<double> &nn_dist,
                         std::size_t block_size,
                         int max_idx = (std::numeric_limits<int>::max)()) {
  VecToHeapWorker<HeapAdd, NbrHeap> worker(heap, nn_idx, n_points, nn_dist,
                                           max_idx, false);
  RInterruptableProgress progress;
  batch_serial_for(worker, progress, n_points, block_size);
}

#endif // RNN_VECTOHEAP_H

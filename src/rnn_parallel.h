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

// Generic parallel helper code

#ifndef RNN_PARALLEL_H
#define RNN_PARALLEL_H

#include <mutex>

#include <Rcpp.h>

#include "tdoann/heap.h"
#include "tdoann/progress.h"
#include "tdoann/typedefs.h"

#include "RcppPerpendicular.h"
#include "rnn.h"

struct BatchParallelWorker : public RcppPerpendicular::Worker {
  void after_parallel(std::size_t begin, std::size_t end) {}
};

template <typename Progress, typename Worker>
void batch_parallel_for(Worker &rnn_worker, Progress &progress, std::size_t n,
                        std::size_t block_size, std::size_t grain_size) {
  const auto n_blocks = (n / block_size) + 1;
  progress.set_n_blocks(n_blocks);
  for (std::size_t i = 0; i < n_blocks; i++) {
    const auto begin = i * block_size;
    const auto end = std::min(n, begin + block_size);
    RcppPerpendicular::parallelFor(begin, end, rnn_worker, grain_size);
    TDOANN_BREAKIFINTERRUPTED();
    rnn_worker.after_parallel(begin, end);
    TDOANN_BLOCKFINISHED();
  }
}

template <typename Progress, typename Worker>
void batch_serial_for(Worker &rnn_worker, Progress &progress, std::size_t n,
                      std::size_t block_size) {
  const auto n_blocks = (n / block_size) + 1;
  progress.set_n_blocks(n_blocks);
  for (std::size_t i = 0; i < n_blocks; i++) {
    const auto begin = i * block_size;
    const auto end = std::min(n, begin + block_size);
    rnn_worker(begin, end);
    TDOANN_BLOCKFINISHED();
  }
}

// Specialization designed to not compile: HeapAddSymmetric should not be used
// with parallel workers: use LockingHeapAddSymmetric
template <typename NbrHeap>
struct RToHeapWorker<HeapAddSymmetric, NbrHeap, RcppPerpendicular::RMatrix<int>,
                    RcppPerpendicular::RMatrix<double>, BatchParallelWorker> {};

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

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void r_to_heap_parallel(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                        Rcpp::NumericMatrix nn_dist, std::size_t block_size,
                        std::size_t grain_size,
                        int max_idx = (std::numeric_limits<int>::max)()) {
  RToHeapWorker<HeapAdd, NbrHeap, RcppPerpendicular::RMatrix<int>,
                RcppPerpendicular::RMatrix<double>, BatchParallelWorker>
      worker(heap, nn_idx, nn_dist, max_idx);
  tdoann::NullProgress progress;
  const std::size_t n_points = nn_idx.nrow();
  batch_parallel_for(worker, progress, n_points, block_size, grain_size);
}

template <typename NbrHeap = SimpleNeighborHeap>
struct HeapSortWorker : public BatchParallelWorker {
  NbrHeap &heap;
  HeapSortWorker(NbrHeap &heap) : heap(heap) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      heap.deheap_sort(i);
    }
  }
};

template <typename NbrHeap = SimpleNeighborHeap>
void sort_heap_parallel(NbrHeap &neighbor_heap, std::size_t block_size,
                        std::size_t grain_size) {
  tdoann::NullProgress null_progress;
  HeapSortWorker<NbrHeap> sort_worker(neighbor_heap);
  batch_parallel_for(sort_worker, null_progress, neighbor_heap.n_points,
                     block_size, grain_size);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void sort_knn_graph_parallel(Rcpp::IntegerMatrix nn_idx,
                             Rcpp::NumericMatrix nn_dist,
                             std::size_t block_size, std::size_t grain_size,
                             int max_idx = (std::numeric_limits<int>::max)()) {
  const std::size_t n_points = nn_idx.nrow();
  const std::size_t n_nbrs = nn_idx.ncol();

  NbrHeap heap(n_points, n_nbrs);
  r_to_heap_parallel<HeapAdd>(heap, nn_idx, nn_dist, block_size, grain_size,
                              max_idx);
  sort_heap_parallel(heap, block_size, grain_size);
  heap_to_r(heap, nn_idx, nn_dist);
}

#endif // RNN_PARALLEL_H

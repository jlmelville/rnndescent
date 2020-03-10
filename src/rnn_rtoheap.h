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

#ifndef RNN_RTOHEAP_H
#define RNN_RTOHEAP_H

#include <mutex>

#include "tdoann/heap.h"

#include "RcppPerpendicular.h"
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
template <typename HeapAdd, typename NbrHeap,
          typename IdxMatrix = Rcpp::IntegerMatrix,
          typename DistMatrix = Rcpp::NumericMatrix>
void r_to_heap(NbrHeap &current_graph, IdxMatrix nn_idx, DistMatrix nn_dist,
               const std::size_t begin, const std::size_t end,
               HeapAdd &heap_add,
               const int max_idx = (std::numeric_limits<int>::max)()) {
  const std::size_t n_nbrs = nn_idx.ncol();

  for (std::size_t i = begin; i < end; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      const int k = nn_idx(i, j) - 1;
      if (k < 0 || k > max_idx) {
        Rcpp::stop("Bad indexes in input: " + std::to_string(k));
      }
      double d = nn_dist(i, j);
      heap_add.push(current_graph, i, k, d);
    }
  }
}

template <typename HeapAdd, typename NbrHeap>
void r_to_heap(NbrHeap &current_graph, Rcpp::IntegerMatrix nn_idx,
               Rcpp::NumericMatrix nn_dist,
               int max_idx = (std::numeric_limits<int>::max)()) {
  std::size_t n_points = nn_idx.nrow();
  HeapAdd heap_add;
  r_to_heap<HeapAdd>(current_graph, nn_idx, nn_dist, 0, n_points, heap_add,
                     max_idx);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap,
          typename IdxMatrix = Rcpp::IntegerMatrix,
          typename DistMatrix = Rcpp::NumericMatrix, typename Base = Empty>
struct RToHeapWorker : public Base {
  NbrHeap &heap;
  IdxMatrix nn_idx;
  DistMatrix nn_dist;
  int max_idx;
  HeapAdd heap_add;

  RToHeapWorker(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                Rcpp::NumericMatrix nn_dist,
                int max_idx = (std::numeric_limits<int>::max)())
      : heap(heap), nn_idx(nn_idx), nn_dist(nn_dist), max_idx(max_idx),
        heap_add() {}

  void operator()(std::size_t begin, std::size_t end) {
    r_to_heap<HeapAdd, NbrHeap, IdxMatrix, DistMatrix>(
        heap, nn_idx, nn_dist, begin, end, heap_add, max_idx);
  }
};

// Specialization designed to not compile: HeapAddSymmetric should not be used
// with parallel workers: use LockingHeapAddSymmetric
template <typename NbrHeap>
struct RToHeapWorker<HeapAddSymmetric, NbrHeap, RcppPerpendicular::RMatrix<int>,
                     RcppPerpendicular::RMatrix<double>, BatchParallelWorker> {
};

;

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void r_to_heap_serial(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                      Rcpp::NumericMatrix nn_dist, std::size_t block_size,
                      int max_idx = (std::numeric_limits<int>::max)()) {
  RToHeapWorker<HeapAdd, NbrHeap, Rcpp::IntegerMatrix, Rcpp::NumericMatrix,
                Empty>
      worker(heap, nn_idx, nn_dist, max_idx);
  InterruptableProgress progress;
  const std::size_t n_points = nn_idx.nrow();
  batch_serial_for(worker, progress, n_points, block_size);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void r_to_heap_parallel(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                        Rcpp::NumericMatrix nn_dist, std::size_t block_size,
                        std::size_t grain_size,
                        int max_idx = (std::numeric_limits<int>::max)()) {
  RToHeapWorker<HeapAdd, NbrHeap, RcppPerpendicular::RMatrix<int>,
                RcppPerpendicular::RMatrix<double>, BatchParallelWorker>
      worker(heap, nn_idx, nn_dist, max_idx);
  tdoann::NullProgress progress;
  std::size_t n_points = nn_idx.nrow();
  batch_parallel_for(worker, progress, n_points, block_size, grain_size);
}

#endif // RNN_RTOHEAP_H

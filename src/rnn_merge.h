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

#ifndef RNN_MERGE_H
#define RNN_MERGE_H

#include <Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>

#include "rnn.h"
#include "rnn_parallel.h"

template <typename HeapAdd> struct MergeWorker : BatchParallelWorker {
  HeapAdd heap_add;
  const SimpleNeighborHeap &from;
  SimpleNeighborHeap &into;

  MergeWorker(const SimpleNeighborHeap &from, SimpleNeighborHeap &into)
      : heap_add(), from(from), into(into) {}

  void operator()(std::size_t begin, std::size_t end) {
    merge_window(from, into, begin, end, heap_add);
  }
};

// The thing that should not be: use LockingHeapAddSymmetric for parallel work
template <> struct MergeWorker<HeapAddSymmetric> {};

template <typename HeapAdd>
void merge(const SimpleNeighborHeap &from, SimpleNeighborHeap &into) {
  const auto n_points = from.n_points;
  HeapAdd heap_add;
  merge_window(from, into, 0, n_points, heap_add);
}

template <typename HeapAdd>
void merge_window(const SimpleNeighborHeap &from, SimpleNeighborHeap &into,
                  std::size_t begin, std::size_t end, HeapAdd &heap_add) {
  const auto n_nbrs = from.n_nbrs;
  for (std::size_t i = begin; i < end; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      std::size_t p = from.index(i, j);
      if (p == NeighborHeap::npos()) {
        continue;
      }
      auto d = from.distance(i, j);
      heap_add.push(into, i, p, d);
    }
  }
}

struct SerialHeapImpl {
  template <typename HeapAdd>
  void init(SimpleNeighborHeap &heap, Rcpp::IntegerMatrix nn_idx,
            Rcpp::NumericMatrix nn_dist) {
    r_to_heap<HeapAdd>(heap, nn_idx, nn_dist);
  }
  template <typename HeapAdd>
  void apply(const SimpleNeighborHeap &from, SimpleNeighborHeap &into) {
    merge<HeapAdd>(from, into);
  }
};

struct ParallelHeapImpl {
  std::size_t block_size;
  std::size_t grain_size;

  ParallelHeapImpl(std::size_t block_size, std::size_t grain_size)
      : block_size(block_size), grain_size(grain_size) {}

  template <typename HeapAdd>
  void init(SimpleNeighborHeap &heap, Rcpp::IntegerMatrix nn_idx,
            Rcpp::NumericMatrix nn_dist) {
    r_to_heap_parallel<HeapAdd>(heap, nn_idx, nn_dist, block_size, grain_size);
  }

  template <typename HeapAdd>
  void apply(const SimpleNeighborHeap &from, SimpleNeighborHeap &into) {
    MergeWorker<HeapAdd> worker(from, into);
    tdoann::NullProgress progress;
    batch_parallel_for(worker, progress, from.n_points, block_size, grain_size);
  }
};

template <typename MergeImpl, typename HeapAdd>
Rcpp::List merge_nn_impl(Rcpp::IntegerMatrix nn_idx1,
                         Rcpp::NumericMatrix nn_dist1,
                         Rcpp::IntegerMatrix nn_idx2,
                         Rcpp::NumericMatrix nn_dist2, MergeImpl &merge_impl) {
  const auto n_points = nn_idx1.nrow();
  const auto n_nbrs = nn_idx1.ncol();

  SimpleNeighborHeap nn_merged(n_points, n_nbrs);
  merge_impl.template init<HeapAdd>(nn_merged, nn_idx1, nn_dist1);

  SimpleNeighborHeap nn_from(n_points, n_nbrs);
  merge_impl.template init<HeapAdd>(nn_from, nn_idx2, nn_dist2);

  merge_impl.template apply<HeapAdd>(nn_from, nn_merged);
  nn_merged.deheap_sort();
  return heap_to_r(nn_merged);
}

#endif // RNN_MERGE_H

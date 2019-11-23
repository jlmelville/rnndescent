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

struct SerialHeapImpl {
  template <typename HeapAdd>
  void init(SimpleNeighborHeap &heap, Rcpp::IntegerMatrix nn_idx,
            Rcpp::NumericMatrix nn_dist) {
    r_to_heap<HeapAdd>(heap, nn_idx, nn_dist);
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
};

template <typename MergeImpl, typename HeapAdd>
Rcpp::List merge_nn_impl(Rcpp::IntegerMatrix nn_idx1,
                         Rcpp::NumericMatrix nn_dist1,
                         Rcpp::IntegerMatrix nn_idx2,
                         Rcpp::NumericMatrix nn_dist2, MergeImpl &merge_impl) {
  const auto n_points = nn_idx1.nrow();

  SimpleNeighborHeap nn_merged(n_points, nn_idx1.ncol());
  merge_impl.template init<HeapAdd>(nn_merged, nn_idx1, nn_dist1);
  merge_impl.template init<HeapAdd>(nn_merged, nn_idx2, nn_dist2);

  nn_merged.deheap_sort();
  return heap_to_r(nn_merged);
}

#endif // RNN_MERGE_H

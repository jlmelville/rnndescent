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

#include <Rcpp.h>

#include "rnn.h"
#include "rnn_merge.h"

#define MERGE_NN()                                                             \
  return merge_nn_impl<MergeImpl, HeapAdd>(nn_idx1, nn_dist1, nn_idx2,         \
                                           nn_dist2, merge_impl);

// [[Rcpp::export]]
Rcpp::List merge_nn(Rcpp::IntegerMatrix nn_idx1, Rcpp::NumericMatrix nn_dist1,
                    Rcpp::IntegerMatrix nn_idx2, Rcpp::NumericMatrix nn_dist2,
                    bool is_query, bool parallelize, std::size_t block_size,
                    std::size_t grain_size) {

  if (parallelize) {
    using MergeImpl = ParallelHeapImpl;
    MergeImpl merge_impl(block_size, grain_size);
    if (is_query) {
      using HeapAdd = HeapAddQuery;
      MERGE_NN();
    } else {
      using HeapAdd = LockedHeapAddSymmetric;
      MERGE_NN();
    }
  } else {
    using MergeImpl = SerialHeapImpl;
    MergeImpl merge_impl;
    if (is_query) {
      using HeapAdd = HeapAddQuery;
      MERGE_NN();
    } else {
      using HeapAdd = HeapAddSymmetric;
      MERGE_NN();
    }
  }
}

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
  std::size_t block_size;

  SerialHeapImpl(std::size_t block_size) : block_size(block_size) {}

  template <typename HeapAdd>
  void init(SimpleNeighborHeap &heap, Rcpp::IntegerMatrix nn_idx,
            Rcpp::NumericMatrix nn_dist) {
    r_to_heap_serial<HeapAdd>(heap, nn_idx, nn_dist, block_size);
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
Rcpp::List
merge_nn_impl(Rcpp::IntegerMatrix nn_idx1, Rcpp::NumericMatrix nn_dist1,
              Rcpp::IntegerMatrix nn_idx2, Rcpp::NumericMatrix nn_dist2,
              MergeImpl &merge_impl, bool verbose = false) {
  SimpleNeighborHeap nn_merged(nn_idx1.nrow(), nn_idx1.ncol());

  if (verbose) {
    ts("Merging graphs");
  }
  merge_impl.template init<HeapAdd>(nn_merged, nn_idx1, nn_dist1);
  merge_impl.template init<HeapAdd>(nn_merged, nn_idx2, nn_dist2);

  nn_merged.deheap_sort();
  return heap_to_r(nn_merged);
}

template <typename MergeImpl, typename HeapAdd>
Rcpp::List merge_nn_all_impl(Rcpp::List nn_graphs, MergeImpl &merge_impl,
                             bool verbose = false) {
  const auto n_graphs = static_cast<std::size_t>(nn_graphs.size());

  Rcpp::List nn_graph = nn_graphs[0];
  Rcpp::NumericMatrix nn_dist = nn_graph["dist"];
  Rcpp::IntegerMatrix nn_idx = nn_graph["idx"];

  RPProgress progress(n_graphs, verbose);
  SimpleNeighborHeap nn_merged(nn_idx.nrow(), nn_idx.ncol());
  merge_impl.template init<HeapAdd>(nn_merged, nn_idx, nn_dist);
  progress.iter_finished();

  // iterate over other graphs
  for (std::size_t i = 1; i < n_graphs; i++) {
    Rcpp::List nn_graphi = nn_graphs[i];
    Rcpp::NumericMatrix nn_disti = nn_graphi["dist"];
    Rcpp::IntegerMatrix nn_idxi = nn_graphi["idx"];
    merge_impl.template init<HeapAdd>(nn_merged, nn_idxi, nn_disti);
    TDOANN_ITERFINISHED()
  }

  nn_merged.deheap_sort();
  return heap_to_r(nn_merged);
}

#endif // RNN_MERGE_H

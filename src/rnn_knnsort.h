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

#ifndef RNN_KNNSORT_H
#define RNN_KNNSORT_H

#include <Rcpp.h>

#include "tdoann/heap.h"

#include "rnn_heapsort.h"
#include "rnn_heaptor.h"
#include "rnn_rtoheap.h"

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void sort_knn_graph_parallel(Rcpp::IntegerMatrix nn_idx,
                             Rcpp::NumericMatrix nn_dist,
                             std::size_t block_size, std::size_t grain_size,
                             int max_idx = (std::numeric_limits<int>::max)()) {
  std::size_t n_points = nn_idx.nrow();
  std::size_t n_nbrs = nn_idx.ncol();

  NbrHeap heap(n_points, n_nbrs);
  r_to_heap_parallel<HeapAdd>(heap, nn_idx, nn_dist, block_size, grain_size,
                              max_idx);
  sort_heap_parallel(heap, block_size, grain_size);
  heap_to_r(heap, nn_idx, nn_dist);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void sort_knn_graph(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist) {
  std::size_t n_points = nn_idx.nrow();
  std::size_t n_nbrs = nn_idx.ncol();

  NbrHeap heap(n_points, n_nbrs);
  r_to_heap_serial<HeapAdd>(heap, nn_idx, nn_dist, 1000);
  heap.deheap_sort();
  heap_to_r(heap, nn_idx, nn_dist);
}

#endif // RNN_KNNSORT_H

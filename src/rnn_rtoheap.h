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

#include <Rcpp.h>

#include "tdoann/heap.h"

#include "rnn_vectoheap.h"

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void r_to_heap_serial(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                      Rcpp::NumericMatrix nn_dist, std::size_t block_size,
                      int max_idx = (std::numeric_limits<int>::max)()) {
  std::size_t n_points = nn_idx.nrow();

  auto nn_idxv = Rcpp::as<std::vector<int>>(nn_idx);
  auto nn_distv = Rcpp::as<std::vector<double>>(nn_dist);
  return vec_to_heap_serial<HeapAdd, NbrHeap>(heap, nn_idxv, n_points, nn_distv,
                                              block_size, max_idx);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void r_to_heap_parallel(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                        Rcpp::NumericMatrix nn_dist, std::size_t n_threads,
                        std::size_t block_size, std::size_t grain_size,
                        int max_idx = (std::numeric_limits<int>::max)()) {
  auto nn_idxv = Rcpp::as<std::vector<int>>(nn_idx);
  auto nn_distv = Rcpp::as<std::vector<double>>(nn_dist);
  std::size_t n_points = nn_idx.nrow();

  vec_to_heap_parallel<HeapAdd, NbrHeap>(heap, nn_idxv, n_points, nn_distv,
                                         n_threads, block_size, grain_size,
                                         max_idx);
}

#endif // RNN_RTOHEAP_H

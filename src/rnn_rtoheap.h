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
#include "tdoann/nngraph.h"
#include "tdoann/progress.h"

#include "rnn_progress.h"
#include "rnn_util.h"

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void r_to_heap_serial(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                      Rcpp::NumericMatrix nn_dist, std::size_t block_size,
                      int max_idx = (std::numeric_limits<int>::max)()) {
  zero_index(nn_idx, max_idx);

  auto nn_idxv = Rcpp::as<std::vector<int>>(nn_idx);
  auto nn_distv = Rcpp::as<std::vector<double>>(nn_dist);
  std::size_t n_points = nn_idx.nrow();

  return tdoann::vec_to_heap_serial<HeapAdd, RInterruptableProgress, NbrHeap>(
      heap, nn_idxv, n_points, nn_distv, block_size);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void r_to_heap_parallel(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                        Rcpp::NumericMatrix nn_dist, std::size_t n_threads,
                        std::size_t block_size, std::size_t grain_size,
                        int max_idx = (std::numeric_limits<int>::max)()) {
  zero_index(nn_idx, max_idx);

  auto nn_idxv = Rcpp::as<std::vector<int>>(nn_idx);
  auto nn_distv = Rcpp::as<std::vector<double>>(nn_dist);
  std::size_t n_points = nn_idx.nrow();

  tdoann::vec_to_heap_parallel<HeapAdd, tdoann::NullProgress, NbrHeap>(
      heap, nn_idxv, n_points, nn_distv, n_threads, block_size, grain_size);
}

inline tdoann::NNGraph
r_to_graph(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
           int max_idx = (std::numeric_limits<int>::max)()) {
  zero_index(nn_idx, max_idx);

  auto nn_idxv = Rcpp::as<std::vector<int>>(nn_idx);
  auto nn_distv = Rcpp::as<std::vector<double>>(nn_dist);
  std::size_t n_points = nn_idx.nrow();

  return tdoann::NNGraph(nn_idxv, nn_distv, n_points);
}

template <typename Int>
inline std::vector<Int>
r_to_idx(Rcpp::IntegerMatrix nn_idx,
         int max_idx = (std::numeric_limits<int>::max)()) {
  zero_index(nn_idx, max_idx);

  return Rcpp::as<std::vector<Int>>(nn_idx);
}

#endif // RNN_RTOHEAP_H

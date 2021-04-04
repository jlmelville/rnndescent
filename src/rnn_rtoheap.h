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

#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

template <typename HeapAdd, typename NbrHeap>
void r_to_heap_serial(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                      Rcpp::NumericMatrix nn_dist, std::size_t block_size,
                      int max_idx = RNND_MAX_IDX, bool missing_ok = false,
                      bool transpose = true) {
  zero_index(nn_idx, max_idx, missing_ok);

  auto nn_idxv = Rcpp::as<std::vector<typename NbrHeap::Index>>(nn_idx);
  auto nn_distv = Rcpp::as<std::vector<typename NbrHeap::DistanceOut>>(nn_dist);
  std::size_t n_points = nn_idx.nrow();

  tdoann::vec_to_heap_serial<HeapAdd, RInterruptableProgress, NbrHeap>(
      heap, nn_idxv, n_points, nn_distv, block_size, transpose);
}

template <typename HeapAdd, typename NbrHeap>
auto r_to_heap_serial(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
                      std::size_t block_size = 1000, int max_idx = RNND_MAX_IDX,
                      bool missing_ok = false, bool transpose = true)
    -> NbrHeap {
  auto nn_idx_copy = Rcpp::clone(nn_idx);
  NbrHeap nn_heap(nn_idx_copy.nrow(), nn_idx_copy.ncol());
  r_to_heap_serial<tdoann::HeapAddQuery>(nn_heap, nn_idx_copy, nn_dist,
                                         block_size, RNND_MAX_IDX, missing_ok,
                                         transpose);
  return nn_heap;
}

template <typename HeapAdd, typename NbrHeap>
void r_to_heap_parallel(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                        Rcpp::NumericMatrix nn_dist, std::size_t n_threads,
                        std::size_t block_size, std::size_t grain_size,
                        int max_idx = RNND_MAX_IDX, bool missing_ok = false,
                        bool transpose = true) {
  zero_index(nn_idx, max_idx, missing_ok);

  auto nn_idxv = Rcpp::as<std::vector<typename NbrHeap::Index>>(nn_idx);
  auto nn_distv = Rcpp::as<std::vector<typename NbrHeap::DistanceOut>>(nn_dist);
  std::size_t n_points = nn_idx.nrow();

  tdoann::vec_to_heap_parallel<HeapAdd, tdoann::NullProgress, RParallel,
                               NbrHeap>(heap, nn_idxv, n_points, nn_distv,
                                        n_threads, block_size, grain_size,
                                        transpose);
}

template <typename DistOut, typename Idx>
auto r_to_graph(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
                int max_idx = RNND_MAX_IDX) -> tdoann::NNGraph<DistOut, Idx> {
  zero_index(nn_idx, max_idx, true);

  auto nn_idxv = Rcpp::as<std::vector<Idx>>(nn_idx);
  auto nn_distv = Rcpp::as<std::vector<DistOut>>(nn_dist);
  std::size_t n_points = nn_idx.nrow();

  return tdoann::NNGraph<DistOut, Idx>(nn_idxv, nn_distv, n_points);
}

template <typename Int>
inline auto r_to_idx(Rcpp::IntegerMatrix nn_idx, int max_idx = RNND_MAX_IDX)
    -> std::vector<Int> {
  zero_index(nn_idx, max_idx, true);

  return Rcpp::as<std::vector<Int>>(nn_idx);
}

template <typename Int>
inline auto r_to_idxt(Rcpp::IntegerMatrix nn_idx, int max_idx = RNND_MAX_IDX)
    -> std::vector<Int> {
  zero_index(nn_idx, max_idx, true);

  return Rcpp::as<std::vector<Int>>(Rcpp::transpose(nn_idx));
}

#endif // RNN_RTOHEAP_H

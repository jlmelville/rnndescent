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

const constexpr std::size_t DEFAULT_BLOCK_SIZE{1024};

template <typename HeapAdd, typename NbrHeap>
void r_to_heap(NbrHeap &heap, const Rcpp::IntegerMatrix &nn_idx,
               Rcpp::NumericMatrix nn_dist,
               std::size_t block_size = DEFAULT_BLOCK_SIZE,
               int max_idx = RNND_MAX_IDX, bool missing_ok = false,
               bool transpose = true) {
  auto nn_idx_copy = Rcpp::clone(nn_idx);
  zero_index(nn_idx_copy, max_idx, missing_ok);
  auto nn_idxv = Rcpp::as<std::vector<typename NbrHeap::Index>>(nn_idx_copy);
  auto nn_distv = Rcpp::as<std::vector<typename NbrHeap::DistanceOut>>(nn_dist);
  std::size_t n_points = nn_idx_copy.nrow();

  tdoann::vec_to_heap<HeapAdd, RInterruptableProgress>(
      heap, nn_idxv, n_points, nn_distv, block_size, transpose);
}

template <typename HeapAdd, typename NbrHeap>
auto r_to_heap(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
               std::size_t block_size = DEFAULT_BLOCK_SIZE,
               int max_idx = RNND_MAX_IDX, bool missing_ok = false,
               bool transpose = true) -> NbrHeap {
  NbrHeap nn_heap(nn_idx.nrow(), nn_idx.ncol());
  r_to_heap<HeapAdd>(nn_heap, nn_idx, nn_dist, block_size, max_idx, missing_ok,
                     transpose);
  return nn_heap;
}

template <typename HeapAdd, typename NbrHeap>
auto r_to_heap_missing_ok(Rcpp::IntegerMatrix nn_idx,
                          Rcpp::NumericMatrix nn_dist,
                          std::size_t block_size = DEFAULT_BLOCK_SIZE,
                          int max_idx = RNND_MAX_IDX, bool transpose = true)
    -> NbrHeap {
  return r_to_heap<HeapAdd, NbrHeap>(nn_idx, nn_dist, block_size, max_idx, true,
                                     transpose);
}

template <typename HeapAdd, typename NbrHeap>
void r_to_heap(NbrHeap &heap, const Rcpp::IntegerMatrix &nn_idx,
               Rcpp::NumericMatrix nn_dist, std::size_t n_threads,
               std::size_t grain_size,
               std::size_t block_size = DEFAULT_BLOCK_SIZE,
               int max_idx = RNND_MAX_IDX, bool missing_ok = false,
               bool transpose = true) {
  auto nn_idx_copy = Rcpp::clone(nn_idx);
  zero_index(nn_idx_copy, max_idx, missing_ok);
  auto nn_idxv = Rcpp::as<std::vector<typename NbrHeap::Index>>(nn_idx_copy);
  auto nn_distv = Rcpp::as<std::vector<typename NbrHeap::DistanceOut>>(nn_dist);
  std::size_t n_points = nn_idx_copy.nrow();

  tdoann::vec_to_heap<HeapAdd, tdoann::NullProgress, RParallel>(
      heap, nn_idxv, n_points, nn_distv, block_size, n_threads, grain_size,
      transpose);
}

template <typename HeapAdd, typename NbrHeap>
auto r_to_heap(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
               std::size_t n_threads, std::size_t grain_size = 1,
               std::size_t block_size = DEFAULT_BLOCK_SIZE,
               int max_idx = RNND_MAX_IDX, bool missing_ok = false,
               bool transpose = true) -> NbrHeap {
  NbrHeap nn_heap(nn_idx.nrow(), nn_idx.ncol());
  r_to_heap<HeapAdd>(nn_heap, nn_idx, nn_dist, n_threads, grain_size,
                     block_size, max_idx, missing_ok, transpose);
  return nn_heap;
}

template <typename HeapAdd, typename NbrHeap>
auto r_to_heap_missing_ok(Rcpp::IntegerMatrix nn_idx,
                          Rcpp::NumericMatrix nn_dist, std::size_t n_threads,
                          std::size_t grain_size = 1,
                          std::size_t block_size = DEFAULT_BLOCK_SIZE,
                          int max_idx = RNND_MAX_IDX, bool transpose = true)
    -> NbrHeap {
  return r_to_heap<HeapAdd, NbrHeap>(nn_idx, nn_dist, n_threads, grain_size,
                                     block_size, max_idx, true, transpose);
}

#endif // RNN_RTOHEAP_H

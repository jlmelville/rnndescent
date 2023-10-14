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

#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

// Add R graph to existing neighbor heap

template <typename NbrHeap>
void r_add_to_knn_heap(NbrHeap &heap, const Rcpp::IntegerMatrix &nn_idx,
                       Rcpp::NumericMatrix nn_dist, std::size_t n_threads,
                       bool missing_ok = true, int max_idx = RNND_MAX_IDX,
                       bool transpose = true) {
  auto nn_idx_copy = Rcpp::clone(nn_idx);
  zero_index(nn_idx_copy, max_idx, missing_ok);
  auto nn_idxv = Rcpp::as<std::vector<typename NbrHeap::Index>>(nn_idx_copy);
  auto nn_distv = Rcpp::as<std::vector<typename NbrHeap::DistanceOut>>(nn_dist);
  std::size_t n_points = nn_idx_copy.nrow();

  RInterruptableProgress progress;
  tdoann::vec_to_knn_heap<RParallel>(heap, nn_idxv, n_points, nn_distv,
                                     n_threads, transpose, progress);
}

template <typename NbrHeap>
void r_add_to_knn_heap(NbrHeap &heap, const Rcpp::IntegerMatrix &nn_idx,
                       Rcpp::NumericMatrix nn_dist, bool missing_ok = true,
                       int max_idx = RNND_MAX_IDX, bool transpose = true) {
  constexpr std::size_t n_threads = 0;
  return r_add_to_knn_heap(heap, nn_idx, nn_dist, n_threads, missing_ok,
                           RNND_MAX_IDX, transpose);
}

template <typename NbrHeap>
void r_add_to_query_heap(NbrHeap &heap, const Rcpp::IntegerMatrix &nn_idx,
                         Rcpp::NumericMatrix nn_dist, std::size_t n_threads,
                         bool missing_ok = true, int max_idx = RNND_MAX_IDX,
                         bool transpose = true) {
  auto nn_idx_copy = Rcpp::clone(nn_idx);
  zero_index(nn_idx_copy, max_idx, missing_ok);
  auto nn_idxv = Rcpp::as<std::vector<typename NbrHeap::Index>>(nn_idx_copy);
  auto nn_distv = Rcpp::as<std::vector<typename NbrHeap::DistanceOut>>(nn_dist);
  std::size_t n_points = nn_idx_copy.nrow();

  RInterruptableProgress progress;

  tdoann::vec_to_query_heap<RParallel>(heap, nn_idxv, n_points, nn_distv,
                                       n_threads, transpose, progress);
}

template <typename NbrHeap>
void r_add_to_query_heap(NbrHeap &heap, const Rcpp::IntegerMatrix &nn_idx,
                         Rcpp::NumericMatrix nn_dist, bool missing_ok = true,
                         int max_idx = RNND_MAX_IDX, bool transpose = true) {
  constexpr std::size_t n_threads = 0;
  return r_add_to_query_heap(heap, nn_idx, nn_dist, n_threads, missing_ok,
                             RNND_MAX_IDX, transpose);
}

// Convert R graph to neighbor heap

template <typename NbrHeap>
auto r_to_knn_heap(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
                   std::size_t n_threads, bool missing_ok = true,
                   int max_idx = RNND_MAX_IDX, bool transpose = true)
    -> NbrHeap {
  NbrHeap nn_heap(nn_idx.nrow(), nn_idx.ncol());
  r_add_to_knn_heap(nn_heap, nn_idx, nn_dist, n_threads, missing_ok, max_idx,
                    transpose);
  return nn_heap;
}

template <typename NbrHeap>
auto r_to_knn_heap(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
                   bool missing_ok = true, int max_idx = RNND_MAX_IDX,
                   bool transpose = true) -> NbrHeap {
  const constexpr std::size_t n_threads = 0;
  return r_to_knn_heap<NbrHeap>(nn_idx, nn_dist, n_threads, missing_ok, max_idx,
                                transpose);
}

template <typename NbrHeap>
auto r_to_query_heap(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
                     std::size_t n_threads, bool missing_ok = true,
                     int max_idx = RNND_MAX_IDX, bool transpose = true)
    -> NbrHeap {
  NbrHeap nn_heap(nn_idx.nrow(), nn_idx.ncol());
  r_add_to_query_heap(nn_heap, nn_idx, nn_dist, n_threads, missing_ok, max_idx,
                      transpose);
  return nn_heap;
}

template <typename NbrHeap>
auto r_to_query_heap(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
                     bool missing_ok = true, int max_idx = RNND_MAX_IDX,
                     bool transpose = true) -> NbrHeap {
  constexpr std::size_t n_threads = 0;
  return r_to_query_heap<NbrHeap>(nn_idx, nn_dist, n_threads, missing_ok,
                                  max_idx, transpose);
}

#endif // RNN_RTOHEAP_H

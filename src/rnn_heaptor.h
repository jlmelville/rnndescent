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

#ifndef RNN_HEAPTOR_H
#define RNN_HEAPTOR_H

#include <Rcpp.h>

#include "tdoann/heap.h"

// input heap index is 0-indexed
// output idx R matrix is 1-indexed and untransposed
template <typename NbrHeap>
void heap_to_r(const NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
               Rcpp::NumericMatrix nn_dist) {
  std::size_t n_points = heap.n_points;
  std::size_t n_nbrs = heap.n_nbrs;
  for (std::size_t i = 0; i < n_points; i++) {
    std::size_t innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      std::size_t ij = innbrs + j;
      nn_idx(i, j) = heap.idx[ij] + 1;
      nn_dist(i, j) = heap.dist[ij];
    }
  }
}

template <typename NbrHeap>
auto heap_to_r_impl(const NbrHeap &heap) -> Rcpp::List {
  std::size_t n_points = heap.n_points;
  std::size_t n_nbrs = heap.n_nbrs;

  Rcpp::IntegerMatrix nn_idx(n_points, n_nbrs);
  Rcpp::NumericMatrix nn_dist(n_points, n_nbrs);

  heap_to_r(heap, nn_idx, nn_dist);

  return Rcpp::List::create(Rcpp::Named("idx") = nn_idx,
                            Rcpp::Named("dist") = nn_dist);
}

// input heap index is 0-indexed
// output idx R matrix is 1-indexed and untransposed
template <typename NbrHeap> auto heap_to_r(NbrHeap &heap) -> Rcpp::List {
  tdoann::sort_heap(heap);
  return heap_to_r_impl(heap);
}

template <typename NbrHeap>
auto heap_to_r(NbrHeap &heap, std::size_t block_size, std::size_t n_threads,
               std::size_t grain_size) -> Rcpp::List {
  tdoann::sort_heap(heap, block_size, n_threads, grain_size);
  return heap_to_r_impl(heap);
}

template <typename NbrHeap>
auto heap_to_r(NbrHeap &heap, std::size_t n_threads) -> Rcpp::List {
  tdoann::sort_heap(heap, n_threads);
  return heap_to_r_impl(heap);
}

#endif // RNN_HEAPTOR_H

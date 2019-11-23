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

#include "rnn.h"

template <typename HeapAdd>
void merge(const SimpleNeighborHeap &from, SimpleNeighborHeap &into) {
  const auto n_points = from.n_points;
  const auto n_nbrs = from.n_nbrs;
  HeapAdd heap_add;
  for (std::size_t i = 0; i < n_points; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      std::size_t p = from.index(i, j);
      if (p == NeighborHeap::npos()) {
        continue;
      }
      auto d = from.distance(i, j);
      heap_add.push(into, i, p, d);
    }
  }
}

template <typename HeapAdd>
Rcpp::List
merge_nn_impl(Rcpp::IntegerMatrix nn_idx1, Rcpp::NumericMatrix nn_dist1,
              Rcpp::IntegerMatrix nn_idx2, Rcpp::NumericMatrix nn_dist2) {
  const auto n_points = nn_idx1.nrow();
  const auto n_nbrs = nn_idx1.ncol();

  SimpleNeighborHeap nn_merged(n_points, n_nbrs);
  r_to_heap<HeapAdd>(nn_merged, nn_idx1, nn_dist1);

  SimpleNeighborHeap nn_from(n_points, n_nbrs);
  r_to_heap<HeapAdd>(nn_from, nn_idx2, nn_dist2);

  merge<HeapAdd>(nn_from, nn_merged);

  nn_merged.deheap_sort();

  return heap_to_r(nn_merged);
}

#endif // RNN_MERGE_H

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

#include "rnn.h"
#include "rnn_bruteforceparallel.h"
#include "tdoann/bruteforce.h"
#include "tdoann/heap.h"
#include <Rcpp.h>

#define BRUTE_FORCE_BUILD()                                                    \
  return rnn_brute_force_impl<Distance>(data, k, parallelize, block_size,      \
                                        grain_size, verbose);

#define BRUTE_FORCE_QUERY()                                                    \
  return rnn_brute_force_query_impl<Distance>(                                 \
      x, y, k, parallelize, block_size, grain_size, verbose);

template <typename Distance>
Rcpp::List
rnn_brute_force_impl(Rcpp::NumericMatrix data, int k, bool parallelize = false,
                     std::size_t block_size = 64, std::size_t grain_size = 1,
                     bool verbose = false) {
  const std::size_t n_points = data.nrow();
  const std::size_t n_nbrs = k;

  const std::size_t ndim = data.ncol();
  data = Rcpp::transpose(data);
  auto data_vec = Rcpp::as<std::vector<typename Distance::Input>>(data);

  Distance distance(data_vec, ndim);
  tdoann::SimpleNeighborHeap neighbor_heap(n_points, n_nbrs);

  if (parallelize) {
    RPProgress progress(1, verbose);
    nnbf_parallel(neighbor_heap, distance, progress, block_size, grain_size);
  } else {
    RPProgress progress(n_points, verbose);
    nnbf(neighbor_heap, distance, progress);
  }

  return heap_to_r(neighbor_heap);
}

// [[Rcpp::export]]
Rcpp::List rnn_brute_force(Rcpp::NumericMatrix data, int k,
                           const std::string &metric = "euclidean",
                           bool parallelize = false,
                           std::size_t block_size = 64,
                           std::size_t grain_size = 1, bool verbose = false) {
  DISPATCH_ON_DISTANCES(BRUTE_FORCE_BUILD)
}

template <typename Distance>
Rcpp::List
rnn_brute_force_query_impl(Rcpp::NumericMatrix x, Rcpp::NumericMatrix y, int k,
                           bool parallelize = false,
                           std::size_t block_size = 64,
                           std::size_t grain_size = 1, bool verbose = false) {
  const std::size_t n_xpoints = x.nrow();
  const std::size_t n_ypoints = y.nrow();
  const std::size_t n_nbrs = k;

  const std::size_t ndim = x.ncol();
  x = Rcpp::transpose(x);
  auto x_vec = Rcpp::as<std::vector<typename Distance::Input>>(x);

  y = Rcpp::transpose(y);
  auto y_vec = Rcpp::as<std::vector<typename Distance::Input>>(y);

  Distance distance(x_vec, y_vec, ndim);
  tdoann::SimpleNeighborHeap neighbor_heap(n_ypoints, n_nbrs);

  if (parallelize) {
    RPProgress progress(1, verbose);
    nnbf_parallel_query(neighbor_heap, distance, n_xpoints, progress,
                        block_size, grain_size);
  } else {
    RPProgress progress(n_xpoints, verbose);
    nnbf_query(neighbor_heap, distance, n_xpoints, progress);
  }

  return heap_to_r(neighbor_heap);
}

// [[Rcpp::export]]
Rcpp::List rnn_brute_force_query(Rcpp::NumericMatrix x, Rcpp::NumericMatrix y,
                                 int k, const std::string &metric = "euclidean",
                                 bool parallelize = false,
                                 std::size_t block_size = 64,
                                 std::size_t grain_size = 1,
                                 bool verbose = false) {
  DISPATCH_ON_DISTANCES(BRUTE_FORCE_QUERY)
}

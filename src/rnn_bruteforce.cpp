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

#include <Rcpp.h>

#include "tdoann/bruteforce.h"
#include "tdoann/heap.h"

#include "rnn.h"
#include "rnn_bruteforceparallel.h"
#include "rnn_progress.h"

using namespace Rcpp;

#define BRUTE_FORCE_BUILD()                                                    \
  return rnn_brute_force_impl<Distance>(data, k, parallelize, block_size,      \
                                        grain_size, verbose);

#define BRUTE_FORCE_QUERY()                                                    \
  return rnn_brute_force_query_impl<Distance>(                                 \
      x, y, k, parallelize, block_size, grain_size, verbose);

template <typename Distance>
auto rnn_brute_force_impl(NumericMatrix data, std::size_t k,
                          bool parallelize = false, std::size_t block_size = 64,
                          std::size_t grain_size = 1, bool verbose = false)
    -> List {
  std::size_t n_points = data.nrow();
  std::size_t ndim = data.ncol();

  data = transpose(data);
  auto data_vec = as<std::vector<typename Distance::Input>>(data);

  Distance distance(data_vec, ndim);
  SimpleNeighborHeap neighbor_heap(n_points, k);

  if (parallelize) {
    RPProgress progress(1, verbose);
    nnbf_parallel(neighbor_heap, distance, progress, block_size, grain_size);
  } else {
    RPProgress progress(n_points, verbose);
    nnbf(neighbor_heap, distance, progress);
  }

  return heap_to_r(neighbor_heap);
}

template <typename Distance>
auto rnn_brute_force_query_impl(NumericMatrix x, NumericMatrix y, std::size_t k,
                                bool parallelize = false,
                                std::size_t block_size = 64,
                                std::size_t grain_size = 1,
                                bool verbose = false) -> List {
  std::size_t n_xpoints = x.nrow();
  std::size_t n_ypoints = y.nrow();
  std::size_t ndim = x.ncol();

  x = transpose(x);
  auto x_vec = as<std::vector<typename Distance::Input>>(x);

  y = transpose(y);
  auto y_vec = as<std::vector<typename Distance::Input>>(y);

  Distance distance(x_vec, y_vec, ndim);
  SimpleNeighborHeap neighbor_heap(n_ypoints, k);

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
List rnn_brute_force(NumericMatrix data, int k,
                     const std::string &metric = "euclidean",
                     bool parallelize = false, std::size_t block_size = 64,
                     std::size_t grain_size = 1, bool verbose = false){
    DISPATCH_ON_DISTANCES(BRUTE_FORCE_BUILD)}

// [[Rcpp::export]]
List
    rnn_brute_force_query(NumericMatrix x, NumericMatrix y, int k,
                          const std::string &metric = "euclidean",
                          bool parallelize = false, std::size_t block_size = 64,
                          std::size_t grain_size = 1, bool verbose = false) {
  DISPATCH_ON_QUERY_DISTANCES(BRUTE_FORCE_QUERY)
}

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

#include "bruteforce.h"
#include "distance.h"
#include "heap.h"
#include "rnn.h"
#include "rnn_bruteforceparallel.h"
#include <Rcpp.h>

#define BruteForce(Distance)                                                   \
  return rnn_brute_force_impl<Distance>(data, k, parallelize, grain_size,      \
                                        verbose);

#define BruteForceQuery(Distance)                                              \
  return rnn_brute_force_query_impl<Distance>(x, y, k, parallelize,            \
                                              grain_size, verbose);

template <typename Distance>
Rcpp::List
rnn_brute_force_impl(Rcpp::NumericMatrix data, int k, bool parallelize = false,
                     std::size_t grain_size = 1, bool verbose = false) {
  const std::size_t n_points = data.nrow();
  const std::size_t n_nbrs = k;

  const std::size_t ndim = data.ncol();
  data = Rcpp::transpose(data);
  auto data_vec = Rcpp::as<std::vector<typename Distance::in_type>>(data);

  RPProgress progress(n_points, verbose);
  Distance distance(data_vec, ndim);
  SimpleNeighborHeap neighbor_heap(n_points, n_nbrs);

  if (parallelize) {
    nnbf_parallel(neighbor_heap, distance, progress, grain_size);
  } else {
    nnbf(neighbor_heap, distance, progress);
  }

  return heap_to_r(neighbor_heap);
}

// [[Rcpp::export]]
Rcpp::List rnn_brute_force(Rcpp::NumericMatrix data, int k,
                           const std::string &metric = "euclidean",
                           bool parallelize = false, std::size_t grain_size = 1,
                           bool verbose = false) {
  if (metric == "euclidean") {
    using Distance = Euclidean<float, float>;
    BruteForce(Distance)
  } else if (metric == "l2") {
    using Distance = L2<float, float>;
    BruteForce(Distance)
  } else if (metric == "cosine") {
    using Distance = Cosine<float, float>;
    BruteForce(Distance)
  } else if (metric == "manhattan") {
    using Distance = Manhattan<float, float>;
    BruteForce(Distance)
  } else if (metric == "hamming") {
    using Distance = Hamming<uint8_t, std::size_t>;
    BruteForce(Distance)
  } else {
    Rcpp::stop("Bad metric");
  }
}

template <typename Distance>
Rcpp::List
rnn_brute_force_query_impl(Rcpp::NumericMatrix x, Rcpp::NumericMatrix y, int k,
                           bool parallelize = false, std::size_t grain_size = 1,
                           bool verbose = false) {
  const std::size_t n_xpoints = x.nrow();
  const std::size_t n_ypoints = y.nrow();
  const std::size_t n_nbrs = k;

  const std::size_t ndim = x.ncol();
  x = Rcpp::transpose(x);
  auto x_vec = Rcpp::as<std::vector<typename Distance::in_type>>(x);

  y = Rcpp::transpose(y);
  auto y_vec = Rcpp::as<std::vector<typename Distance::in_type>>(y);

  RPProgress progress(n_xpoints, verbose);
  Distance distance(x_vec, y_vec, ndim);
  SimpleNeighborHeap neighbor_heap(n_ypoints, n_nbrs);

  if (parallelize) {
    nnbf_query(neighbor_heap, distance, n_xpoints, progress);
  } else {
    nnbf_query(neighbor_heap, distance, n_xpoints, progress);
  }

  return heap_to_r(neighbor_heap);
}

// [[Rcpp::export]]
Rcpp::List rnn_brute_force_query(Rcpp::NumericMatrix x, Rcpp::NumericMatrix y,
                                 int k, const std::string &metric = "euclidean",
                                 bool parallelize = false,
                                 std::size_t grain_size = 1,
                                 bool verbose = false) {
  if (metric == "euclidean") {
    using Distance = Euclidean<float, float>;
    BruteForceQuery(Distance)
  } else if (metric == "l2") {
    using Distance = L2<float, float>;
    BruteForceQuery(Distance)
  } else if (metric == "cosine") {
    using Distance = Cosine<float, float>;
    BruteForceQuery(Distance)
  } else if (metric == "manhattan") {
    using Distance = Manhattan<float, float>;
    BruteForceQuery(Distance)
  } else if (metric == "hamming") {
    using Distance = Hamming<uint8_t, std::size_t>;
    BruteForceQuery(Distance)
  } else {
    Rcpp::stop("Bad metric");
  }
}

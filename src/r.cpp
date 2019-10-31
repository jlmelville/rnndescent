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

#include "arrayheap.h"
#include "distance.h"
#include "nndescent.h"
#include "rnn.h"
#include "rnnd_parallel.h"
#include "setheap.h"

#define NNDS(DistType, RandType, use_set, parallelize)            \
if (use_set) {                                                    \
  return nn_descent_impl<SetHeap,                                 \
                         DistType,                                \
                         RandType>                               \
  (data, idx, dist, max_candidates, n_iters, delta, rho, false,   \
   grain_size, verbose);                                          \
}                                                                 \
else {                                                            \
  return nn_descent_impl<ArrayHeap,                               \
                         DistType,                                \
                         RandType>                                \
  (data, idx, dist, max_candidates, n_iters, delta, rho,          \
   parallelize, grain_size, verbose);                             \
}

#define NNDR(DistType, use_set, use_fast_rand, parallelize)       \
if (use_fast_rand) {                                              \
  NNDS(DistType, TauRand, use_set, parallelize)                   \
}                                                                 \
else {                                                            \
  NNDS(DistType, RRand, use_set, parallelize)                     \
}

template <template<typename> class Heap,
          typename Distance,
          typename Rand>
Rcpp::List nn_descent_impl(
    Rcpp::NumericMatrix data,
    Rcpp::IntegerMatrix idx,
    Rcpp::NumericMatrix dist,
    const std::size_t max_candidates = 50,
    const std::size_t n_iters = 10,
    const double delta = 0.001,
    const double rho = 0.5,
    bool parallelize = false,
    std::size_t grain_size = 1,
    bool verbose = false) {
  const std::size_t npoints = idx.nrow();
  const std::size_t nnbrs = idx.ncol();

  const std::size_t ndim = data.ncol();
  data = Rcpp::transpose(data);
  auto data_vec = Rcpp::as<std::vector<typename Distance::in_type>>(data);

  RProgress progress(n_iters, verbose);
  Rand rand;
  Distance distance(data_vec, ndim);
  Heap<Distance> heap = r_to_heap<Heap, Distance>(distance, idx, dist);

  const double tol = delta * nnbrs * npoints;
  if (parallelize) {
    nnd_parallel(heap, max_candidates, n_iters, rand, progress, rho, tol,
                 grain_size);
  }
  else {
    nnd_full(heap, max_candidates, n_iters, rand, progress, rho, tol);
  }

  return heap_to_r(heap.neighbor_heap);
}

// [[Rcpp::export]]
Rcpp::List nn_descent(
    Rcpp::NumericMatrix data,
    Rcpp::IntegerMatrix idx,
    Rcpp::NumericMatrix dist,
    const std::string metric = "euclidean",
    const std::size_t max_candidates = 50,
    const std::size_t n_iters = 10,
    const double delta = 0.001,
    const double rho = 0.5,
    bool use_set = false,
    bool fast_rand = false,
    bool parallelize = false,
    std::size_t grain_size = 1,
    bool verbose = false) {

  if (metric == "euclidean") {
    using DistType = Euclidean<float, float>;
    NNDR(DistType, use_set, fast_rand, parallelize)
  }
  else if (metric == "l2") {
    using DistType = L2<float, float>;
    NNDR(DistType, use_set, fast_rand, parallelize)
  }
  else if (metric == "cosine") {
    using DistType = Cosine<float, float>;
    NNDR(DistType, use_set, fast_rand, parallelize)
  }
  else if (metric == "manhattan") {
    using DistType = Manhattan<float, float>;
    NNDR(DistType, use_set, fast_rand, parallelize)
  }
  else if (metric == "hamming") {
    using DistType = Hamming<uint8_t, std::size_t>;
    NNDR(DistType, use_set, fast_rand, parallelize)
  }
  else {
    Rcpp::stop("Bad metric: " + metric);
  }
}

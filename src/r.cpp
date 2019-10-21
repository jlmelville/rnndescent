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

#include "distance.h"
#include "heap.h"
#include "nndescent.h"
#include "rnnd_parallel.h"
#include "setheap.h"
#include "tauprng.h"


#define NNDS(DistType, RandType, use_set, parallelize)            \
if (use_set) {                                                    \
  return nn_descent_impl<SetHeap,                                 \
                         DistType,                                \
                         RandType,                                \
                         RProgress>                               \
  (data, idx, dist, max_candidates, n_iters, delta, rho, false, verbose);\
}                                                                 \
else {                                                            \
  return nn_descent_impl<ArrayHeap,                               \
                         DistType,                                \
                         RandType,                                \
                         RProgress>                               \
  (data, idx, dist, max_candidates, n_iters, delta, rho, parallelize, verbose);\
}

#define NNDR(DistType, use_set, use_fast_rand, parallelize)    \
if (use_fast_rand) {                                           \
  NNDS(DistType, TauRand, use_set, parallelize)                \
}                                                              \
else {                                                         \
  NNDS(DistType, RRand, use_set, parallelize)                  \
}

struct RRand {
  // a random uniform value between 0 and 1
  double unif() {
    return Rcpp::runif(1, 0.0, 1.0)[0];
  }
};

struct RProgress {
  template <typename Heap>
  void iter(std::size_t n, std::size_t n_iters, const Heap& heap) {
    double sum = 0.0;
    for (std::size_t i = 0; i < heap.n_points; i++) {
      for (std::size_t j = 0; j < heap.n_nbrs; j++) {
        sum += heap.distance(i, j);
      }
    }
    Rcpp::Rcout << (n + 1) << " / " << n_iters << " " << sum << std::endl;
  }
  void converged(const std::size_t c, const double tol) {
    Rcpp::Rcout << "c = " << c << " tol = " << tol << std::endl;
  }
  void check_interrupt() {
    Rcpp::checkUserInterrupt();
  }
  template<typename PHeap>
  void report(PHeap& heap)
  {
    Rcpp::Rcout << heap.report() << std::endl;
  }
};

template <template<typename> class Heap, typename Distance>
Heap<Distance> r_to_heap(
    Distance& distance,
    Rcpp::IntegerMatrix idx,
    Rcpp::NumericMatrix dist
) {
  const std::size_t npoints = idx.nrow();
  const std::size_t nnbrs = idx.ncol();

  Heap<Distance> heap(distance, npoints, nnbrs);
  const int max_idx = npoints - 1; // internally we need to be 0-indexed
  for (std::size_t i = 0; i < npoints; i++) {
    for (std::size_t j = 0; j < nnbrs; j++) {
      const int k = idx(i, j);
      if (k < 0 || k > max_idx) {
        Rcpp::stop("Bad indexes in input");
      }
      heap.add_pair(i, k, true);
    }
  }

  return heap;
}

// transfer data into R Matrices
Rcpp::List heap_to_r(const NeighborHeap& heap) {
  const std::size_t npoints = heap.n_points;
  const std::size_t nnbrs = heap.n_nbrs;

  Rcpp::IntegerMatrix idxres(npoints, nnbrs);
  Rcpp::NumericMatrix distres(npoints, nnbrs);
  for (std::size_t i = 0; i < npoints; i++) {
    for (std::size_t j = 0; j < nnbrs; j++) {
      idxres(i, j) = heap.index(i, j) + 1;
      distres(i, j) = heap.distance(i, j);
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("idx") = idxres,
    Rcpp::Named("dist") = distres
  );
}

template <template<typename> class Heap,
          typename Distance,
          typename Rand,
          typename Progress>
Rcpp::List nn_descent_impl(
    Rcpp::NumericMatrix data,
    Rcpp::IntegerMatrix idx,
    Rcpp::NumericMatrix dist,
    const std::size_t max_candidates = 50,
    const std::size_t n_iters = 10,
    const double delta = 0.001,
    const double rho = 0.5,
    bool parallelize = false,
    bool verbose = false) {
  const std::size_t npoints = idx.nrow();
  const std::size_t nnbrs = idx.ncol();

  const std::size_t ndim = data.ncol();
  data = Rcpp::transpose(data);
  auto data_vec = Rcpp::as<std::vector<typename Distance::in_type>>(data);

  Progress progress;
  Rand rand;
  Distance distance(data_vec, ndim);
  Heap<Distance> heap = r_to_heap<Heap, Distance>(distance, idx, dist);


  const double tol = delta * nnbrs * npoints;
  if (parallelize) {
    // FIXME: add grain_size param
    nnd_parallel(heap, max_candidates, n_iters, rand, progress, rho, tol, 1, verbose);
  }
  else {
    nnd_full(heap, max_candidates, n_iters, rand, progress, rho, tol, verbose);
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
    bool verbose = false) {

  if (metric == "euclidean") {
    using dist_type = Euclidean<float, float>;
    NNDR(dist_type, use_set, fast_rand, parallelize)
  }
  else if (metric == "l2") {
    using dist_type = L2<float, float>;
    NNDR(dist_type, use_set, fast_rand, parallelize)
  }
  else if (metric == "cosine") {
    using dist_type = Cosine<float, float>;
    NNDR(dist_type, use_set, fast_rand, parallelize)
  }
  else if (metric == "manhattan") {
    using dist_type = Manhattan<float, float>;
    NNDR(dist_type, use_set, fast_rand, parallelize)
  }
  else if (metric == "hamming") {
    using dist_type = Hamming<uint8_t, std::size_t>;
    NNDR(dist_type, use_set, fast_rand, parallelize)
  }
  else {
    Rcpp::stop("Bad metric");
  }
}




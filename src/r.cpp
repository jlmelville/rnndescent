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


struct RRand {
  // a random uniform value between 0 and 1
  double unif() {
    return Rcpp::runif(1, 0.0, 1.0)[0];
  }
};

struct RProgress {
  void iter(std::size_t n, std::size_t n_iters, const Heap& heap) {
    double sum = 0.0;
    for (std::size_t i = 0; i < heap.dist.size(); i++) {
      for (std::size_t j = 0; j < heap.dist[i].size(); j++) {
        sum += heap.dist[i][j];
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
};


Heap r_to_heap(
    Rcpp::IntegerMatrix idx,
    Rcpp::NumericMatrix dist
) {
  const std::size_t npoints = idx.nrow();
  const std::size_t nnbrs = idx.ncol();

  Heap heap(npoints, nnbrs);
  const int max_idx = npoints - 1; // internally we need to be 0-indexed
  for (std::size_t i = 0; i < npoints; i++) {
    for (std::size_t j = 0; j < nnbrs; j++) {
      const int k = idx(i, j);
      if (k < 0 || k > max_idx) {
        Rcpp::stop("Bad indexes in input");
      }
      const double d = dist(i, j);
      heap.push(i, d, k, true);
      heap.push(k, d, i, true);
    }
  }

  return heap;
}

// transfer data into R Matrices
Rcpp::List heap_to_r(const Heap& heap) {
  const std::size_t npoints = heap.idx.size();
  const std::size_t nnbrs = heap.idx[0].size();

  Rcpp::IntegerMatrix idxres(npoints, nnbrs);
  Rcpp::NumericMatrix distres(npoints, nnbrs);
  for (std::size_t i = 0; i < npoints; i++) {
    for (std::size_t j = 0; j < nnbrs; j++) {
      idxres(i, j) = heap.idx[i][j];
      distres(i, j) = heap.dist[i][j];
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("idx") = idxres,
    Rcpp::Named("dist") = distres
  );
}

template <typename Distance,
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
    bool verbose = false) {
  const std::size_t npoints = idx.nrow();
  const std::size_t nnbrs = idx.ncol();

  Heap heap = r_to_heap(idx, dist);

  const std::size_t ndim = data.ncol();
  data = Rcpp::transpose(data);
  auto data_vec = Rcpp::as<std::vector<typename Distance::in_type>>(data);

  Progress progress;
  Rand rand;
  Distance distance(data_vec, ndim);
  const double tol = delta * nnbrs * npoints;

  nnd(heap, max_candidates, n_iters, npoints, nnbrs, distance, rand, progress,
      rho, tol, verbose);

  heap.deheap_sort();

  return heap_to_r(heap);
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
    bool verbose = false) {

  if (metric == "euclidean") {
    return nn_descent_impl<Euclidean<float, float>,
                           RRand,
                           RProgress>
    (data, idx, dist, max_candidates, n_iters, delta, rho, verbose);
  }
  else if (metric == "l2") {
    return nn_descent_impl<L2<float, float>,
                           RRand,
                           RProgress>
    (data, idx, dist, max_candidates, n_iters, delta, rho, verbose);
  }
  else if (metric == "cosine") {
    return nn_descent_impl<Cosine<float, float>,
                           RRand,
                           RProgress>
    (data, idx, dist, max_candidates, n_iters, delta, rho, verbose);
  }
  else if (metric == "manhattan") {
    return nn_descent_impl<Manhattan<float, float>,
                           RRand,
                           RProgress>
    (data, idx, dist, max_candidates, n_iters, delta, rho, verbose);
  }
  else if (metric == "hamming") {
    return nn_descent_impl<Hamming<uint8_t, std::size_t>,
                           RRand,
                           RProgress>
    (data, idx, dist, max_candidates, n_iters, delta, rho, verbose);
  }
  else {
    Rcpp::stop("Bad metric");
  }
}

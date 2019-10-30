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

#ifndef RNND_RNN_H
#define RNND_RNN_H

#include <Rcpp.h>

#include "heap.h"


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

template <typename NbrHeap>
Rcpp::List heap_to_r(const NbrHeap& heap)
{
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

#endif // RNND_RNN_H

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

#include <time.h>

#include <Rcpp.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include "heap.h"

inline std::string time_unit(int u)
{
  std::string ustr(std::to_string(u));
  return u < 10 ? "0" + ustr : ustr;
}

inline void print_time(bool print_date = false) {
  std::time_t current_time;
  struct tm now;

  time(&current_time);
  localtime_s(&now, &current_time);
  if (print_date) {
    Rcpp::Rcout << (now.tm_year + 1900) << '-'
                << time_unit(now.tm_mon + 1) << ":" << '-'
                << time_unit(now.tm_mday) << " ";
  }
  Rcpp::Rcout << time_unit(now.tm_hour) << ":"
              << time_unit(now.tm_min) << ":"
              << time_unit(now.tm_sec) << " ";
}

inline void ts(const std::string& msg) {
  print_time();
  Rcpp::Rcout << msg << std::endl;
}

struct RRand {
  // a random uniform value between 0 and 1
  double unif() {
    return Rcpp::runif(1, 0.0, 1.0)[0];
  }
};

// Sums the distances in a neighbor heap as a way of measuring progress.
// Useful for diagnostic purposes
struct HeapSumProgress {
  NeighborHeap& neighbor_heap;
  const std::size_t n_iters;
  bool verbose;

  HeapSumProgress(
    NeighborHeap& neighbor_heap,
    std::size_t n_iters,
    bool verbose
    ) :
    neighbor_heap(neighbor_heap),
    n_iters(n_iters),
    verbose(verbose)
  {}

  void update(std::size_t n) {
    if (verbose) {
      const std::size_t n_points = neighbor_heap.n_points;
      const std::size_t n_nbrs = neighbor_heap.n_nbrs;
      double sum = 0.0;
      for (std::size_t i = 0; i < n_points; i++) {
        const std::size_t innbrs = i * n_nbrs;
        for (std::size_t j = 0; j < n_nbrs; j++) {
          sum += neighbor_heap.distance(innbrs + j);
        }
      }

      std::ostringstream os;
      os << (n + 1) << " / " << n_iters << " " << sum;
      ts(os.str());
    }
  }
  void stopping_early() {
  }
  bool check_interrupt() {
    try {
      Rcpp::checkUserInterrupt();
    }
    catch (Rcpp::internal::InterruptedException&) {
      return true;
    }
    return false;
  }
};

struct RPProgress {

  Progress progress;
  const std::size_t n_iters;
  bool verbose;

  RPProgress(
    std::size_t n_iters,
    bool verbose) :
    progress(n_iters, verbose),
    n_iters(n_iters),
    verbose(verbose)
  {}

  void increment(std::size_t amount = 1) {
    progress.increment(amount);
  }
  void update(std::size_t current) {
    progress.update(current);
  }
  void stopping_early() {
    progress.update(n_iters);
  }
  bool check_interrupt() {
    if (Progress::check_abort()) {
      progress.cleanup();
      return true;
    }
    return false;
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

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

#include <limits>

#include <Rcpp.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include "tdoann/graphupdate.h"
#include "tdoann/heap.h"

void print_time(bool print_date = false);
void ts(const std::string &msg);

// Sums the distances in a neighbor heap as a way of measuring progress.
// Useful for diagnostic purposes
struct HeapSumProgress {
  tdoann::NeighborHeap &neighbor_heap;
  const std::size_t n_iters;
  bool verbose;

  HeapSumProgress(tdoann::NeighborHeap &neighbor_heap, std::size_t n_iters,
                  bool verbose);
  void increment(std::size_t amount = 1);
  void iter(std::size_t n);
  double dist_sum() const;
  void stopping_early();
  bool check_interrupt();
  void converged(std::size_t n_updates, double tol);
};

struct RPProgress {
  Progress progress;
  const std::size_t n_iters;
  bool verbose;

  RPProgress(std::size_t n_iters, bool verbose);
  void increment(std::size_t amount = 1);
  void update(std::size_t current);
  void iter(std::size_t iter);
  void stopping_early();
  bool check_interrupt();
};

struct HeapAddSymmetric {
  template <typename NbrHeap>
  static void push(NbrHeap &current_graph, std::size_t ref, std::size_t query,
                   double d) {
    current_graph.checked_push_pair(ref, d, query);
  }
};

struct HeapAddQuery {
  template <typename NbrHeap>
  static void push(NbrHeap &current_graph, std::size_t ref, std::size_t query,
                   double d) {
    current_graph.checked_push(ref, d, query);
  }
};

// input idx R matrix is 1-indexed and transposed
// output heap index is 0-indexed
template <typename HeapAdd, typename NbrHeap>
void r_to_heap(NbrHeap &current_graph, Rcpp::IntegerMatrix idx,
               Rcpp::NumericMatrix dist,
               const int max_idx = std::numeric_limits<int>::max()) {
  const std::size_t n_points = idx.nrow();
  const std::size_t n_nbrs = idx.ncol();

  for (std::size_t i = 0; i < n_points; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      const int k = idx(i, j) - 1;
      if (k < 0 || k > max_idx) {
        Rcpp::stop("Bad indexes in input");
      }
      double d = dist(i, j);
      HeapAdd::push(current_graph, i, k, d);
    }
  }
}

// input heap index is 0-indexed
// output idx R matrix is 1-indexed and untransposed
template <typename NbrHeap>
void heap_to_r(const NbrHeap &heap, Rcpp::IntegerMatrix idx,
               Rcpp::NumericMatrix dist) {
  const std::size_t n_points = heap.n_points;
  const std::size_t n_nbrs = heap.n_nbrs;

  for (std::size_t i = 0; i < n_points; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      idx(i, j) = heap.index(i, j) + 1;
      dist(i, j) = heap.distance(i, j);
    }
  }
}

// input heap index is 0-indexed
// output idx R matrix is 1-indexed and transposed
template <typename NbrHeap> Rcpp::List heap_to_r(const NbrHeap &heap) {
  const std::size_t n_points = heap.n_points;
  const std::size_t n_nbrs = heap.n_nbrs;

  Rcpp::IntegerMatrix idx(n_points, n_nbrs);
  Rcpp::NumericMatrix dist(n_points, n_nbrs);

  heap_to_r(heap, idx, dist);

  return Rcpp::List::create(Rcpp::Named("idx") = idx,
                            Rcpp::Named("dist") = dist);
}

#endif // RNND_RNN_H

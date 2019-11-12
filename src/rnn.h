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
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include "tdoann/graphupdate.h"
#include "tdoann/heap.h"

std::string time_unit(int u);
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
  void update(std::size_t n);
  double dist_sum() const;
  void stopping_early();
  bool check_interrupt();
};

struct RPProgress {
  Progress progress;
  const std::size_t n_iters;
  bool verbose;

  RPProgress(std::size_t n_iters, bool verbose);
  void increment(std::size_t amount = 1);
  void update(std::size_t current);
  void stopping_early();
  bool check_interrupt();
};

template <template <typename> class GraphUpdater, typename Distance>
void r_to_heap(tdoann::NeighborHeap &current_graph, Distance &distance,
               Rcpp::IntegerMatrix idx, Rcpp::NumericMatrix dist,
               const int max_idx) {
  GraphUpdater<Distance> heap_initializer(current_graph, distance);

  const std::size_t n_points = idx.nrow();
  const std::size_t n_nbrs = idx.ncol();

  for (std::size_t i = 0; i < n_points; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      const int k = idx(i, j);
      if (k < 0 || k > max_idx) {
        Rcpp::stop("Bad indexes in input");
      }
      heap_initializer.generate_and_apply(i, k);
    }
  }
}

template <typename NbrHeap> Rcpp::List heap_to_r(const NbrHeap &heap) {
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

  return Rcpp::List::create(Rcpp::Named("idx") = idxres,
                            Rcpp::Named("dist") = distres);
}

#endif // RNND_RNN_H

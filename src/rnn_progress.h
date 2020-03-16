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

#ifndef RNN_PROGRESS_H
#define RNN_PROGRESS_H

#include <Rcpp.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include "tdoann/heap.h"
#include "tdoann/progress.h"
#include "tdoann/typedefs.h"

// Sums the distances in a neighbor heap as a way of measuring progress.
// Useful for diagnostic purposes
struct HeapSumProgress {
  NeighborHeap &neighbor_heap;
  const std::size_t n_iters;
  bool verbose;

  std::size_t iter;
  bool is_aborted;

  HeapSumProgress(NeighborHeap &neighbor_heap, std::size_t n_iters,
                  bool verbose = false);
  void set_n_blocks(std::size_t n_blocks);
  void block_finished() {}
  void iter_finished();
  void stopping_early(){};
  bool check_interrupt();
  void converged(std::size_t n_updates, double tol);
  double dist_sum() const;
  void iter_msg(std::size_t iter) const;
};

struct RPProgress {
  const std::size_t scale;
  Progress progress;
  const std::size_t n_iters;
  std::size_t n_blocks_;
  bool verbose;

  std::size_t iter;
  std::size_t block;
  bool is_aborted;

  RPProgress(std::size_t n_iters, bool verbose);
  RPProgress(NeighborHeap &, std::size_t n_iters, bool verbose);
  void set_n_blocks(std::size_t n_blocks);
  void block_finished();
  void iter_finished();
  void stopping_early();
  bool check_interrupt();
  void converged(std::size_t n_updates, double tol);
  // convert float between 0...n_iters to int from 0...scale
  int scaled(double d);
};

struct RInterruptableProgress {
  bool is_aborted;

  RInterruptableProgress();
  RInterruptableProgress(std::size_t, bool);
  void set_n_blocks(std::size_t n_blocks) {}
  void block_finished() {}
  void iter_finished() {}
  void stopping_early() {}
  bool check_interrupt();
};

#endif // RNN_PROGRESS_H

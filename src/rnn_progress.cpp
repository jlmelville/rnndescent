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
#include <progress.hpp>

#include "tdoann/typedefs.h"

#include "rnn_progress.h"
#include "rnn_util.h"

HeapSumProgress::HeapSumProgress(NeighborHeap &neighbor_heap,
                                 std::size_t n_iters, bool verbose)
    : neighbor_heap(neighbor_heap), n_iters(n_iters), verbose(verbose), iter(0),
      is_aborted(false) {
  iter_msg(0);
}

void HeapSumProgress::iter_msg(std::size_t iter) const {
  if (verbose) {
    std::ostringstream os;
    os << iter << " / " << n_iters << " " << dist_sum();
    ts(os.str());
  }
}
void HeapSumProgress::set_n_blocks(std::size_t) {}
void HeapSumProgress::iter_finished() {
  ++iter;
  iter_msg(iter);
}
bool HeapSumProgress::check_interrupt() {
  if (is_aborted) {
    return true;
  }
  try {
    Rcpp::checkUserInterrupt();
  } catch (Rcpp::internal::InterruptedException &) {
    is_aborted = true;
    stopping_early();
    return true;
  }
  return false;
}
void HeapSumProgress::converged(std::size_t n_updates, double tol) {
  if (verbose) {
    Rcpp::Rcerr << "Convergence: c = " << n_updates << " tol = " << tol
                << std::endl;
  }
  stopping_early();
}
double HeapSumProgress::dist_sum() const {
  std::size_t n_points = neighbor_heap.n_points;
  std::size_t n_nbrs = neighbor_heap.n_nbrs;
  double sum = 0.0;
  for (std::size_t i = 0; i < n_points; i++) {
    std::size_t innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      sum += neighbor_heap.dist[innbrs + j];
    }
  }
  return sum;
}
RPProgress::RPProgress(std::size_t n_iters, bool verbose)
    : scale(100), progress(scale, verbose), n_iters(n_iters), n_blocks_(0),
      verbose(verbose), iter(0), block(0), is_aborted(false) {}
RPProgress::RPProgress(NeighborHeap &, std::size_t n_iters, bool verbose)
    : scale(100), progress(scale, verbose), n_iters(n_iters), n_blocks_(0),
      verbose(verbose), iter(0), block(0), is_aborted(false) {}

void RPProgress::set_n_blocks(std::size_t n_blocks) {
  n_blocks_ = n_blocks;
  block = 0;
}
void RPProgress::block_finished() {
  ++block;
  if (verbose) {
    progress.update(scaled(iter + (static_cast<double>(block) / n_blocks_)));
  }
}
void RPProgress::iter_finished() {
  if (verbose) {
    ++iter;
    progress.update(scaled(iter));
  }
}
void RPProgress::stopping_early() {
  progress.update(n_iters);
  progress.cleanup();
}
bool RPProgress::check_interrupt() {
  if (is_aborted) {
    return true;
  }
  if (Progress::check_abort()) {
    stopping_early();
    is_aborted = true;
    return true;
  }
  return false;
}
void RPProgress::converged(std::size_t n_updates, double tol) {
  stopping_early();
  if (verbose) {
    Rcpp::Rcerr << "Convergence at iteration " << iter << ": c = " << n_updates
                << " tol = " << tol << std::endl;
  }
}
int RPProgress::scaled(double d) {
  int res = std::nearbyint(scale * (d / n_iters));
  return res;
}

RInterruptableProgress::RInterruptableProgress() : is_aborted(false) {}
RInterruptableProgress::RInterruptableProgress(std::size_t, bool)
    : is_aborted(false) {}
bool RInterruptableProgress::check_interrupt() {
  if (is_aborted) {
    return true;
  }
  try {
    Rcpp::checkUserInterrupt();
  } catch (Rcpp::internal::InterruptedException &) {
    is_aborted = true;
    stopping_early();
    return true;
  }
  return false;
}

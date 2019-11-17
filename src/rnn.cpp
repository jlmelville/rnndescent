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

#include <chrono>
#include <cmath>

#include <Rcpp.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include "tdoann/heap.h"

#include "rnn.h"

using namespace tdoann;

void print_time(bool print_date) {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto secs =
      std::chrono::duration_cast<std::chrono::seconds>(duration).count();

  std::string fmt = print_date ? "%Y-%m-%d %H:%M:%S" : "%H:%M:%S";
  Rcpp::Datetime dt(secs);
  std::string dt_str = dt.format(fmt.c_str());
  // for some reason format always adds ".000000", so remove it
  if (dt_str.size() >= 7) {
    dt_str = dt_str.substr(0, dt_str.size() - 7);
  }
  Rcpp::Rcout << dt_str << " ";
}

void ts(const std::string &msg) {
  print_time();
  Rcpp::Rcout << msg << std::endl;
}

HeapSumProgress::HeapSumProgress(NeighborHeap &neighbor_heap,
                                 std::size_t n_iters, bool verbose)
    : neighbor_heap(neighbor_heap), n_iters(n_iters), iter(0), verbose(verbose),
      is_aborted(false) {
  if (verbose) {
    std::ostringstream os;
    os << "0 / " << n_iters << " " << dist_sum();
    ts(os.str());
  }
}
void HeapSumProgress::block_finished() {}
void HeapSumProgress::iter_finished() {
  ++iter;
  if (verbose) {
    std::ostringstream os;
    os << iter << " / " << n_iters << " " << dist_sum();
    ts(os.str());
  }
}
void HeapSumProgress::stopping_early() {}
bool HeapSumProgress::check_interrupt() {
  if (is_aborted) {
    return true;
  }
  try {
    Rcpp::checkUserInterrupt();
  } catch (Rcpp::internal::InterruptedException &) {
    is_aborted = true;
    return true;
  }
  return false;
}
void HeapSumProgress::converged(std::size_t n_updates, double tol) {
  if (verbose) {
    Rcpp::Rcout << "c = " << n_updates << " tol = " << tol << std::endl;
  }
}
double HeapSumProgress::dist_sum() const {
  const std::size_t n_points = neighbor_heap.n_points;
  const std::size_t n_nbrs = neighbor_heap.n_nbrs;
  double sum = 0.0;
  for (std::size_t i = 0; i < n_points; i++) {
    const std::size_t innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      sum += neighbor_heap.dist[innbrs + j];
    }
  }
  return sum;
}
RPProgress::RPProgress(std::size_t n_iters, std::size_t n_blocks, bool verbose)
    : scale(100), progress(scale, verbose), n_iters(n_iters),
      n_blocks(n_blocks), verbose(verbose), iter(0), block(0),
      is_aborted(false) {}
RPProgress::RPProgress(std::size_t n_iters, bool verbose)
    : scale(100), progress(scale, verbose), n_iters(n_iters), n_blocks(0),
      verbose(verbose), iter(0), block(0), is_aborted(false) {}
void RPProgress::block_finished() {
  if (verbose) {
    ++block;
    progress.update(scaled(iter + (block / n_blocks)));
  }
}
void RPProgress::iter_finished() {
  if (verbose) {
    block = 0;
    ++iter;
    progress.update(scaled(iter));
  }
}
void RPProgress::stopping_early() { progress.update(n_iters); }
bool RPProgress::check_interrupt() {
  if (is_aborted || Progress::check_abort()) {
    progress.cleanup();
    is_aborted = true;
    return true;
  }
  return false;
}
void RPProgress::converged(std::size_t n_updates, double tol) {}
int RPProgress::scaled(double d) {
  int res = std::nearbyint(scale * (d / n_iters));
  return res;
}

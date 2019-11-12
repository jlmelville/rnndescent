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

#include <Rcpp.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include "tdoann/heap.h"

#include "rnn.h"

using namespace tdoann;

std::string time_unit(int u) {
  std::string ustr(std::to_string(u));
  return u < 10 ? "0" + ustr : ustr;
}

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
    : neighbor_heap(neighbor_heap), n_iters(n_iters), verbose(verbose) {}
void HeapSumProgress::update(std::size_t n) {
  if (verbose) {
    std::ostringstream os;
    os << (n + 1) << " / " << n_iters << " " << dist_sum();
    ts(os.str());
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
void HeapSumProgress::stopping_early() {}
bool HeapSumProgress::check_interrupt() {
  try {
    Rcpp::checkUserInterrupt();
  } catch (Rcpp::internal::InterruptedException &) {
    return true;
  }
  return false;
}

RPProgress::RPProgress(std::size_t n_iters, bool verbose)
    : progress(n_iters, verbose), n_iters(n_iters), verbose(verbose) {}
void RPProgress::increment(std::size_t amount) { progress.increment(amount); }
void RPProgress::update(std::size_t current) { progress.update(current); }
void RPProgress::stopping_early() { progress.update(n_iters); }
bool RPProgress::check_interrupt() {
  if (Progress::check_abort()) {
    progress.cleanup();
    return true;
  }
  return false;
}

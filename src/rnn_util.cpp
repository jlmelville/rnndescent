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
#include <thread>

#include <Rcpp.h>
#include <progress.hpp>

#include "rnn_distance.h"
#include "rnn_parallel.h"
#include "rnn_util.h"

using namespace Rcpp;

void print_time(bool print_date) {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto secs =
      std::chrono::duration_cast<std::chrono::seconds>(duration).count();

  std::string fmt = print_date ? "%Y-%m-%d %H:%M:%S" : "%H:%M:%S";
  Datetime dt(secs);
  std::string dt_str = dt.format(fmt.c_str());
  // for some reason format always adds ".000000", so remove it
  if (dt_str.size() >= 7) {
    dt_str = dt_str.substr(0, dt_str.size() - 7);
  }
  Rcerr << dt_str << " ";
}

void ts(const std::string &msg) {
  print_time();
  Rcerr << msg << std::endl;
}

void zero_index(IntegerMatrix m, int max_idx, bool missing_ok) {
  const int min_idx = missing_ok ? -1 : 0;
  for (auto j = 0; j < m.ncol(); j++) {
    for (auto i = 0; i < m.nrow(); i++) {
      auto idx0 = m(i, j) - 1;
      if (idx0 < min_idx || idx0 > max_idx) {
        stop("Bad indexes in input: " + std::to_string(idx0));
      }
      m(i, j) = idx0;
    }
  }
}

// [[Rcpp::export]]
List sort_graph(List graph_list, std::size_t n_threads = 0) {
  auto nn_graph = r_to_graph<DummyDistance>(graph_list);
  const std::size_t block_size = 100;
  const std::size_t grain_size = 1;
  tdoann::sort_knn_graph<tdoann::HeapAddQuery, RParallel>(
      nn_graph, block_size, n_threads, grain_size);
  // unzero = true
  return graph_to_r(nn_graph, true);
}

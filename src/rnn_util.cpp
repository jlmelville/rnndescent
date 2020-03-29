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

#include "tdoann/heap.h"
#include "tdoann/nngraph.h"

#include "rnn_util.h"

using namespace tdoann;
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

void zero_index(IntegerMatrix m, int max_idx) {
  for (auto j = 0; j < m.ncol(); j++) {
    for (auto i = 0; i < m.nrow(); i++) {
      auto idx0 = m(i, j) - 1;
      if (idx0 < 0 || idx0 > max_idx) {
        stop("Bad indexes in input: " + std::to_string(idx0));
      }
      m(i, j) = idx0;
    }
  }
}

auto graph_to_r(const NNGraph &graph) -> List {
  IntegerMatrix indices(graph.n_nbrs, graph.n_points, graph.idx.begin());
  NumericMatrix dist(graph.n_nbrs, graph.n_points, graph.dist.begin());

  return List::create(_("idx") = transpose(indices),
                      _("dist") = transpose(dist));
}

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

#include "tdoann/nngraph.h"

#include "rnn_distance.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_rtoheap.h"

using namespace Rcpp;

#define IDX_TO_GRAPH_SELF()                                                    \
  auto distance = tr_to_dist<Distance>(data);                                  \
  return idx_to_graph_impl(distance, idx, n_threads, verbose);

#define IDX_TO_GRAPH_QUERY()                                                   \
  auto distance = tr_to_dist<Distance>(reference, query);                      \
  return idx_to_graph_impl<Distance>(distance, idx, n_threads, verbose);

template <typename Distance>
auto idx_to_graph_impl(const Distance &distance, IntegerMatrix idx,
                       std::size_t n_threads = 0, bool verbose = false)
    -> List {
  auto idx_vec = r_to_idxt<typename Distance::Index>(idx);
  if (n_threads > 0) {
    auto nn_graph = tdoann::idx_to_graph<Distance, RPProgress, RParallel>(
        distance, idx_vec, n_threads, verbose);
    return graph_to_r(nn_graph, true);
  } else {
    auto nn_graph =
        tdoann::idx_to_graph<Distance, RPProgress>(distance, idx_vec, verbose);
    return graph_to_r(nn_graph, true);
  }
}

// [[Rcpp::export]]
List rnn_idx_to_graph_self(NumericMatrix data, IntegerMatrix idx,
                           const std::string &metric = "euclidean",
                           std::size_t n_threads = 0, bool verbose = false){
    DISPATCH_ON_DISTANCES(IDX_TO_GRAPH_SELF)}

// [[Rcpp::export]]
List rnn_idx_to_graph_query(NumericMatrix reference, NumericMatrix query,
                            IntegerMatrix idx,
                            const std::string &metric = "euclidean",
                            std::size_t n_threads = 0, bool verbose = false) {
  DISPATCH_ON_QUERY_DISTANCES(IDX_TO_GRAPH_QUERY)
}

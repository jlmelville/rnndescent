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

#include "tdoann/randnbrs.h"

#include "rnn_distance.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_sample.h"
#include "rnn_util.h"

using namespace Rcpp;

/* Macros */

#define RANDOM_NBRS_BUILD()                                                    \
  return random_build_impl<Distance>(data, k, order_by_distance, n_threads,    \
                                     verbose);

#define RANDOM_NBRS_QUERY()                                                    \
  return random_query_impl<Distance>(reference, query, k, order_by_distance,   \
                                     n_threads, verbose);

/* Functions */

template <typename Distance>
auto random_build_impl(NumericMatrix data, typename Distance::Index k,
                       bool order_by_distance, std::size_t n_threads,
                       bool verbose) -> List {

  auto data_vec = r_to_dist_vect<Distance>(data);

  auto nn_graph =
      tdoann::random_build<Distance, DQIntSampler, RPProgress, RParallel>(
          data_vec, data.ncol(), k, order_by_distance, n_threads, verbose);

  return graph_to_r(nn_graph);
}

template <typename Distance>
auto random_query_impl(NumericMatrix reference, NumericMatrix query,
                       typename Distance::Index k, bool order_by_distance,
                       std::size_t n_threads, bool verbose) -> List {

  auto ref_vec = r_to_dist_vect<Distance>(reference);
  auto query_vec = r_to_dist_vect<Distance>(query);

  auto nn_graph =
      tdoann::random_query<Distance, DQIntSampler, RPProgress, RParallel>(
          ref_vec, reference.ncol(), query_vec, k, order_by_distance, n_threads,
          verbose);

  return graph_to_r(nn_graph);
}

/* Exports */

// [[Rcpp::export]]
List random_knn_cpp(Rcpp::NumericMatrix data, uint32_t k,
                    const std::string &metric = "euclidean",
                    bool order_by_distance = true, std::size_t n_threads = 0,
                    bool verbose = false){
    DISPATCH_ON_DISTANCES(RANDOM_NBRS_BUILD)}

// [[Rcpp::export]]
List random_knn_query_cpp(NumericMatrix reference, NumericMatrix query,
                          uint32_t k, const std::string &metric = "euclidean",
                          bool order_by_distance = true,
                          std::size_t n_threads = 0, bool verbose = false) {
  DISPATCH_ON_QUERY_DISTANCES(RANDOM_NBRS_QUERY)
}

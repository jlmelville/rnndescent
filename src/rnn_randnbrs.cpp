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

// NOLINTBEGIN(modernize-use-trailing-return-type)

#include <Rcpp.h>

#include "rnndescent/random.h"
#include "tdoann/randnbrs.h"

#include "rnn_distance.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

using Rcpp::List;
using Rcpp::NumericMatrix;

/* Macros */

#define RANDOM_NBRS_BUILD()                                                    \
  return random_build_impl<Distance>(data, nnbrs, order_by_distance,           \
                                     n_threads, verbose);

#define RANDOM_NBRS_QUERY()                                                    \
  return random_query_impl<Distance>(reference, query, nnbrs,                  \
                                     order_by_distance, n_threads, verbose);

/* Functions */

template <typename Distance>
auto random_build_impl(NumericMatrix data, typename Distance::Index nnbrs,
                       bool order_by_distance, std::size_t n_threads,
                       bool verbose) -> List {

  auto distance = tr_to_dist<Distance>(data);
  auto progress = std::make_unique<RPProgress>(verbose);

  auto nn_graph = tdoann::random_build<
      Distance, rnndescent::DQIntSampler<typename Distance::Index>, RParallel>(
      distance, nnbrs, order_by_distance, n_threads, *progress);

  return graph_to_r(nn_graph);
}

template <typename Distance>
auto random_query_impl(NumericMatrix reference, NumericMatrix query,
                       typename Distance::Index nnbrs, bool order_by_distance,
                       std::size_t n_threads, bool verbose) -> List {

  auto distance = tr_to_dist<Distance>(reference, query);
  auto progress = std::make_unique<RPProgress>(verbose);

  auto nn_graph = tdoann::random_query<
      Distance, rnndescent::DQIntSampler<typename Distance::Index>, RParallel>(
      distance, nnbrs, order_by_distance, n_threads, *progress);

  return graph_to_r(nn_graph);
}

/* Exports */

// [[Rcpp::export]]
List random_knn_cpp(const NumericMatrix &data, uint32_t nnbrs,
                    const std::string &metric = "euclidean",
                    bool order_by_distance = true, std::size_t n_threads = 0,
                    bool verbose = false){
    DISPATCH_ON_DISTANCES(RANDOM_NBRS_BUILD)}

// [[Rcpp::export]]
List random_knn_query_cpp(const NumericMatrix &reference,
                          const NumericMatrix &query, uint32_t nnbrs,
                          const std::string &metric = "euclidean",
                          bool order_by_distance = true,
                          std::size_t n_threads = 0, bool verbose = false) {
  DISPATCH_ON_QUERY_DISTANCES(RANDOM_NBRS_QUERY)
}

// NOLINTEND(modernize-use-trailing-return-type)

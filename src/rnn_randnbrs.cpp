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
#include "rnn_rng.h"
#include "rnn_sample.h"
#include "rnn_util.h"

using namespace Rcpp;

/* Macros */

#define RANDOM_NBRS_BUILD()                                                    \
  auto distance = create_build_distance<Distance>(data);                       \
  return random_knn_build_impl<Distance>(distance, k, order_by_distance,       \
                                         block_size, verbose, n_threads,       \
                                         grain_size);

#define RANDOM_NBRS_QUERY()                                                    \
  auto distance = create_query_distance<Distance>(reference, query);           \
  return random_knn_query_impl<Distance>(distance, k, order_by_distance,       \
                                         block_size, verbose, n_threads,       \
                                         grain_size);

/* Functions */

template <typename Distance>
auto random_knn_build_impl(Distance &distance, std::size_t k,
                           bool order_by_distance, std::size_t block_size,
                           bool verbose, std::size_t n_threads,
                           std::size_t grain_size) -> List {
  auto nn_graph =
      tdoann::build_nn<Distance, DQIntSampler, RPProgress, RParallel>(
          distance, k, order_by_distance, block_size, verbose, n_threads,
          grain_size);
  return graph_to_r(nn_graph);
}

template <typename Distance>
auto random_knn_query_impl(Distance &distance, std::size_t k,
                           bool order_by_distance, std::size_t block_size,
                           bool verbose, std::size_t n_threads,
                           std::size_t grain_size) -> List {
  auto nn_graph =
      tdoann::query_nn<Distance, DQIntSampler, RPProgress, RParallel>(
          distance, k, order_by_distance, block_size, verbose, n_threads,
          grain_size);
  return graph_to_r(nn_graph);
}

/* Exports */

// [[Rcpp::export]]
List random_knn_cpp(Rcpp::NumericMatrix data, int k,
                    const std::string &metric = "euclidean",
                    bool order_by_distance = true, std::size_t n_threads = 0,
                    std::size_t block_size = 4096, std::size_t grain_size = 1,
                    bool verbose = false){
    DISPATCH_ON_DISTANCES(RANDOM_NBRS_BUILD)}

// [[Rcpp::export]]
List random_knn_query_cpp(NumericMatrix reference, NumericMatrix query, int k,
                          const std::string &metric = "euclidean",
                          bool order_by_distance = true,
                          std::size_t n_threads = 0,
                          std::size_t block_size = 4096,
                          std::size_t grain_size = 1, bool verbose = false) {
  DISPATCH_ON_QUERY_DISTANCES(RANDOM_NBRS_QUERY)
}

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
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

using Rcpp::List;
using Rcpp::NumericMatrix;

// function that exists purely to hide that monstrous using declaration
template <typename DistancePtr>
decltype(auto) create_sampler(DistancePtr &&distance) {
  using IndexT = typename std::remove_reference_t<decltype(*distance)>::Index;
  return rnndescent::DQIntSampler<IndexT>();
}

// [[Rcpp::export]]
List random_knn_cpp(const NumericMatrix &data, uint32_t nnbrs,
                    const std::string &metric = "euclidean",
                    bool order_by_distance = true, std::size_t n_threads = 0,
                    bool verbose = false) {
  auto distance = create_self_distance(data, metric);
  auto sampler = create_sampler(distance);
  RPProgress progress(verbose);
  RParallelExecutor executor;

  auto nn_graph =
      tdoann::random_build(*distance, nnbrs, sampler, order_by_distance,
                           n_threads, progress, executor);

  return graph_to_r(nn_graph);
}

// [[Rcpp::export]]
List random_knn_query_cpp(const NumericMatrix &reference,
                          const NumericMatrix &query, uint32_t nnbrs,
                          const std::string &metric = "euclidean",
                          bool order_by_distance = true,
                          std::size_t n_threads = 0, bool verbose = false) {
  auto distance = create_query_distance(reference, query, metric);
  auto sampler = create_sampler(distance);
  RPProgress progress(verbose);
  RParallelExecutor executor;

  auto nn_graph =
      tdoann::random_query(*distance, nnbrs, sampler, order_by_distance,
                           n_threads, progress, executor);

  return graph_to_r(nn_graph);
}

// NOLINTEND(modernize-use-trailing-return-type)

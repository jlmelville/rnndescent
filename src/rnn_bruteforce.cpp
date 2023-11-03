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

#include "tdoann/bruteforce.h"

#include "rnn_distance.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

template <typename Out, typename Idx>
List rnn_brute_force_impl(const tdoann::BaseDistance<Out, Idx> &distance,
                          uint32_t nnbrs, std::size_t n_threads = 0,
                          bool verbose = false) {
  RPProgress progress(verbose);
  RParallelExecutor executor;

  auto nn_graph =
      tdoann::brute_force_build(distance, nnbrs, n_threads, progress, executor);
  constexpr bool unzero = false;
  return graph_to_r(nn_graph, unzero);
}

// [[Rcpp::export]]
List rnn_brute_force(const NumericMatrix &data, uint32_t nnbrs,
                     const std::string &metric = "euclidean",
                     std::size_t n_threads = 0, bool verbose = false) {
  auto distance_ptr = create_self_distance(data, metric);
  return rnn_brute_force_impl(*distance_ptr, nnbrs, n_threads, verbose);
}

// [[Rcpp::export]]
List rnn_brute_force_sparse(const NumericVector &data, const IntegerVector &ind,
                            const IntegerVector &ptr, std::size_t nobs,
                            std::size_t ndim, uint32_t nnbrs,
                            const std::string &metric = "euclidean",
                            std::size_t n_threads = 0, bool verbose = false) {
  auto distance_ptr =
      create_sparse_self_distance(data, ind, ptr, nobs, ndim, metric);
  return rnn_brute_force_impl(*distance_ptr, nnbrs, n_threads, verbose);
}

template <typename Out, typename Idx>
List rnn_brute_force_query_impl(const tdoann::BaseDistance<Out, Idx> &distance,
                                uint32_t nnbrs, std::size_t n_threads = 0,
                                bool verbose = false) {
  RPProgress progress(verbose);
  RParallelExecutor executor;

  auto nn_graph =
      tdoann::brute_force_query(distance, nnbrs, n_threads, progress, executor);
  constexpr bool unzero = false;
  return graph_to_r(nn_graph, unzero);
}

// [[Rcpp::export]]
List rnn_brute_force_query(const NumericMatrix &reference,
                           const NumericMatrix &query, uint32_t nnbrs,
                           const std::string &metric = "euclidean",
                           std::size_t n_threads = 0, bool verbose = false) {
  auto distance_ptr = create_query_distance(reference, query, metric);
  return rnn_brute_force_query_impl(*distance_ptr, nnbrs, n_threads, verbose);
}

// [[Rcpp::export]]
List rnn_brute_force_query_sparse(
    const NumericVector &ref_data, const IntegerVector &ref_ind,
    const IntegerVector &ref_ptr, std::size_t nref,
    const NumericVector &query_data, const IntegerVector &query_ind,
    const IntegerVector &query_ptr, std::size_t nquery, std::size_t ndim,
    uint32_t nnbrs, const std::string &metric = "euclidean",
    std::size_t n_threads = 0, bool verbose = false) {
  auto distance_ptr =
      create_sparse_query_distance(ref_data, ref_ind, ref_ptr, nref, query_data,
                                   query_ind, query_ptr, nquery, ndim, metric);
  return rnn_brute_force_query_impl(*distance_ptr, nnbrs, n_threads, verbose);
}

// NOLINTEND(modernize-use-trailing-return-type)

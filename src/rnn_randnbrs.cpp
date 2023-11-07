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

#include <optional>

#include <Rcpp.h>

#include "rnndescent/random.h"
#include "tdoann/randnbrs.h"

#include "rnn_distance.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

template <typename Out, typename Idx>
List random_knn_cpp_impl(const tdoann::BaseDistance<Out, Idx> &distance,
                         uint32_t nnbrs, bool order_by_distance = true,
                         std::size_t n_threads = 0, bool verbose = false) {
  RPProgress progress(verbose);
  RParallelExecutor executor;

  std::optional<tdoann::NNGraph<Out, Idx>> nn_graph;

  if (n_threads > 0) {
    rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler>
        sampler_provider;
    nn_graph =
        tdoann::random_build(distance, nnbrs, sampler_provider,
                             order_by_distance, n_threads, progress, executor);
  } else {
    rnndescent::DQIntSampler<Idx> sampler;
    nn_graph = tdoann::random_build(distance, nnbrs, sampler, order_by_distance,
                                    progress);
  }
  constexpr bool unzero = false;
  return graph_to_r(*nn_graph, unzero);
}

// [[Rcpp::export]]
List random_knn_sparse(const IntegerVector &ind, const IntegerVector &ptr,
                       const NumericVector &data, std::size_t ndim,
                       uint32_t nnbrs, const std::string &metric = "euclidean",
                       bool order_by_distance = true, std::size_t n_threads = 0,
                       bool verbose = false) {
  auto distance_ptr = create_sparse_self_distance(ind, ptr, data, ndim, metric);
  return random_knn_cpp_impl(*distance_ptr, nnbrs, order_by_distance, n_threads,
                             verbose);
}

// [[Rcpp::export]]
List random_knn_cpp(const NumericMatrix &data, uint32_t nnbrs,
                    const std::string &metric = "euclidean",
                    bool order_by_distance = true, std::size_t n_threads = 0,
                    bool verbose = false) {
  auto distance_ptr = create_self_distance(data, metric);
  return random_knn_cpp_impl(*distance_ptr, nnbrs, order_by_distance, n_threads,
                             verbose);
}

template <typename Out, typename Idx>
List random_knn_query_impl(const tdoann::BaseDistance<Out, Idx> &distance,
                           uint32_t nnbrs, bool order_by_distance = true,
                           std::size_t n_threads = 0, bool verbose = false) {
  RPProgress progress(verbose);
  RParallelExecutor executor;

  std::optional<tdoann::NNGraph<Out, Idx>> nn_graph;

  if (n_threads > 0) {
    rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler>
        sampler_provider;
    nn_graph =
        tdoann::random_query(distance, nnbrs, sampler_provider,
                             order_by_distance, n_threads, progress, executor);
  } else {
    rnndescent::DQIntSampler<Idx> sampler;
    nn_graph = tdoann::random_query(distance, nnbrs, sampler, order_by_distance,
                                    progress);
  }
  constexpr bool unzero = false;
  return graph_to_r(*nn_graph, unzero);
}

// [[Rcpp::export]]
List random_knn_query_cpp(const NumericMatrix &reference,
                          const NumericMatrix &query, uint32_t nnbrs,
                          const std::string &metric = "euclidean",
                          bool order_by_distance = true,
                          std::size_t n_threads = 0, bool verbose = false) {
  auto distance_ptr = create_query_distance(reference, query, metric);
  return random_knn_query_impl(*distance_ptr, nnbrs, order_by_distance,
                               n_threads, verbose);
}

// [[Rcpp::export]]
List random_knn_query_sparse(
    const IntegerVector &ref_ind, const IntegerVector &ref_ptr,
    const NumericVector &ref_data, const IntegerVector &query_ind,
    const IntegerVector &query_ptr, const NumericVector &query_data,
    std::size_t ndim, uint32_t nnbrs, const std::string &metric = "euclidean",
    bool order_by_distance = true, std::size_t n_threads = 0,
    bool verbose = false) {
  auto distance_ptr =
      create_sparse_query_distance(ref_ind, ref_ptr, ref_data,
                                   query_ind, query_ptr, query_data, ndim, metric);
  return random_knn_query_impl(*distance_ptr, nnbrs, order_by_distance,
                               n_threads, verbose);
}

// NOLINTEND(modernize-use-trailing-return-type)

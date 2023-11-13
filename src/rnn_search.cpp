//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2021 James Melville
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

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,modernize-use-trailing-return-type,readability-magic-numbers)

#include <Rcpp.h>

#include "tdoann/search.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_init.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_rtoheap.h"

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::LogicalMatrix;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

template <typename Out, typename Idx>
List nn_query_impl(const tdoann::BaseDistance<Out, Idx> &distance,
                   const List &reference_graph_list,
                   const IntegerMatrix &nn_idx, const NumericMatrix &nn_dist,
                   const std::string &metric = "euclidean",
                   double epsilon = 0.1, std::size_t n_threads = 0,
                   bool verbose = false) {
  const auto reference_graph =
      r_to_sparse_graph<Out, Idx>(reference_graph_list);
  auto nn_heap = r_to_query_heap<tdoann::NNHeap<Out, Idx>>(nn_idx, nn_dist);

  // replace missing data with randomly chosen neighbors so all points have
  // k initial guesses
  fill_random(nn_heap, distance, n_threads, verbose);

  RParallelExecutor executor;
  RPProgress progress(verbose);
  tdoann::nn_query(reference_graph, nn_heap, distance, epsilon, n_threads,
                   progress, executor);

  return heap_to_r(nn_heap, n_threads, progress, executor);
}

// [[Rcpp::export]]
List rnn_query(const NumericMatrix &reference, const List &reference_graph_list,
               const NumericMatrix &query, const IntegerMatrix &nn_idx,
               const NumericMatrix &nn_dist,
               const std::string &metric = "euclidean", double epsilon = 0.1,
               std::size_t n_threads = 0, bool verbose = false) {
  auto distance_ptr = create_query_distance(reference, query, metric);
  return nn_query_impl(*distance_ptr, reference_graph_list, nn_idx, nn_dist,
                       metric, epsilon, n_threads, verbose);
}

// [[Rcpp::export]]
List rnn_logical_query(const LogicalMatrix &reference,
                       const List &reference_graph_list,
                       const LogicalMatrix &query, const IntegerMatrix &nn_idx,
                       const NumericMatrix &nn_dist,
                       const std::string &metric = "euclidean",
                       double epsilon = 0.1, std::size_t n_threads = 0,
                       bool verbose = false) {
  auto distance_ptr = create_query_distance(reference, query, metric);
  return nn_query_impl(*distance_ptr, reference_graph_list, nn_idx, nn_dist,
                       metric, epsilon, n_threads, verbose);
}

// [[Rcpp::export]]
List rnn_sparse_query(
    const IntegerVector &ref_ind, const IntegerVector &ref_ptr,
    const NumericVector &ref_data, const IntegerVector &query_ind,
    const IntegerVector &query_ptr, const NumericVector &query_data,
    std::size_t ndim, const List &reference_graph_list,
    const IntegerMatrix &nn_idx, const NumericMatrix &nn_dist,
    const std::string &metric = "euclidean", double epsilon = 0.1,
    std::size_t n_threads = 0, bool verbose = false) {
  auto distance_ptr =
      create_sparse_query_distance(ref_ind, ref_ptr, ref_data, query_ind,
                                   query_ptr, query_data, ndim, metric);
  return nn_query_impl(*distance_ptr, reference_graph_list, nn_idx, nn_dist,
                       metric, epsilon, n_threads, verbose);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,modernize-use-trailing-return-type,readability-magic-numbers)

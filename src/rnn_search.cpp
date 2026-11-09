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

#include <cmath>
#include <iomanip>
#include <sstream>

#include <Rcpp.h>

#include "tdoann/search.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_init.h"
#include "rnn_nnd_batch.h"
#include "rnn_parallel.h"
#include "rnn_prepared_index.h"
#include "rnn_progress.h"
#include "rnn_rtoheap.h"

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::LogicalMatrix;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

std::string fmt_double(double d, int precision = 2) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << d;
  return oss.str();
}

template <typename Out, typename Idx, typename CalculateFour = std::nullptr_t>
List nn_query_impl(const tdoann::BaseDistance<Out, Idx> &distance,
                   const tdoann::SparseNNGraph<Out, Idx> &search_graph,
                   const IntegerMatrix &nn_idx, const NumericMatrix &nn_dist,
                   double epsilon, double max_search_fraction,
                   std::size_t n_threads, bool verbose,
                   const CalculateFour &calculate_four = nullptr) {
  auto nn_heap = r_to_query_heap<tdoann::NNHeap<Out, Idx>>(nn_idx, nn_dist);

  auto max_distance_calculations =
      static_cast<std::size_t>(search_graph.n_points * max_search_fraction);

  if (max_search_fraction < 1 && verbose) {
    tsmessage() << "max distance calculation = " << max_distance_calculations
                << "\n";
  }
  // replace missing data with randomly chosen neighbors so all points have
  // k initial guesses
  fill_random(nn_heap, distance, n_threads, verbose);

  std::vector<std::size_t> distance_counts(nn_heap.n_points, 0);

  RParallelExecutor executor;
  RPProgress progress(verbose);
  tdoann::nn_query(search_graph, nn_heap, distance, epsilon,
                   max_distance_calculations, distance_counts, n_threads,
                   progress, executor, calculate_four);

  if (verbose) {
    std::size_t min_count = 0UL;
    std::size_t max_count = 0UL;
    std::size_t sum_counts = 0UL;
    double n_points = static_cast<double>(search_graph.n_points);
    if (!distance_counts.empty()) {
      min_count = distance_counts.front();
      max_count = distance_counts.front();
    }
    for (auto count : distance_counts) {
      if (count > max_count) {
        max_count = count;
      }
      if (count < min_count) {
        min_count = count;
      }
      sum_counts += count;
    }
    double avg_count = distance_counts.empty()
                           ? 0.0
                           : static_cast<double>(sum_counts) /
                                 static_cast<double>(distance_counts.size());

    tsmessage() << "min distance calculation = " << min_count << " ("
                << fmt_double(100.0 * min_count / n_points)
                << "%) of reference data\n";
    tsmessage() << "max distance calculation = " << max_count << " ("
                << fmt_double(100.0 * max_count / n_points)
                << "%) of reference data\n";
    tsmessage() << "avg distance calculation = " << std::lround(avg_count)
                << " (" << fmt_double(100.0 * avg_count / n_points)
                << "%) of reference data\n";
  }

  return heap_to_r(nn_heap, n_threads, progress, executor);
}

template <typename Out, typename Idx, typename CalculateFour = std::nullptr_t>
List nn_query_impl(const tdoann::BaseDistance<Out, Idx> &distance,
                   const List &graph, const IntegerMatrix &nn_idx,
                   const NumericMatrix &nn_dist, double epsilon,
                   double max_search_fraction, std::size_t n_threads,
                   bool verbose,
                   const CalculateFour &calculate_four = nullptr) {
  return nn_query_impl(distance, r_to_sparse_graph<Out, Idx>(graph), nn_idx,
                       nn_dist, epsilon, max_search_fraction, n_threads,
                       verbose, calculate_four);
}

template <typename Graph>
List dense_query_impl(
    const tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST,
                                 RNN_DEFAULT_IDX> &distance,
    const Graph &graph, const IntegerMatrix &nn_idx,
    const NumericMatrix &nn_dist, const std::string &metric,
    std::size_t dimension, double epsilon, double max_search_fraction,
    std::size_t n_threads, bool verbose) {
  if (metric == "sqeuclidean" && distance.get_nx() >= 4) {
    using Kernel =
        rnndescent::DenseDistanceFour<RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>;
    const auto *data = &*distance.get_x(0);
    auto calculate_four = [&](RNN_DEFAULT_IDX row,
                              const Kernel::Targets &targets) {
      return Kernel::run_from<rnndescent::DenseBatchMetric::squared_euclidean>(
          &*distance.get_y(row), data, dimension, targets);
    };
    return nn_query_impl(distance, graph, nn_idx, nn_dist, epsilon,
                         max_search_fraction, n_threads, verbose,
                         calculate_four);
  }
  return nn_query_impl(distance, graph, nn_idx, nn_dist, epsilon,
                       max_search_fraction, n_threads, verbose);
}

// [[Rcpp::export(rng = false)]]
List rnn_query_prepared(SEXP handle, const NumericMatrix &query, uint32_t k,
                        double epsilon, double max_search_fraction,
                        std::size_t n_threads, bool verbose) {
  const auto &index = rnndescent::prepared_dense_index(handle);
  if (static_cast<std::size_t>(query.nrow()) != index.ndim) {
    Rcpp::stop("query must have the same number of features as the reference");
  }
  if (k == 0 || k > index.graph.n_points) {
    Rcpp::stop("k must be a single integer between 1 and " +
               std::to_string(index.graph.n_points));
  }
  const Rcpp::RNGScope scope;
  rnndescent::PreparedIndexQuery distance(index, query);
  using Idx = RNN_DEFAULT_IDX;
  rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler> rng_provider;
  RParallelExecutor executor;
  RPProgress progress(verbose);
  auto seed_heap =
      tdoann::search_forest(index.forest, distance, k, rng_provider, true,
                            n_threads, progress, executor);
  // Preserve the current sorted-seed order before building the traversal heap.
  const auto seeds = heap_to_r(seed_heap);
  auto result = dense_query_impl(
      distance, index.graph, seeds["idx"], seeds["dist"], index.metric,
      index.ndim, epsilon, max_search_fraction, n_threads, verbose);
  if (index.metric == "sqeuclidean") {
    NumericMatrix distances = result["dist"];
    for (auto &value : distances) {
      if (!R_IsNA(value)) {
        value = std::sqrt(value);
      }
    }
  }
  return result;
}

// The nested native search calls manage RNG state themselves.
// [[Rcpp::export(rng = false)]]
SEXP rnn_with_prepared_query(Rcpp::Function search) {
  PreparedDenseQuery distance;
  Rcpp::XPtr<PreparedDenseQuery> handle(&distance, false, prepared_query_tag());
  // Clear the handle before the owning stack frame goes away, including on
  // error.
  struct ClearHandle {
    SEXP ptr;
    ~ClearHandle() { R_ClearExternalPtr(ptr); }
  } clear{handle};
  return search(handle);
}

// [[Rcpp::export]]
List rnn_query(const NumericMatrix &reference, const List &reference_graph_list,
               const NumericMatrix &query, const IntegerMatrix &nn_idx,
               const NumericMatrix &nn_dist, const std::string &metric,
               double epsilon, double max_search_fraction,
               std::size_t n_threads, bool verbose,
               bool check_reference_during_copy = false,
               SEXP prepared_query = R_NilValue) {
  auto distance_ptr =
      prepared_query == R_NilValue
          ? create_query_vector_distance(reference, query, metric,
                                         check_reference_during_copy)
          : std::move(prepared_query_slot(prepared_query));
  if (!distance_ptr) {
    Rcpp::stop("No prepared query calculator from forest seeding");
  }
  return dense_query_impl(*distance_ptr, reference_graph_list, nn_idx, nn_dist,
                          metric, reference.nrow(), epsilon,
                          max_search_fraction, n_threads, verbose);
}

// [[Rcpp::export]]
List rnn_logical_query(const LogicalMatrix &reference,
                       const List &reference_graph_list,
                       const LogicalMatrix &query, const IntegerMatrix &nn_idx,
                       const NumericMatrix &nn_dist, const std::string &metric,
                       double epsilon, double max_search_fraction,
                       std::size_t n_threads, bool verbose) {
  auto distance_ptr = create_query_distance(reference, query, metric);
  return nn_query_impl(*distance_ptr, reference_graph_list, nn_idx, nn_dist,
                       epsilon, max_search_fraction, n_threads, verbose);
}

// [[Rcpp::export]]
List rnn_sparse_query(
    const IntegerVector &ref_ind, const IntegerVector &ref_ptr,
    const NumericVector &ref_data, const IntegerVector &query_ind,
    const IntegerVector &query_ptr, const NumericVector &query_data,
    std::size_t ndim, const List &reference_graph_list,
    const IntegerMatrix &nn_idx, const NumericMatrix &nn_dist,
    const std::string &metric, double epsilon, double max_search_fraction,
    std::size_t n_threads, bool verbose) {
  auto distance_ptr =
      create_sparse_query_distance(ref_ind, ref_ptr, ref_data, query_ind,
                                   query_ptr, query_data, ndim, metric);
  return nn_query_impl(*distance_ptr, reference_graph_list, nn_idx, nn_dist,
                       epsilon, max_search_fraction, n_threads, verbose);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,modernize-use-trailing-return-type,readability-magic-numbers)

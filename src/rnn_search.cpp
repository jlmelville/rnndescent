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

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
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

template <rnndescent::DenseBatchMetric Metric, typename Graph>
List dense_query_four_impl(
    const tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST,
                                 RNN_DEFAULT_IDX> &distance,
    const Graph &graph, const IntegerMatrix &nn_idx,
    const NumericMatrix &nn_dist, std::size_t dimension, double epsilon,
    double max_search_fraction, std::size_t n_threads, bool verbose) {
  using Kernel =
      rnndescent::DenseDistanceFour<RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>;
  const auto *data = &*distance.get_x(0);
  auto calculate_four = [&](RNN_DEFAULT_IDX row,
                            const Kernel::Targets &targets) {
    return Kernel::run_from<Metric>(&*distance.get_y(row), data, dimension,
                                    targets);
  };
  return nn_query_impl(distance, graph, nn_idx, nn_dist, epsilon,
                       max_search_fraction, n_threads, verbose, calculate_four);
}

namespace {

using QueryDistance =
    tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>;
using QueryKernel =
    rnndescent::DenseDistanceFour<RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>;
using EvaluateFour = decltype(&QueryKernel::run_from<
                              rnndescent::DenseBatchMetric::inner_product>);

struct SharedQueryFour {
  const QueryDistance &distance;
  const float *reference;
  std::size_t dimension;
  EvaluateFour evaluate;

  auto operator()(RNN_DEFAULT_IDX row,
                  const QueryKernel::Targets &targets) const
      -> QueryKernel::Distances {
    return evaluate(&*distance.get_y(row), reference, dimension, targets);
  }
};

enum class QueryProductMetric {
  cosine,
  alternative_cosine,
  correlation,
  hellinger,
  alternative_hellinger
};

template <QueryProductMetric Metric>
auto query_product_four(const float *x, const float *data,
                        std::size_t dimension,
                        const QueryKernel::Targets &targets)
    -> QueryKernel::Distances {
  constexpr bool centered = Metric == QueryProductMetric::correlation;
  constexpr bool probability =
      Metric == QueryProductMetric::hellinger ||
      Metric == QueryProductMetric::alternative_hellinger;
  std::array<const float *, 4> ys;
  for (std::size_t lane = 0; lane < ys.size(); ++lane) {
    ys[lane] = data + dimension * targets[lane];
  }
  float x_mean = 0;
  QueryKernel::Distances y_mean{};
  if constexpr (centered) {
    for (std::size_t j = 0; j < dimension; ++j) {
      x_mean += x[j];
      for (std::size_t lane = 0; lane < ys.size(); ++lane) {
        y_mean[lane] += ys[lane][j];
      }
    }
    x_mean /= dimension;
    for (auto &mean : y_mean) {
      mean /= dimension;
    }
  }

  // Accumulate the query norm once for all four candidates.
  float x_norm = 0;
  QueryKernel::Distances y_norm{}, product{};
  for (std::size_t j = 0; j < dimension; ++j) {
    const float xj = centered ? x[j] - x_mean : x[j];
    x_norm += probability ? xj : xj * xj;
    for (std::size_t lane = 0; lane < ys.size(); ++lane) {
      const float yj = centered ? ys[lane][j] - y_mean[lane] : ys[lane][j];
      const float xy = xj * yj;
      y_norm[lane] += probability ? yj : yj * yj;
      product[lane] += probability ? std::sqrt(xy) : xy;
    }
  }

  // Preserve each scalar metric's zero-norm and rounding conventions.
  QueryKernel::Distances result{};
  for (std::size_t lane = 0; lane < ys.size(); ++lane) {
    const float yn = y_norm[lane];
    const float dot = product[lane];
    if (x_norm == 0 && yn == 0) {
      result[lane] = 0;
    } else if constexpr (Metric == QueryProductMetric::alternative_cosine ||
                         Metric == QueryProductMetric::alternative_hellinger) {
      if (x_norm == 0 || yn == 0 || dot <= 0) {
        result[lane] = std::numeric_limits<float>::max();
      } else {
        const float distance = std::log2(std::sqrt(x_norm * yn) / dot);
        result[lane] = Metric == QueryProductMetric::alternative_cosine
                           ? std::max(distance, 0.0f)
                           : distance;
      }
    } else if (x_norm == 0 || yn == 0) {
      result[lane] = 1;
    } else if constexpr (Metric == QueryProductMetric::hellinger) {
      result[lane] =
          std::sqrt(std::max(1 - dot / std::sqrt(x_norm * yn), 0.0f));
    } else {
      const float distance = 1 - dot / (std::sqrt(x_norm) * std::sqrt(yn));
      result[lane] = centered ? distance : std::max(distance, 0.0f);
    }
  }
  return result;
}

enum class QueryCoordinateMetric { bray_curtis, canberra, chebyshev };

template <QueryCoordinateMetric Metric>
auto query_coordinate_four(const float *x, const float *data,
                           std::size_t dimension,
                           const QueryKernel::Targets &targets)
    -> QueryKernel::Distances {
  std::array<const float *, 4> candidates;
  for (std::size_t lane = 0; lane < candidates.size(); ++lane) {
    candidates[lane] = data + dimension * targets[lane];
  }
  QueryKernel::Distances result{};
  QueryKernel::Distances denominator{};
  for (std::size_t j = 0; j < dimension; ++j) {
    const auto anchor = x[j];
    for (std::size_t lane = 0; lane < candidates.size(); ++lane) {
      const auto other = candidates[lane][j];
      const auto difference = std::abs(anchor - other);
      if constexpr (Metric == QueryCoordinateMetric::bray_curtis) {
        result[lane] += difference;
        denominator[lane] += std::abs(anchor + other);
      } else if constexpr (Metric == QueryCoordinateMetric::canberra) {
        const auto scale = std::abs(anchor) + std::abs(other);
        if (scale > 0) {
          result[lane] += difference / scale;
        }
      } else {
        result[lane] = std::max(result[lane], difference);
      }
    }
  }
  if constexpr (Metric == QueryCoordinateMetric::bray_curtis) {
    for (std::size_t lane = 0; lane < candidates.size(); ++lane) {
      result[lane] =
          denominator[lane] == 0 ? 0 : result[lane] / denominator[lane];
    }
  }
  return result;
}

auto shared_query_kernel(const std::string &metric) -> EvaluateFour {
  if (metric == "braycurtis") {
    return query_coordinate_four<QueryCoordinateMetric::bray_curtis>;
  }
  if (metric == "canberra") {
    return query_coordinate_four<QueryCoordinateMetric::canberra>;
  }
  if (metric == "chebyshev") {
    return query_coordinate_four<QueryCoordinateMetric::chebyshev>;
  }
  using Metric = rnndescent::DenseBatchMetric;
  if (metric == "cosine") {
    return query_product_four<QueryProductMetric::cosine>;
  }
  if (metric == "alternative-cosine") {
    return query_product_four<QueryProductMetric::alternative_cosine>;
  }
  if (metric == "correlation") {
    return query_product_four<QueryProductMetric::correlation>;
  }
  if (metric == "hellinger") {
    return query_product_four<QueryProductMetric::hellinger>;
  }
  if (metric == "alternative-hellinger") {
    return query_product_four<QueryProductMetric::alternative_hellinger>;
  }
  if (metric == "manhattan") {
    return QueryKernel::run_from<Metric::manhattan>;
  }
  if (metric == "cosine-preprocess" || metric == "correlation-preprocess") {
    return QueryKernel::run_from<Metric::inner_product>;
  }
  if (metric == "dot") {
    return QueryKernel::run_from<Metric::dot>;
  }
  if (metric == "alternative-dot") {
    return QueryKernel::run_from<Metric::alternative_dot>;
  }
  return nullptr;
}

} // namespace

template <typename Graph>
List dense_query_impl(
    const tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST,
                                 RNN_DEFAULT_IDX> &distance,
    const Graph &graph, const IntegerMatrix &nn_idx,
    const NumericMatrix &nn_dist, const std::string &metric,
    std::size_t dimension, double epsilon, double max_search_fraction,
    std::size_t n_threads, bool verbose) {
  if (metric == "sqeuclidean" && distance.get_nx() >= 4) {
    return dense_query_four_impl<
        rnndescent::DenseBatchMetric::squared_euclidean>(
        distance, graph, nn_idx, nn_dist, dimension, epsilon,
        max_search_fraction, n_threads, verbose);
  }
  if (const auto evaluate = shared_query_kernel(metric);
      evaluate && distance.get_nx() >= 4) {
    const SharedQueryFour calculate_four{distance, &*distance.get_x(0),
                                         dimension, evaluate};
    return nn_query_impl(distance, graph, nn_idx, nn_dist, epsilon,
                         max_search_fraction, n_threads, verbose,
                         calculate_four);
  }
  return nn_query_impl(distance, graph, nn_idx, nn_dist, epsilon,
                       max_search_fraction, n_threads, verbose);
}

namespace {

auto prepared_seed_distances(const List &seeds, SEXP convert_seed_distances)
    -> NumericMatrix {
  const NumericMatrix distances = seeds["dist"];
  if (convert_seed_distances == R_NilValue) {
    return distances;
  }
  // Match the ordinary path's conversion to the public metric and back.
  const Rcpp::Function convert(convert_seed_distances);
  auto converted = Rcpp::as<NumericMatrix>(convert(distances));
  if (converted.nrow() != distances.nrow() ||
      converted.ncol() != distances.ncol()) {
    Rcpp::stop("Converted seed distances must preserve their dimensions");
  }
  return converted;
}

} // namespace

// [[Rcpp::export(rng = false)]]
List rnn_query_prepared(SEXP handle, const NumericMatrix &query, uint32_t k,
                        double epsilon, double max_search_fraction,
                        std::size_t n_threads, bool verbose,
                        SEXP convert_seed_distances) {
  const auto &index =
      rnndescent::prepared_index<rnndescent::PreparedDenseIndex>(
          handle, rnndescent::prepared_index_tag());
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
  const auto distances = prepared_seed_distances(seeds, convert_seed_distances);
  return dense_query_impl(distance, index.graph, seeds["idx"], distances,
                          index.metric, index.ndim, epsilon,
                          max_search_fraction, n_threads, verbose);
}

// [[Rcpp::export(rng = false)]]
List rnn_query_prepared_sparse(SEXP handle, const IntegerVector &query_ind,
                               const IntegerVector &query_ptr,
                               const NumericVector &query_data, int ndim,
                               uint32_t k, double epsilon,
                               double max_search_fraction,
                               std::size_t n_threads, bool verbose,
                               SEXP convert_seed_distances) {
  using Index = rnndescent::PreparedSparseIndex;
  const auto &index = rnndescent::prepared_index<Index>(
      handle, rnndescent::prepared_sparse_index_tag());
  if (static_cast<std::size_t>(ndim) != index.ndim) {
    Rcpp::stop("query must have the same number of features as the reference");
  }
  if (k == 0 || k > index.graph.n_points) {
    Rcpp::stop("k must be a single integer between 1 and " +
               std::to_string(index.graph.n_points));
  }
  rnndescent::check_prepared_sparse_storage(query_ind, query_ptr,
                                            query_data.size(), ndim);
  const Rcpp::RNGScope scope;
  rnndescent::PreparedSparseQuery distance(index, query_ind, query_ptr,
                                           query_data);
  rnndescent::ParallelIntRNGAdapter<Index::Idx, rnndescent::DQIntSampler>
      rng_provider;
  RParallelExecutor executor;
  RPProgress progress(verbose);
  auto seed_heap = std::visit(
      [&](const auto &forest) {
        return tdoann::search_forest(forest, distance, k, rng_provider, true,
                                     n_threads, progress, executor);
      },
      index.forest);
  const auto seeds = heap_to_r(seed_heap);
  const auto distances = prepared_seed_distances(seeds, convert_seed_distances);
  return nn_query_impl(distance, index.graph, seeds["idx"], distances, epsilon,
                       max_search_fraction, n_threads, verbose);
}

// [[Rcpp::export(rng = false)]]
List rnn_query_prepared_logical(SEXP handle, const LogicalMatrix &query,
                                uint32_t k, double epsilon,
                                double max_search_fraction,
                                std::size_t n_threads, bool verbose,
                                SEXP convert_seed_distances) {
  using Index = rnndescent::PreparedLogicalIndex;
  const auto &index = rnndescent::prepared_index<Index>(
      handle, rnndescent::prepared_logical_index_tag());
  if (static_cast<std::size_t>(query.nrow()) != index.ndim) {
    Rcpp::stop("query must have the same number of features as the reference");
  }
  if (k == 0 || k > index.graph.n_points) {
    Rcpp::stop("k must be a single integer between 1 and " +
               std::to_string(index.graph.n_points));
  }
  const Rcpp::RNGScope scope;
  rnndescent::PreparedLogicalQuery distance(index, query);
  rnndescent::ParallelIntRNGAdapter<Index::Idx, rnndescent::DQIntSampler>
      rng_provider;
  RParallelExecutor executor;
  RPProgress progress(verbose);
  auto seed_heap =
      tdoann::search_forest(index.forest, distance, k, rng_provider, true,
                            n_threads, progress, executor);
  const auto seeds = heap_to_r(seed_heap);
  const auto distances = prepared_seed_distances(seeds, convert_seed_distances);
  return nn_query_impl(distance, index.graph, seeds["idx"], distances, epsilon,
                       max_search_fraction, n_threads, verbose);
}

// The nested native search calls manage RNG state themselves.
// [[Rcpp::export(rng = false)]]
SEXP rnn_with_prepared_query(Rcpp::Function search) {
  PreparedQuery distance;
  Rcpp::XPtr<PreparedQuery> handle(&distance, false, prepared_query_tag());
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
  auto distance_ptr = take_prepared_query(prepared_query);
  if (!distance_ptr) {
    distance_ptr = create_query_vector_distance(reference, query, metric,
                                                check_reference_during_copy);
  }
  using Distance =
      tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>;
  return dense_query_impl(dynamic_cast<const Distance &>(*distance_ptr),
                          reference_graph_list, nn_idx, nn_dist, metric,
                          reference.nrow(), epsilon, max_search_fraction,
                          n_threads, verbose);
}

// [[Rcpp::export]]
List rnn_logical_query(const LogicalMatrix &reference,
                       const List &reference_graph_list,
                       const LogicalMatrix &query, const IntegerMatrix &nn_idx,
                       const NumericMatrix &nn_dist, const std::string &metric,
                       double epsilon, double max_search_fraction,
                       std::size_t n_threads, bool verbose,
                       SEXP prepared_query = R_NilValue) {
  auto distance_ptr = take_prepared_query(prepared_query);
  if (!distance_ptr) {
    distance_ptr = create_query_distance(reference, query, metric);
  }
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
    std::size_t n_threads, bool verbose, SEXP prepared_query = R_NilValue) {
  auto distance_ptr = take_prepared_query(prepared_query);
  if (!distance_ptr) {
    distance_ptr =
        create_sparse_query_distance(ref_ind, ref_ptr, ref_data, query_ind,
                                     query_ptr, query_data, ndim, metric);
  }
  return nn_query_impl(*distance_ptr, reference_graph_list, nn_idx, nn_dist,
                       epsilon, max_search_fraction, n_threads, verbose);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,modernize-use-trailing-return-type,readability-magic-numbers)

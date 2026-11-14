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

#include <algorithm>
#include <cmath>
#include <limits>

#include <Rcpp.h>

#include "rnndescent/random.h"
#include "tdoann/distancebase.h"
#include "tdoann/nndcommon.h"
#include "tdoann/nndescent.h"
#include "tdoann/nndparallel.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_init.h"
#include "rnn_nnd_batch.h"
#include "rnn_progress.h"
#include "rnn_rtoheap.h"

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::LogicalMatrix;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

std::unique_ptr<tdoann::NNDProgressBase>
create_nnd_progress(const std::string &progress_type, uint32_t n_iters,
                    bool verbose) {
  if (progress_type == "bar") {
    return std::make_unique<tdoann::NNDProgress>(
        std::make_unique<RPProgress>(n_iters, verbose));
  }
  return std::make_unique<tdoann::HeapSumProgress>(
      std::make_unique<RIterProgress>(n_iters, verbose));
}

template <typename Out, typename Idx>
std::unique_ptr<tdoann::ParallelLocalJoin<Out, Idx>>
create_parallel_local_join(const tdoann::NNDHeap<Out, Idx> &nn_heap,
                           const tdoann::BaseDistance<Out, Idx> &distance,
                           bool low_memory) {
  if (low_memory) {
    return std::make_unique<tdoann::LowMemParallelLocalJoin<Out, Idx>>(
        distance);
  }
  return std::make_unique<tdoann::CacheParallelLocalJoin<Out, Idx>>(nn_heap,
                                                                    distance);
}

template <typename Out, typename Idx>
std::unique_ptr<tdoann::SerialLocalJoin<Out, Idx>>
create_serial_local_join(const tdoann::NNDHeap<Out, Idx> &nn_heap,
                         const tdoann::BaseDistance<Out, Idx> &distance,
                         bool low_memory) {
  if (low_memory) {
    return std::make_unique<tdoann::LowMemSerialLocalJoin<Out, Idx>>(distance);
  }
  return std::make_unique<tdoann::CacheSerialLocalJoin<Out, Idx>>(nn_heap,
                                                                  distance);
}

template <typename Out, typename Idx>
List nn_descent_impl(
    const tdoann::BaseDistance<Out, Idx> &distance, const IntegerMatrix &nn_idx,
    const NumericMatrix &nn_dist, std::size_t max_candidates, uint32_t n_iters,
    double delta, bool low_memory, bool weight_by_degree, std::size_t n_threads,
    bool verbose, const std::string &progress_type,
    const rnndescent::DenseDistanceFour<Out, Idx> *dense_kernel = nullptr) {
  auto nnd_heap =
      r_to_knn_heap<tdoann::NNDHeap<Out, Idx>>(nn_idx, nn_dist, n_threads);

  // fill any space in the heap with random neighbors
  fill_random(nnd_heap, distance, n_threads, verbose);

  auto nnd_progress_ptr = create_nnd_progress(progress_type, n_iters, verbose);
  RParallelExecutor executor;

  if (n_threads > 0) {
    rnndescent::ParallelRNGAdapter<rnndescent::PcgRand> parallel_rand;
    auto build = [&](auto &local_join) {
      tdoann::nnd_build_parallel(nnd_heap, local_join, max_candidates, n_iters,
                                 delta, weight_by_degree, *nnd_progress_ptr,
                                 parallel_rand, n_threads, executor);
    };

    if (dense_kernel && low_memory) {
      rnndescent::DenseParallelLocalJoin<Out, Idx, false> local_join(
          *dense_kernel, distance);
      build(local_join);
    } else if (dense_kernel && dense_kernel->evaluate) {
      rnndescent::DenseParallelLocalJoin<Out, Idx, true> local_join(
          *dense_kernel, nnd_heap, distance);
      build(local_join);
    } else {
      auto local_join =
          create_parallel_local_join(nnd_heap, distance, low_memory);
      build(*local_join);
    }
  } else {
    auto local_join_ptr =
        create_serial_local_join(nnd_heap, distance, low_memory);
    rnndescent::RRand rand;
    tdoann::nnd_build(nnd_heap, *local_join_ptr, max_candidates, n_iters, delta,
                      weight_by_degree, rand, *nnd_progress_ptr);
  }

  return heap_to_r(nnd_heap, n_threads, nnd_progress_ptr->get_base_progress(),
                   executor);
}

namespace {
using DenseDistance =
    tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>;
using BatchDistance =
    rnndescent::DenseDistanceFour<RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>;

auto cosine_norms(const float *data, std::size_t count, std::size_t dimension)
    -> std::vector<float> {
  std::vector<float> norms(count);
  for (std::size_t row = 0; row < count; ++row) {
    float squared = 0;
    const auto *values = data + row * dimension;
    // Keep the indexed scan and feature order for float rounding parity.
    for (std::size_t j = 0; j < dimension; ++j) {
      squared += values[j] * values[j];
    }
    // Keep the original kernel for zero and near the limits of float
    // arithmetic, where rounding can change underflow or overflow.
    constexpr float lower = 4 * std::numeric_limits<float>::min();
    constexpr float upper = std::numeric_limits<float>::max() / 4;
    norms[row] =
        squared >= lower && squared <= upper ? std::sqrt(squared) : -1.0f;
  }
  return norms;
}

class CosineSelfDistance final
    : public tdoann::SelfDistanceCalculator<RNN_DEFAULT_IN, RNN_DEFAULT_DIST,
                                            RNN_DEFAULT_IDX> {
  using Base = tdoann::SelfDistanceCalculator<RNN_DEFAULT_IN, RNN_DEFAULT_DIST,
                                              RNN_DEFAULT_IDX>;
  std::vector<float> norms;

public:
  CosineSelfDistance(std::vector<float> &&data, std::size_t dimension,
                     DistanceFunc ordinary_cosine)
      : Base(std::move(data), dimension, ordinary_cosine),
        norms(cosine_norms(x.data(), nx, ndim)) {}

  float calculate(const RNN_DEFAULT_IDX &i,
                  const RNN_DEFAULT_IDX &j) const override {
    const float norm_i = norms[i];
    const float norm_j = norms[j];
    auto first = get_x(i);
    const auto end = first + ndim;
    auto second = get_x(j);
    if (norm_i < 0 || norm_j < 0) {
      return distance_func(first, end, second);
    }
    float dot = 0;
    for (; first != end; ++first, ++second) {
      const float a = *first;
      const float b = *second;
      dot += a * b;
    }
    return std::max(1.0f - dot / (norm_i * norm_j), 0.0f);
  }
};

auto batch_kernel(const std::string &metric) -> BatchDistance::Evaluate {
  using Metric = rnndescent::DenseBatchMetric;
  if (metric == "sqeuclidean") {
    return BatchDistance::run<Metric::squared_euclidean>;
  }
  if (metric == "euclidean") {
    return BatchDistance::run<Metric::euclidean>;
  }
  if (metric == "manhattan") {
    return BatchDistance::run<Metric::manhattan>;
  }
  if (metric == "cosine-preprocess" || metric == "correlation-preprocess") {
    return BatchDistance::run<Metric::inner_product>;
  }
  if (metric == "dot") {
    return BatchDistance::run<Metric::dot>;
  }
  if (metric == "alternative-dot") {
    return BatchDistance::run<Metric::alternative_dot>;
  }
  return nullptr;
}
} // namespace

// [[Rcpp::export]]
List rnn_descent(const NumericMatrix &data, const IntegerMatrix &nn_idx,
                 const NumericMatrix &nn_dist, const std::string &metric,
                 std::size_t max_candidates, uint32_t n_iters, double delta,
                 bool low_memory, bool weight_by_degree, std::size_t n_threads,
                 bool verbose, const std::string &progress_type) {
  BatchDistance kernel;
  if (metric == "cosine") {
    const CosineSelfDistance distance(
        r_to_vec<RNN_DEFAULT_IN>(data), data.nrow(),
        get_dense_distance_funcs<RNN_DEFAULT_IN, RNN_DEFAULT_DIST>(metric)
            .first);
    return nn_descent_impl(distance, nn_idx, nn_dist, max_candidates, n_iters,
                           delta, low_memory, weight_by_degree, n_threads,
                           verbose, progress_type, &kernel);
  }
  auto distance_ptr = create_self_distance_impl<DenseDistance>(data, metric);
  if (n_threads > 0 && distance_ptr->get_nx() > 0) {
    if (const auto evaluate = batch_kernel(metric)) {
      kernel = {&*distance_ptr->get_x(0), static_cast<std::size_t>(data.nrow()),
                evaluate};
    }
  }
  return nn_descent_impl(*distance_ptr, nn_idx, nn_dist, max_candidates,
                         n_iters, delta, low_memory, weight_by_degree,
                         n_threads, verbose, progress_type, &kernel);
}

// [[Rcpp::export]]
List rnn_logical_descent(const LogicalMatrix &data, const IntegerMatrix &nn_idx,
                         const NumericMatrix &nn_dist,
                         const std::string &metric, std::size_t max_candidates,
                         uint32_t n_iters, double delta, bool low_memory,
                         bool weight_by_degree, std::size_t n_threads,
                         bool verbose, const std::string &progress_type) {
  auto distance_ptr = create_self_distance(data, metric);
  return nn_descent_impl(*distance_ptr, nn_idx, nn_dist, max_candidates,
                         n_iters, delta, low_memory, weight_by_degree,
                         n_threads, verbose, progress_type);
}

// [[Rcpp::export]]
List rnn_sparse_descent(const IntegerVector &ind, const IntegerVector &ptr,
                        const NumericVector &data, std::size_t ndim,
                        const IntegerMatrix &nn_idx,
                        const NumericMatrix &nn_dist, const std::string &metric,
                        std::size_t max_candidates, uint32_t n_iters,
                        double delta, bool low_memory, bool weight_by_degree,
                        std::size_t n_threads, bool verbose,
                        const std::string &progress_type) {
  auto distance_ptr = create_sparse_self_distance(ind, ptr, data, ndim, metric);
  return nn_descent_impl(*distance_ptr, nn_idx, nn_dist, max_candidates,
                         n_iters, delta, low_memory, weight_by_degree,
                         n_threads, verbose, progress_type);
}

// NOLINTEND(modernize-use-trailing-return-type)

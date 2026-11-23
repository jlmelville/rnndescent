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

#include <Rcpp.h>

#include "rnndescent/random.h"
#include "tdoann/distancebase.h"
#include "tdoann/nndcommon.h"
#include "tdoann/nndescent.h"
#include "tdoann/nndparallel.h"

#include "rnn_candidate_routing.h"
#include "rnn_distance.h"
#include "rnn_distance_cosine.h"
#include "rnn_heaptor.h"
#include "rnn_init.h"
#include "rnn_nnd_batch.h"
#include "rnn_nnd_route.h"
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
List refine_from_heap(
    const tdoann::BaseDistance<Out, Idx> &distance,
    tdoann::NNDHeap<Out, Idx> &nnd_heap, std::size_t max_candidates,
    uint32_t n_iters, double delta, bool low_memory, bool weight_by_degree,
    std::size_t n_threads, bool verbose, const std::string &progress_type,
    bool packed_binary = false,
    const rnndescent::DenseDistanceFour<Out, Idx> *dense_kernel = nullptr) {
  // fill any space in the heap with random neighbors
  fill_random(nnd_heap, distance, n_threads, verbose);

  auto nnd_progress_ptr = create_nnd_progress(progress_type, n_iters, verbose);
  RParallelExecutor executor;

  if (n_threads > 0) {
    rnndescent::ParallelRNGAdapter<rnndescent::PcgRand> parallel_rand;
    auto build = [&](auto &local_join) {
      using Routing = rnndescent::CompactCandidates<Out, Idx>;
      if ((dense_kernel || packed_binary) &&
          Routing::supported(n_threads, nnd_heap.idx.size())) {
        Routing routing;
        auto populate = [&](auto &fresh, auto &old) {
          routing.build(nnd_heap, fresh, old, weight_by_degree, parallel_rand,
                        n_threads, executor);
        };
        tdoann::nnd_build_parallel(nnd_heap, local_join, max_candidates,
                                   n_iters, delta, *nnd_progress_ptr, n_threads,
                                   executor, populate);
        return;
      }
      tdoann::nnd_build_parallel(nnd_heap, local_join, max_candidates, n_iters,
                                 delta, weight_by_degree, *nnd_progress_ptr,
                                 parallel_rand, n_threads, executor);
    };

    if (packed_binary && low_memory && n_threads > 1) {
      rnndescent::BinaryDirectJoin<Out, Idx> local_join(distance);
      build(local_join);
    } else if (dense_kernel && low_memory) {
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
    std::unique_ptr<tdoann::SerialLocalJoin<Out, Idx>> local_join_ptr;
    if (low_memory && dense_kernel && dense_kernel->evaluate) {
      local_join_ptr =
          std::make_unique<rnndescent::DenseLowMemSerialLocalJoin<Out, Idx>>(
              *dense_kernel, distance);
    } else {
      local_join_ptr = create_serial_local_join(nnd_heap, distance, low_memory);
    }
    rnndescent::RRand rand;
    tdoann::nnd_build(nnd_heap, *local_join_ptr, max_candidates, n_iters, delta,
                      weight_by_degree, rand, *nnd_progress_ptr);
  }

  return heap_to_r(nnd_heap, n_threads, nnd_progress_ptr->get_base_progress(),
                   executor);
}

template <typename Out, typename Idx>
List nn_descent_impl(
    const tdoann::BaseDistance<Out, Idx> &distance, const IntegerMatrix &nn_idx,
    const NumericMatrix &nn_dist, std::size_t max_candidates, uint32_t n_iters,
    double delta, bool low_memory, bool weight_by_degree, std::size_t n_threads,
    bool verbose, const std::string &progress_type, bool packed_binary = false,
    const rnndescent::DenseDistanceFour<Out, Idx> *dense_kernel = nullptr) {
  auto nnd_heap =
      r_to_knn_heap<tdoann::NNDHeap<Out, Idx>>(nn_idx, nn_dist, n_threads);

  return refine_from_heap(distance, nnd_heap, max_candidates, n_iters, delta,
                          low_memory, weight_by_degree, n_threads, verbose,
                          progress_type, packed_binary, dense_kernel);
}

namespace {
using DenseDistance =
    tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>;
using SparseDistance =
    tdoann::SparseSelfDistanceCalculator<RNN_DEFAULT_IN, RNN_DEFAULT_DIST,
                                         RNN_DEFAULT_IDX>;
using BatchDistance =
    rnndescent::DenseDistanceFour<RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>;

auto batch_kernel(const std::string &metric, const float *data,
                  std::size_t dimension) -> BatchDistance {
  using Metric = rnndescent::DenseBatchMetric;
  if (metric == "sqeuclidean") {
    return BatchDistance::bind<Metric::squared_euclidean>(data, dimension);
  }
  if (metric == "euclidean") {
    return BatchDistance::bind<Metric::euclidean>(data, dimension);
  }
  if (metric == "manhattan") {
    return BatchDistance::bind<Metric::manhattan>(data, dimension);
  }
  if (metric == "cosine-preprocess" || metric == "correlation-preprocess") {
    return BatchDistance::bind<Metric::inner_product>(data, dimension);
  }
  if (metric == "dot") {
    return BatchDistance::bind<Metric::dot>(data, dimension);
  }
  if (metric == "alternative-dot") {
    return BatchDistance::bind<Metric::alternative_dot>(data, dimension);
  }
  return {};
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
    const rnndescent::CosineSelfDistance<
        rnndescent::CosineBuildStage::refinement>
        distance(r_to_vec<RNN_DEFAULT_IN>(data), data.nrow());
    kernel = distance.batch_kernel();
    return nn_descent_impl(distance, nn_idx, nn_dist, max_candidates, n_iters,
                           delta, low_memory, weight_by_degree, n_threads,
                           verbose, progress_type, false, &kernel);
  }
  auto distance_ptr = create_self_distance_impl<DenseDistance>(data, metric);
  if (n_threads > 0 && distance_ptr->get_nx() > 0) {
    kernel = batch_kernel(metric, &*distance_ptr->get_x(0), data.nrow());
  }
  return nn_descent_impl(*distance_ptr, nn_idx, nn_dist, max_candidates,
                         n_iters, delta, low_memory, weight_by_degree,
                         n_threads, verbose, progress_type, false, &kernel);
}

// [[Rcpp::export]]
List rnn_binary_tree_descent(const LogicalMatrix &data, uint32_t k,
                             const std::string &metric, uint32_t n_trees,
                             uint32_t leaf_size, uint32_t max_tree_depth,
                             bool ret_forest, std::size_t max_candidates,
                             uint32_t n_iters, double delta, bool low_memory,
                             bool weight_by_degree, std::size_t n_threads,
                             bool verbose, const std::string &progress_type) {
  auto distance = create_self_distance(data, metric);
  List forest;
  RParallelExecutor executor;
  RInterruptableProgress progress;
  auto graph = [&]() {
    auto leaves = implicit_leaf_heap(
        *distance, data.nrow(), k, metric, n_trees, leaf_size, max_tree_depth,
        false, n_threads, verbose, ret_forest ? &forest : nullptr);
    if (verbose && std::find(leaves.idx.begin(), leaves.idx.end(),
                             leaves.npos()) != leaves.idx.end()) {
      tsmessage() << "Warning: initialization failed to find " << k
                  << " neighbors for all points\n";
    }
    return rnndescent::import_leaf_heap(leaves, n_threads, progress, executor);
  }();
  if (verbose) {
    auto &out = tsmessage();
    out << "Running nearest neighbor descent for " << n_iters << " iterations";
    if (n_threads > 0) {
      out << " using " << n_threads << " threads";
    }
    out << '\n';
  }
  auto result = refine_from_heap(*distance, graph, max_candidates, n_iters,
                                 delta, low_memory, weight_by_degree, n_threads,
                                 verbose, progress_type, true);
  if (ret_forest) {
    result["forest"] = forest;
  }
  return result;
}

// [[Rcpp::export]]
List rnn_logical_descent(const LogicalMatrix &data, const IntegerMatrix &nn_idx,
                         const NumericMatrix &nn_dist,
                         const std::string &metric, std::size_t max_candidates,
                         uint32_t n_iters, double delta, bool low_memory,
                         bool weight_by_degree, std::size_t n_threads,
                         bool verbose, const std::string &progress_type) {
  auto distance_ptr = create_self_distance(data, metric);
  const bool packed =
      find_packed_distance_func<RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>(metric) !=
      nullptr;
  return nn_descent_impl(*distance_ptr, nn_idx, nn_dist, max_candidates,
                         n_iters, delta, low_memory, weight_by_degree,
                         n_threads, verbose, progress_type, packed);
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
  auto distance_ptr = create_sparse_self_distance_impl<SparseDistance>(
      ind, ptr, data, ndim, metric);
  if (metric == "cosine") {
    distance_ptr->distance_func =
        tdoann::sparse_cosine<RNN_DEFAULT_DIST, SparseDistance::DataIterator,
                              true>;
  }
  return nn_descent_impl(*distance_ptr, nn_idx, nn_dist, max_candidates,
                         n_iters, delta, low_memory, weight_by_degree,
                         n_threads, verbose, progress_type);
}

// NOLINTEND(modernize-use-trailing-return-type)

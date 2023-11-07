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
#include "tdoann/distancebase.h"
#include "tdoann/nndcommon.h"
#include "tdoann/nndescent.h"
#include "tdoann/nndparallel.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_init.h"
#include "rnn_progress.h"
#include "rnn_rtoheap.h"

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
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
List nn_descent_impl(const tdoann::BaseDistance<Out, Idx> &distance,
                     const IntegerMatrix &nn_idx, const NumericMatrix &nn_dist,
                     std::size_t max_candidates, uint32_t n_iters, double delta,
                     bool low_memory, std::size_t n_threads, bool verbose,
                     const std::string &progress_type) {
  auto nnd_heap =
      r_to_knn_heap<tdoann::NNDHeap<Out, Idx>>(nn_idx, nn_dist, n_threads);

  // fill any space in the heap with random neighbors
  fill_random(nnd_heap, distance, n_threads, verbose);

  auto nnd_progress_ptr = create_nnd_progress(progress_type, n_iters, verbose);
  RParallelExecutor executor;

  if (n_threads > 0) {
    auto local_join_ptr =
        create_parallel_local_join(nnd_heap, distance, low_memory);
    rnndescent::ParallelRNGAdapter<rnndescent::PcgRand> parallel_rand;
    tdoann::nnd_build(nnd_heap, *local_join_ptr, max_candidates, n_iters, delta,
                      *nnd_progress_ptr, parallel_rand, n_threads, executor);
  } else {
    auto local_join_ptr =
        create_serial_local_join(nnd_heap, distance, low_memory);
    rnndescent::RRand rand;
    tdoann::nnd_build(nnd_heap, *local_join_ptr, max_candidates, n_iters, delta,
                      rand, *nnd_progress_ptr);
  }

  return heap_to_r(nnd_heap, n_threads, nnd_progress_ptr->get_base_progress(),
                   executor);
}

// [[Rcpp::export]]
List rnn_descent(const NumericMatrix &data, const IntegerMatrix &nn_idx,
                 const NumericMatrix &nn_dist, const std::string &metric,
                 std::size_t max_candidates, uint32_t n_iters, double delta,
                 bool low_memory, std::size_t n_threads, bool verbose,
                 const std::string &progress_type) {
  auto distance_ptr = create_self_distance(data, metric);
  return nn_descent_impl(*distance_ptr, nn_idx, nn_dist, max_candidates,
                         n_iters, delta, low_memory, n_threads, verbose,
                         progress_type);
}
// [[Rcpp::export]]
List rnn_sparse_descent(const IntegerVector &ind, const IntegerVector &ptr,
                        const NumericVector &data, std::size_t ndim,
                        const IntegerMatrix &nn_idx,
                        const NumericMatrix &nn_dist, const std::string &metric,
                        std::size_t max_candidates, uint32_t n_iters,
                        double delta, bool low_memory, std::size_t n_threads,
                        bool verbose, const std::string &progress_type) {
  auto distance_ptr = create_sparse_self_distance(ind, ptr, data, ndim, metric);
  return nn_descent_impl(*distance_ptr, nn_idx, nn_dist, max_candidates,
                         n_iters, delta, low_memory, n_threads, verbose,
                         progress_type);
}

// NOLINTEND(modernize-use-trailing-return-type)

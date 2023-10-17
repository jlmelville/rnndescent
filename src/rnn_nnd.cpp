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
#include "rnn_progress.h"
#include "rnn_rtoheap.h"

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

std::unique_ptr<tdoann::NNDProgressBase>
create_nnd_progress(const std::string &progress_type, unsigned int n_iters,
                    bool verbose) {
  if (progress_type == "bar") {
    return std::make_unique<tdoann::NNDProgress>(
        std::make_unique<RPProgress>(n_iters, verbose));
  }
  return std::make_unique<tdoann::HeapSumProgress>(
      std::make_unique<RIterProgress>(n_iters, verbose));
}

template <typename Out, typename Index>
std::unique_ptr<tdoann::ParallelLocalJoin<Out, Index>>
create_parallel_local_join(const tdoann::NNDHeap<Out, Index> &nn_heap,
                           const tdoann::BaseDistance<Out, Index> &distance,
                           bool low_memory) {
  if (low_memory) {
    return std::make_unique<tdoann::LowMemParallelLocalJoin<Out, Index>>(
        distance);
  }
  return std::make_unique<tdoann::CacheParallelLocalJoin<Out, Index>>(nn_heap,
                                                                      distance);
}

template <typename Out, typename Index>
std::unique_ptr<tdoann::SerialLocalJoin<Out, Index>>
create_serial_local_join(const tdoann::NNDHeap<Out, Index> &nn_heap,
                         const tdoann::BaseDistance<Out, Index> &distance,
                         bool low_memory) {
  if (low_memory) {
    return std::make_unique<tdoann::LowMemSerialLocalJoin<Out, Index>>(
        distance);
  }
  return std::make_unique<tdoann::CacheSerialLocalJoin<Out, Index>>(nn_heap,
                                                                    distance);
}

// [[Rcpp::export]]
List nn_descent(const NumericMatrix &data, const IntegerMatrix &nn_idx,
                const NumericMatrix &nn_dist, const std::string &metric,
                std::size_t max_candidates, unsigned int n_iters, double delta,
                bool low_memory, std::size_t n_threads, bool verbose,
                const std::string &progress_type) {
  auto distance = create_self_distance(data, metric);
  using Out = typename tdoann::DistanceTraits<decltype(distance)>::Output;
  using Idx = typename tdoann::DistanceTraits<decltype(distance)>::Index;

  constexpr bool missing_ok = false;
  auto nnd_heap = r_to_knn_heap<tdoann::NNDHeap<Out, Idx>>(
      nn_idx, nn_dist, n_threads, missing_ok);

  auto nnd_progress = create_nnd_progress(progress_type, n_iters, verbose);
  RParallelExecutor executor;

  if (n_threads > 0) {
    auto local_join =
        create_parallel_local_join(nnd_heap, *distance, low_memory);
    rnndescent::ParallelRNGAdapter<rnndescent::PcgRand> parallel_rand;
    tdoann::nnd_build(nnd_heap, *local_join, max_candidates, n_iters, delta,
                      *nnd_progress, parallel_rand, n_threads, executor);
  } else {
    auto local_join = create_serial_local_join(nnd_heap, *distance, low_memory);
    rnndescent::RRand rand;
    tdoann::nnd_build(nnd_heap, *local_join, max_candidates, n_iters, delta,
                      rand, *nnd_progress);
  }

  return heap_to_r(nnd_heap, n_threads, nnd_progress->get_base_progress(),
                   executor);
}

// NOLINTEND(modernize-use-trailing-return-type)

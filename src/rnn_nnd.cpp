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
#include "tdoann/nndcommon.h"
#include "tdoann/nndescent.h"
#include "tdoann/nndparallel.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_macros.h"
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

template <typename Distance>
std::unique_ptr<tdoann::ParallelLocalJoin<Distance>> create_parallel_local_join(
    const tdoann::NNDHeap<typename Distance::Output, typename Distance::Index>
        &nn_heap,
    Distance &distance, bool low_memory) {
  if (low_memory) {
    return std::make_unique<tdoann::LowMemParallelLocalJoin<Distance>>(
        distance);
  }
  return std::make_unique<tdoann::CacheParallelLocalJoin<Distance>>(nn_heap,
                                                                    distance);
}

template <typename Distance>
std::unique_ptr<tdoann::SerialLocalJoin<Distance>> create_serial_local_join(
    const tdoann::NNDHeap<typename Distance::Output, typename Distance::Index>
        &nn_heap,
    Distance &distance, bool low_memory) {
  if (low_memory) {
    return std::make_unique<tdoann::LowMemSerialLocalJoin<Distance>>(distance);
  }
  return std::make_unique<tdoann::CacheSerialLocalJoin<Distance>>(nn_heap,
                                                                  distance);
}

#define NND_IMPL()                                                             \
  return nnd_impl.get_nn<Distance>(nn_idx, nn_dist, max_candidates, n_iters,   \
                                   delta, progress_type, low_memory, verbose);

#define NND_BUILD_UPDATER()                                                    \
  if (n_threads > 0) {                                                         \
    using NNDImpl = NNDBuildParallel;                                          \
    NNDImpl nnd_impl(data, n_threads);                                         \
    NND_IMPL()                                                                 \
  }                                                                            \
  using NNDImpl = NNDBuildSerial;                                              \
  NNDImpl nnd_impl(data);                                                      \
  NND_IMPL()

class NNDBuildSerial {

public:
  explicit NNDBuildSerial(const NumericMatrix &data) : data(data) {}

  template <typename Distance>
  auto get_nn(const IntegerMatrix &nn_idx, const NumericMatrix &nn_dist,
              std::size_t max_candidates, unsigned int n_iters, double delta,
              const std::string &progress_type, bool low_memory, bool verbose)
      -> List {
    using Out = typename Distance::Output;
    using Index = typename Distance::Index;

    constexpr bool missing_ok = false; // but should it be?
    auto nnd_heap =
        r_to_knn_heap<tdoann::NNDHeap<Out, Index>>(nn_idx, nn_dist, missing_ok);
    auto distance = tr_to_dist<Distance>(data);
    auto local_join =
        create_serial_local_join<Distance>(nnd_heap, distance, low_memory);
    auto nnd_progress = create_nnd_progress(progress_type, n_iters, verbose);
    rnndescent::RRand rand;

    tdoann::nnd_build(nnd_heap, *local_join, max_candidates, n_iters, delta,
                      rand, *nnd_progress);
    return heap_to_r(nnd_heap);
  }

private:
  NumericMatrix data;
};

class NNDBuildParallel {
public:
  NNDBuildParallel(const NumericMatrix &data, std::size_t n_threads)
      : data(data), n_threads(n_threads) {}

  template <typename Distance>
  auto get_nn(const IntegerMatrix &nn_idx, const NumericMatrix &nn_dist,
              std::size_t max_candidates, unsigned int n_iters, double delta,
              const std::string &progress_type, bool low_memory, bool verbose)
      -> List {
    using Out = typename Distance::Output;
    using Index = typename Distance::Index;

    constexpr bool missing_ok = false;
    auto nnd_heap = r_to_knn_heap<tdoann::NNDHeap<Out, Index>>(
        nn_idx, nn_dist, n_threads, missing_ok);
    auto distance = tr_to_dist<Distance>(data);
    auto local_join =
        create_parallel_local_join<Distance>(nnd_heap, distance, low_memory);
    auto nnd_progress = create_nnd_progress(progress_type, n_iters, verbose);
    rnndescent::ParallelRNGAdapter<rnndescent::PcgRand> parallel_rand;
    RParallelExecutor executor;

    tdoann::nnd_build(nnd_heap, *local_join, max_candidates, n_iters, delta,
                      *nnd_progress, parallel_rand, n_threads, executor);

    return heap_to_r(nnd_heap, n_threads, nnd_progress->get_base_progress(),
                     executor);
  }

private:
  NumericMatrix data;
  std::size_t n_threads;
};

// [[Rcpp::export]]
List nn_descent(const NumericMatrix &data, const IntegerMatrix &nn_idx,
                const NumericMatrix &nn_dist, const std::string &metric,
                std::size_t max_candidates, unsigned int n_iters, double delta,
                bool low_memory, std::size_t n_threads, bool verbose,
                const std::string &progress_type) {

  DISPATCH_ON_DISTANCES(NND_BUILD_UPDATER);
}

// NOLINTEND(modernize-use-trailing-return-type)

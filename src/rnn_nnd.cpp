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

#include <Rcpp.h>

#include "tdoann/distance.h"
#include "tdoann/graphupdate.h"
#include "tdoann/heap.h"
#include "tdoann/nndescent.h"
#include "tdoann/nndparallel.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_rng.h"
#include "rnn_rtoheap.h"

using namespace Rcpp;

#define NND_IMPL()                                                             \
  return nnd_impl.get_nn<GraphUpdate, Distance, Progress, NNDProgress>(        \
      nn_idx, nn_dist, max_candidates, n_iters, delta, verbose, weighted);

#define NND_QUERY_IMPL()                                                       \
  return nnd_impl.get_nn<Distance, Progress>(nn_idx, nn_dist, max_candidates,  \
                                             epsilon, n_iters, verbose);

#define NND_PROGRESS()                                                         \
  if (progress == "bar") {                                                     \
    using Progress = RPProgress;                                               \
    using NNDProgress = tdoann::NNDProgress<Progress>;                         \
    NND_IMPL()                                                                 \
  } else {                                                                     \
    using Progress = RIterProgress;                                            \
    using NNDProgress = tdoann::HeapSumProgress<Progress>;                     \
    NND_IMPL()                                                                 \
  }

#define NND_QUERY_PROGRESS()                                                   \
  using Progress = RPProgress;                                                 \
  NND_QUERY_IMPL()

#define NND_BUILD_UPDATER()                                                    \
  if (n_threads > 0) {                                                         \
    using NNDImpl = NNDBuildParallel;                                          \
    NNDImpl nnd_impl(data, block_size, n_threads, grain_size);                 \
    if (low_memory) {                                                          \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::Batch>;            \
      NND_PROGRESS()                                                           \
    } else {                                                                   \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::BatchHiMem>;       \
      NND_PROGRESS()                                                           \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDBuildSerial;                                            \
    NNDImpl nnd_impl(data);                                                    \
    if (low_memory) {                                                          \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::Serial>;           \
      NND_PROGRESS()                                                           \
    } else {                                                                   \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::SerialHiMem>;      \
      NND_PROGRESS()                                                           \
    }                                                                          \
  }

#define NND_QUERY_UPDATER()                                                    \
  if (n_threads > 0) {                                                         \
    using NNDImpl = NNDQueryParallel;                                          \
    NNDImpl nnd_impl(reference, query, reference_idx, reference_dist,          \
                     n_threads, grain_size);                                   \
    NND_QUERY_PROGRESS()                                                       \
  } else {                                                                     \
    using NNDImpl = NNDQuerySerial;                                            \
    NNDImpl nnd_impl(reference, query, reference_idx, reference_dist);         \
    NND_QUERY_PROGRESS()                                                       \
  }

struct NNDBuildSerial {
  NumericMatrix data;

  NNDBuildSerial(NumericMatrix data) : data(data) {}

  template <typename GraphUpdate, typename Distance, typename Progress,
            typename NNDProgress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false, bool weighted = false)
      -> List {
    using Out = typename Distance::Output;
    using Index = typename Distance::Index;

    auto nnd_heap =
        r_to_heap<tdoann::HeapAddSymmetric, tdoann::NNDHeap<Out, Index>>(
            nn_idx, nn_dist);
    auto distance = r_to_dist<Distance>(data);
    auto graph_updater = GraphUpdate::create(nnd_heap, distance);
    Progress progress(n_iters, verbose);
    NNDProgress nnd_progress(progress);
    RRand rand;

    tdoann::nnd_build(graph_updater, max_candidates, n_iters, delta, rand,
                      nnd_progress, weighted);

    return heap_to_r(nnd_heap);
  }
};

struct NNDBuildParallel {
  NumericMatrix data;

  std::size_t block_size;
  std::size_t n_threads;
  std::size_t grain_size;

  NNDBuildParallel(NumericMatrix data, std::size_t block_size,
                   std::size_t n_threads, std::size_t grain_size)
      : data(data), block_size(block_size), n_threads(n_threads),
        grain_size(grain_size) {}

  template <typename GraphUpdate, typename Distance, typename Progress,
            typename NNDProgress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false, bool weighted = false)
      -> List {
    using Out = typename Distance::Output;
    using Index = typename Distance::Index;

    auto nnd_heap =
        r_to_heap<tdoann::LockingHeapAddSymmetric, tdoann::NNDHeap<Out, Index>>(
            nn_idx, nn_dist, n_threads, grain_size);
    auto distance = r_to_dist<Distance>(data);
    auto graph_updater = GraphUpdate::create(nnd_heap, distance);
    Progress progress(n_iters, verbose);
    NNDProgress nnd_progress(progress);
    ParallelRand parallel_rand;

    tdoann::nnd_build<RParallel>(graph_updater, max_candidates, n_iters, delta,
                                 nnd_progress, parallel_rand, block_size,
                                 n_threads, grain_size, weighted);

    return heap_to_r(nnd_heap, block_size, n_threads, grain_size);
  }
};

struct NNDQuerySerial {
  NumericMatrix reference;
  NumericMatrix query;

  IntegerMatrix ref_idx;
  NumericMatrix ref_dist;

  NNDQuerySerial(NumericMatrix reference, NumericMatrix query,
                 IntegerMatrix ref_idx, NumericMatrix ref_dist)
      : reference(reference), query(query), ref_idx(ref_idx),
        ref_dist(ref_dist) {}

  template <typename Distance, typename Progress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              std::size_t max_candidates = 50, double epsilon = 0.1,
              std::size_t n_iters = 10, bool verbose = false,
              bool weighted = false) -> List {
    using Out = typename Distance::Output;
    using Index = typename Distance::Index;

    auto nn_heap =
        r_to_heap_missing_ok<tdoann::HeapAddQuery, tdoann::NNHeap<Out, Index>>(
            nn_idx, nn_dist);
    auto distance = r_to_dist<Distance>(reference, query);
    auto ref_idx_vec = r_to_idx<Index>(ref_idx);
    auto ref_dist_vec = r_to_vec<Out>(ref_dist);
    Progress progress(nn_heap.n_points, verbose);

    tdoann::nnd_query(ref_idx_vec, ref_idx.ncol(), ref_dist_vec, nn_heap,
                      distance, max_candidates, epsilon, n_iters, progress);

    return heap_to_r(nn_heap);
  }
};

struct NNDQueryParallel {
  NumericMatrix reference;
  NumericMatrix query;

  IntegerMatrix ref_idx;
  NumericMatrix ref_dist;

  std::size_t n_threads;
  std::size_t grain_size;

  NNDQueryParallel(NumericMatrix reference, NumericMatrix query,
                   IntegerMatrix ref_idx, NumericMatrix ref_dist,
                   std::size_t n_threads, std::size_t grain_size)
      : reference(reference), query(query), ref_idx(ref_idx),
        ref_dist(ref_dist), n_threads(n_threads), grain_size(grain_size) {}

  template <typename Distance, typename Progress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              std::size_t max_candidates = 50, double epsilon = 0.1,
              std::size_t n_iters = 10, bool verbose = false) -> List {
    using Out = typename Distance::Output;
    using Index = typename Distance::Index;

    auto nn_heap =
        r_to_heap_missing_ok<tdoann::HeapAddQuery, tdoann::NNHeap<Out, Index>>(
            nn_idx, nn_dist);
    auto distance = r_to_dist<Distance>(reference, query);
    auto ref_idx_vec = r_to_idx<Index>(ref_idx);
    auto ref_dist_vec = r_to_vec<Out>(ref_dist);
    Progress progress(1, verbose);

    tdoann::nnd_query<RParallel>(ref_idx_vec, ref_idx.ncol(), ref_dist_vec,
                                 nn_heap, distance, max_candidates, epsilon,
                                 n_iters, progress, n_threads, grain_size);

    return heap_to_r(nn_heap, 1024, n_threads, grain_size);
  }
};

// [[Rcpp::export]]
List nn_descent(NumericMatrix data, IntegerMatrix nn_idx, NumericMatrix nn_dist,
                const std::string &metric = "euclidean",
                std::size_t max_candidates = 50, std::size_t n_iters = 10,
                double delta = 0.001, bool low_memory = true,
                std::size_t block_size = 16384, std::size_t n_threads = 0,
                std::size_t grain_size = 1, bool verbose = false,
                const std::string &progress = "bar", bool weighted = false) {
  DISPATCH_ON_DISTANCES(NND_BUILD_UPDATER);
}

// [[Rcpp::export]]
List nn_descent_query(NumericMatrix reference, IntegerMatrix reference_idx,
                      NumericMatrix reference_dist, NumericMatrix query,
                      IntegerMatrix nn_idx, NumericMatrix nn_dist,
                      const std::string &metric = "euclidean",
                      std::size_t max_candidates = 50, double epsilon = 0.1,
                      std::size_t n_iters = 10, std::size_t n_threads = 0,
                      std::size_t grain_size = 1, bool verbose = false) {
  DISPATCH_ON_QUERY_DISTANCES(NND_QUERY_UPDATER)
}

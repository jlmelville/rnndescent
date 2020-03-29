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

#include "rnn_candidatepriority.h"
#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_rng.h"
#include "rnn_rtoheap.h"

using namespace tdoann;
using namespace Rcpp;

#define NND_IMPL()                                                             \
  return nnd_impl                                                              \
      .get_nn<GUFactoryT, Distance, CandidatePriorityFactoryImpl, Progress>(   \
          nn_idx, nn_dist, candidate_priority_factory, max_candidates,         \
          n_iters, delta, verbose);

#define NND_PROGRESS()                                                         \
  if (progress == "bar") {                                                     \
    using Progress = RPProgress;                                               \
    NND_IMPL()                                                                 \
  } else {                                                                     \
    using Progress = HeapSumProgress;                                          \
    NND_IMPL()                                                                 \
  }

#define NND_CANDIDATE_PRIORITY_SERIAL()                                        \
  if (candidate_priority == "random") {                                        \
    using CandidatePriorityFactoryImpl =                                       \
        CandidatePriorityFactory<CandidatePriorityRandomSerial>;               \
    CandidatePriorityFactoryImpl candidate_priority_factory;                   \
    NND_PROGRESS()                                                             \
  } else if (candidate_priority == "distance") {                               \
    using CandidatePriorityFactoryImpl =                                       \
        CandidatePriorityFactory<CandidatePriorityLowDistance>;                \
    CandidatePriorityFactoryImpl candidate_priority_factory;                   \
    NND_PROGRESS()                                                             \
  } else if (candidate_priority == "highdistance") {                           \
    using CandidatePriorityFactoryImpl =                                       \
        CandidatePriorityFactory<CandidatePriorityHighDistance>;               \
    CandidatePriorityFactoryImpl candidate_priority_factory;                   \
    NND_PROGRESS()                                                             \
  } else {                                                                     \
    stop("Unknown candidate priority '%s'", candidate_priority);               \
  }

#define NND_CANDIDATE_PRIORITY_PARALLEL()                                      \
  if (candidate_priority == "random") {                                        \
    using CandidatePriorityFactoryImpl =                                       \
        CandidatePriorityFactory<CandidatePriorityRandomParallel>;             \
    CandidatePriorityFactoryImpl candidate_priority_factory(pseed());          \
    NND_PROGRESS()                                                             \
  } else if (candidate_priority == "distance") {                               \
    using CandidatePriorityFactoryImpl =                                       \
        CandidatePriorityFactory<CandidatePriorityLowDistance>;                \
    CandidatePriorityFactoryImpl candidate_priority_factory;                   \
    NND_PROGRESS()                                                             \
  } else if (candidate_priority == "highdistance") {                           \
    using CandidatePriorityFactoryImpl =                                       \
        CandidatePriorityFactory<CandidatePriorityHighDistance>;               \
    CandidatePriorityFactoryImpl candidate_priority_factory;                   \
    NND_PROGRESS()                                                             \
  } else {                                                                     \
    stop("Unknown candidate priority '%s'", candidate_priority);               \
  }

#define NND_BUILD_UPDATER()                                                    \
  if (n_threads > 0) {                                                         \
    using NNDImpl = NNDBuildParallel;                                          \
    NNDImpl nnd_impl(data, n_threads, block_size, grain_size);                 \
    if (low_memory) {                                                          \
      using GUFactoryT = GUFactory<BatchGraphUpdater>;                         \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    } else {                                                                   \
      using GUFactoryT = GUFactory<BatchGraphUpdaterHiMem>;                    \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDBuildSerial;                                            \
    NNDImpl nnd_impl(data);                                                    \
    if (low_memory) {                                                          \
      using GUFactoryT = GUFactory<SerialGraphUpdater>;                        \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    } else {                                                                   \
      using GUFactoryT = GUFactory<SerialGraphUpdaterHiMem>;                   \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    }                                                                          \
  }

#define NND_QUERY_UPDATER()                                                    \
  if (n_threads > 0) {                                                         \
    using NNDImpl = NNDQueryParallel;                                          \
    NNDImpl nnd_impl(reference, query, reference_idx, n_threads, block_size,   \
                     grain_size);                                              \
    if (low_memory) {                                                          \
      using GUFactoryT = GUFactory<QueryBatchGraphUpdater>;                    \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    } else {                                                                   \
      using GUFactoryT = GUFactory<QueryBatchGraphUpdaterHiMem>;               \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDQuerySerial;                                            \
    NNDImpl nnd_impl(reference, query, reference_idx);                         \
    if (low_memory) {                                                          \
      using GUFactoryT = GUFactory<QuerySerialGraphUpdater>;                   \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    } else {                                                                   \
      using GUFactoryT = GUFactory<QuerySerialGraphUpdaterHiMem>;              \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    }                                                                          \
  }

struct NNDBuildSerial {
  NumericMatrix data;

  NNDBuildSerial(NumericMatrix data) : data(data) {}

  template <typename GUFactoryT, typename Distance,
            typename CandidatePriorityFactoryImpl, typename Progress>
  List get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              CandidatePriorityFactoryImpl &candidate_priority_factory,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false) {
    auto data_vec = r2dvt<Distance>(data);
    auto init_nn = r_to_graph(nn_idx, nn_dist, nn_idx.nrow() - 1);

    auto result = nnd_build<Distance, GUFactoryT, Progress>(
        data_vec, data.ncol(), init_nn, max_candidates, n_iters,
        candidate_priority_factory, delta, verbose);

    return graph_to_r(result);
  }
};

struct NNDBuildParallel {
  NumericMatrix data;

  std::size_t n_threads;
  std::size_t block_size;
  std::size_t grain_size;

  NNDBuildParallel(NumericMatrix data, std::size_t n_threads,
                   std::size_t block_size, std::size_t grain_size)
      : data(data), n_threads(n_threads), block_size(block_size),
        grain_size(grain_size) {}

  template <typename GUFactoryT, typename Distance,
            typename CandidatePriorityFactoryImpl, typename Progress>
  List get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              CandidatePriorityFactoryImpl &candidate_priority_factory,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false) {
    auto data_vec = r2dvt<Distance>(data);
    auto init_nn = r_to_graph(nn_idx, nn_dist, nn_idx.nrow() - 1);

    auto result = nnd_build_parallel<Distance, GUFactoryT, Progress, RParallel>(
        data_vec, data.ncol(), init_nn, max_candidates, n_iters,
        candidate_priority_factory, delta, n_threads, block_size, grain_size,
        verbose);

    return graph_to_r(result);
  }
};

struct NNDQuerySerial {
  NumericMatrix reference;
  NumericMatrix query;

  IntegerMatrix ref_idx;
  std::size_t n_ref_points;

  NNDQuerySerial(NumericMatrix reference, NumericMatrix query,
                 IntegerMatrix ref_idx)
      : reference(reference), query(query), ref_idx(ref_idx),
        n_ref_points(ref_idx.nrow()) {}

  template <typename GUFactoryT, typename Distance,
            typename CandidatePriorityFactoryImpl, typename Progress>
  List get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              CandidatePriorityFactoryImpl &candidate_priority_factory,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false) {
    std::size_t n_points = nn_idx.nrow();
    std::size_t n_nbrs = nn_idx.ncol();
    double tol = delta * n_nbrs * n_points;

    auto distance = create_query_distance<Distance>(reference, query);
    NeighborHeap current_graph(nn_idx.nrow(), nn_idx.ncol());

    r_to_heap_serial<HeapAddQuery>(current_graph, nn_idx, nn_dist, 1000,
                                   n_ref_points - 1);

    auto ref_idx_vec = as<std::vector<std::size_t>>(transpose(ref_idx));

    nnd_query<GUFactoryT, Progress>(distance, current_graph, ref_idx_vec,
                                    n_ref_points, max_candidates, n_iters,
                                    candidate_priority_factory, tol, verbose);

    return heap_to_r(current_graph);
  }
};

struct NNDQueryParallel {
  NumericMatrix reference;
  NumericMatrix query;

  IntegerMatrix ref_idx;
  std::size_t n_ref_points;
  std::size_t n_threads;
  std::size_t block_size;
  std::size_t grain_size;

  NNDQueryParallel(NumericMatrix reference, NumericMatrix query,
                   IntegerMatrix ref_idx, std::size_t n_threads,
                   std::size_t block_size, std::size_t grain_size)
      : reference(reference), query(query), ref_idx(ref_idx),
        n_ref_points(ref_idx.nrow()), n_threads(n_threads),
        block_size(block_size), grain_size(grain_size) {}

  template <typename GUFactoryT, typename Distance,
            typename CandidatePriorityFactoryImpl, typename Progress>
  List get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              CandidatePriorityFactoryImpl &candidate_priority_factory,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false) {
    std::size_t n_points = nn_idx.nrow();
    std::size_t n_nbrs = nn_idx.ncol();
    double tol = delta * n_nbrs * n_points;

    auto distance = create_query_distance<Distance>(reference, query);
    NeighborHeap current_graph(nn_idx.nrow(), nn_idx.ncol());
    r_to_heap_parallel<HeapAddQuery>(current_graph, nn_idx, nn_dist, n_threads,
                                     block_size, grain_size, n_ref_points - 1);

    auto ref_idx_vec = as<std::vector<std::size_t>>(transpose(ref_idx));

    nnd_query_parallel<GUFactoryT, Progress, RParallel>(
        distance, current_graph, ref_idx_vec, n_ref_points, max_candidates,
        n_iters, candidate_priority_factory, tol, n_threads, block_size,
        grain_size, verbose);

    return heap_to_r(current_graph);
  }
};

// [[Rcpp::export]]
List nn_descent(NumericMatrix data, IntegerMatrix nn_idx, NumericMatrix nn_dist,
                const std::string &metric = "euclidean",
                std::size_t max_candidates = 50, std::size_t n_iters = 10,
                double delta = 0.001, bool low_memory = true,
                const std::string &candidate_priority = "random",
                std::size_t n_threads = 0, std::size_t block_size = 16384,
                std::size_t grain_size = 1, bool verbose = false,
                const std::string &progress = "bar") {
  DISPATCH_ON_DISTANCES(NND_BUILD_UPDATER);
}

// [[Rcpp::export]]
List nn_descent_query(NumericMatrix reference, IntegerMatrix reference_idx,
                      NumericMatrix query, IntegerMatrix nn_idx,
                      NumericMatrix nn_dist,
                      const std::string &metric = "euclidean",
                      std::size_t max_candidates = 50, std::size_t n_iters = 10,
                      double delta = 0.001, bool low_memory = true,
                      const std::string &candidate_priority = "random",
                      std::size_t n_threads = 0, std::size_t block_size = 16384,
                      std::size_t grain_size = 1, bool verbose = false,
                      const std::string &progress = "bar") {
  DISPATCH_ON_QUERY_DISTANCES(NND_QUERY_UPDATER)
}

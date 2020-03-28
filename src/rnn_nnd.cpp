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
  return nn_descent_impl<NNDImpl, GUFactoryT, Distance,                        \
                         CandidatePriorityFactoryImpl, Progress>(              \
      distance, nn_idx, nn_dist, nnd_impl, candidate_priority_factory,         \
      max_candidates, n_iters, delta, verbose);

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

#define NND_UPDATER()                                                          \
  auto distance = create_build_distance<Distance>(data);                       \
  if (n_threads > 0) {                                                         \
    using NNDImpl = NNDParallel;                                               \
    NNDImpl nnd_impl(n_threads, block_size, grain_size);                       \
    if (low_memory) {                                                          \
      using GUFactoryT = GUFactory<BatchGraphUpdater>;                         \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    } else {                                                                   \
      using GUFactoryT = GUFactory<BatchGraphUpdaterHiMem>;                    \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDSerial;                                                 \
    NNDImpl nnd_impl;                                                          \
    if (low_memory) {                                                          \
      using GUFactoryT = GUFactory<SerialGraphUpdater>;                        \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    } else {                                                                   \
      using GUFactoryT = GUFactory<SerialGraphUpdaterHiMem>;                   \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    }                                                                          \
  }

#define NND_QUERY_UPDATER()                                                    \
  auto distance = create_query_distance<Distance>(reference, query);           \
  if (n_threads > 0) {                                                         \
    using NNDImpl = NNDQueryParallel;                                          \
    NNDImpl nnd_impl(reference_idx, n_threads, block_size, grain_size);        \
    if (low_memory) {                                                          \
      using GUFactoryT = GUFactory<QueryBatchGraphUpdater>;                    \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    } else {                                                                   \
      using GUFactoryT = GUFactory<QueryBatchGraphUpdaterHiMem>;               \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDQuerySerial;                                            \
    NNDImpl nnd_impl(reference_idx);                                           \
    if (low_memory) {                                                          \
      using GUFactoryT = GUFactory<QuerySerialGraphUpdater>;                   \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    } else {                                                                   \
      using GUFactoryT = GUFactory<QuerySerialGraphUpdaterHiMem>;              \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    }                                                                          \
  }

struct NNDSerial {
  template <typename Progress, typename GUFactoryT,
            typename CandidatePriorityFactoryImpl, typename Distance>
  void build_knn(Distance &distance, NeighborHeap &current_graph,
                 std::size_t max_candidates, std::size_t n_iters, double tol,
                 CandidatePriorityFactoryImpl &candidate_priority_factory,
                 bool verbose) {

    Progress progress(current_graph, n_iters, verbose);
    auto graph_updater = GUFactoryT::create(current_graph, distance);

    nnd_full(current_graph, graph_updater, max_candidates, n_iters,
             candidate_priority_factory, progress, tol, verbose);
  }
  void create_heap(NeighborHeap &current_graph, IntegerMatrix nn_idx,
                   NumericMatrix nn_dist) {
    r_to_heap_serial<HeapAddSymmetric>(current_graph, nn_idx, nn_dist, 1000,
                                       current_graph.n_points - 1);
  }
};

struct NNDParallel {
  std::size_t n_threads;
  std::size_t block_size;
  std::size_t grain_size;

  NNDParallel(std::size_t n_threads, std::size_t block_size,
              std::size_t grain_size)
      : n_threads(n_threads), block_size(block_size), grain_size(grain_size) {}

  template <typename Progress, typename GUFactoryT,
            typename CandidatePriorityFactoryImpl, typename Distance>
  void build_knn(Distance &distance, NeighborHeap &current_graph,
                 std::size_t max_candidates, std::size_t n_iters, double tol,
                 CandidatePriorityFactoryImpl &candidate_priority_factory,
                 bool verbose) {

    Progress progress(current_graph, n_iters, verbose);
    auto graph_updater = GUFactoryT::create(current_graph, distance);

    nnd_parallel<Distance, RParallel>(
        current_graph, graph_updater, max_candidates, n_iters,
        candidate_priority_factory, progress, tol, n_threads, block_size,
        grain_size, verbose);
  }

  void create_heap(NeighborHeap &current_graph, IntegerMatrix nn_idx,
                   NumericMatrix nn_dist) {
    r_to_heap_parallel<LockingHeapAddSymmetric>(
        current_graph, nn_idx, nn_dist, n_threads, block_size, grain_size,
        current_graph.n_points - 1);
  }
};

struct NNDQuerySerial {
  IntegerMatrix ref_idx;
  std::size_t n_ref_points;

  NNDQuerySerial(IntegerMatrix ref_idx)
      : ref_idx(ref_idx), n_ref_points(ref_idx.nrow()) {}

  template <typename Progress, typename GUFactoryT,
            typename CandidatePriorityFactoryImpl, typename Distance>
  void build_knn(Distance &distance, NeighborHeap &current_graph,
                 std::size_t max_candidates, std::size_t n_iters, double tol,
                 CandidatePriorityFactoryImpl &candidate_priority_factory,
                 bool verbose) {

    auto ref_idx_vec = as<std::vector<std::size_t>>(transpose(ref_idx));

    Progress progress(current_graph, n_iters, verbose);
    auto graph_updater = GUFactoryT::create(current_graph, distance);

    nnd_query(current_graph, graph_updater, ref_idx_vec, n_ref_points,
              max_candidates, n_iters, candidate_priority_factory, progress,
              tol, verbose);
  }
  void create_heap(NeighborHeap &current_graph, IntegerMatrix nn_idx,
                   NumericMatrix nn_dist) {
    r_to_heap_serial<HeapAddQuery>(current_graph, nn_idx, nn_dist, 1000,
                                   n_ref_points - 1);
  }
};

struct NNDQueryParallel {

  IntegerMatrix ref_idx;
  std::size_t n_ref_points;
  std::size_t n_threads;
  std::size_t block_size;
  std::size_t grain_size;

  NNDQueryParallel(IntegerMatrix ref_idx, std::size_t n_threads,
                   std::size_t block_size, std::size_t grain_size)
      : ref_idx(ref_idx), n_ref_points(ref_idx.nrow()), n_threads(n_threads),
        block_size(block_size), grain_size(grain_size) {}

  template <typename Progress, typename GUFactoryT,
            typename CandidatePriorityFactoryImpl, typename Distance>
  void build_knn(Distance &distance, NeighborHeap &current_graph,
                 std::size_t max_candidates, std::size_t n_iters, double tol,
                 CandidatePriorityFactoryImpl &candidate_priority_factory,
                 bool verbose) {
    auto ref_idx_vec = as<std::vector<std::size_t>>(transpose(ref_idx));

    Progress progress(current_graph, n_iters, verbose);
    auto graph_updater = GUFactoryT::create(current_graph, distance);

    nnd_query_parallel<Distance, RParallel>(
        current_graph, graph_updater, ref_idx_vec, n_ref_points, max_candidates,
        n_iters, candidate_priority_factory, progress, tol, n_threads,
        block_size, grain_size, verbose);
  }

  void create_heap(NeighborHeap &current_graph, IntegerMatrix nn_idx,
                   NumericMatrix nn_dist) {
    r_to_heap_parallel<HeapAddQuery>(current_graph, nn_idx, nn_dist, n_threads,
                                     block_size, grain_size, n_ref_points - 1);
  }
};

template <typename NNDImpl, typename GUFactoryT, typename Distance,
          typename CandidatePriorityFactoryImpl, typename Progress>
List nn_descent_impl(Distance &distance, IntegerMatrix nn_idx,
                     NumericMatrix nn_dist, NNDImpl &nnd_impl,
                     CandidatePriorityFactoryImpl &candidate_priority_factory,
                     std::size_t max_candidates = 50, std::size_t n_iters = 10,
                     double delta = 0.001, bool verbose = false) {
  std::size_t n_points = nn_idx.nrow();
  std::size_t n_nbrs = nn_idx.ncol();
  double tol = delta * n_nbrs * n_points;

  NeighborHeap current_graph(n_points, n_nbrs);
  nnd_impl.create_heap(current_graph, nn_idx, nn_dist);

  nnd_impl.template build_knn<Progress, GUFactoryT>(
      distance, current_graph, max_candidates, n_iters, tol,
      candidate_priority_factory, verbose);

  return heap_to_r(current_graph);
}

// [[Rcpp::export]]
List nn_descent(NumericMatrix data, IntegerMatrix nn_idx, NumericMatrix nn_dist,
                const std::string &metric = "euclidean",
                std::size_t max_candidates = 50, std::size_t n_iters = 10,
                double delta = 0.001, bool low_memory = true,
                const std::string &candidate_priority = "random",
                std::size_t n_threads = 0, std::size_t block_size = 16384,
                std::size_t grain_size = 1, bool verbose = false,
                const std::string &progress = "bar") {
  DISPATCH_ON_DISTANCES(NND_UPDATER);
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

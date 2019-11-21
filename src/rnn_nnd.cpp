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

#include "rnn.h"
#include "rnn_nndparallel.h"
#include "rnn_parallel.h"
#include "tdoann/distance.h"
#include "tdoann/graphupdate.h"
#include "tdoann/heap.h"
#include "tdoann/nndescent.h"

using namespace tdoann;

#define NND_IMPL()                                                             \
  return nn_descent_impl<KnnFactory, NNDImpl, GraphUpdater, Distance,          \
                         Progress>(factory, nn_idx, nn_dist, nnd_impl,         \
                                   max_candidates, n_iters, delta, verbose);

#define NND_PROGRESS()                                                         \
  if (progress == "bar") {                                                     \
    using Progress = RPProgress;                                               \
    NND_IMPL()                                                                 \
  } else {                                                                     \
    using Progress = HeapSumProgress;                                          \
    NND_IMPL()                                                                 \
  }

#define NND_UPDATER()                                                          \
  using KnnFactory = KnnBuildFactory<Distance>;                                \
  KnnFactory factory(data);                                                    \
  if (parallelize) {                                                           \
    using NNDImpl = NNDParallel;                                               \
    NNDImpl nnd_impl(block_size, grain_size);                                  \
    if (low_memory) {                                                          \
      using GraphUpdater = BatchGraphUpdater<Distance>;                        \
      NND_PROGRESS()                                                           \
    } else {                                                                   \
      using GraphUpdater = BatchGraphUpdaterHiMem<Distance>;                   \
      NND_PROGRESS()                                                           \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDSerial;                                                 \
    NNDImpl nnd_impl;                                                          \
    if (low_memory) {                                                          \
      using GraphUpdater = SerialGraphUpdater<Distance>;                       \
      NND_PROGRESS()                                                           \
    } else {                                                                   \
      using GraphUpdater = SerialGraphUpdaterHiMem<Distance>;                  \
      NND_PROGRESS()                                                           \
    }                                                                          \
  }

#define NND_QUERY_UPDATER()                                                    \
  using KnnFactory = KnnQueryFactory<Distance>;                                \
  KnnFactory factory(reference, query);                                        \
  if (parallelize) {                                                           \
    using NNDImpl = NNDQueryParallel;                                          \
    NNDImpl nnd_impl(reference_idx, block_size, grain_size);                   \
    if (low_memory) {                                                          \
      using GraphUpdater = QuerySerialGraphUpdater<Distance>;                  \
      NND_PROGRESS()                                                           \
    } else {                                                                   \
      using GraphUpdater = QuerySerialGraphUpdater<Distance>;                  \
      NND_PROGRESS()                                                           \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDQuerySerial;                                            \
    NNDImpl nnd_impl(reference_idx);                                           \
    if (low_memory) {                                                          \
      using GraphUpdater = QuerySerialGraphUpdater<Distance>;                  \
      NND_PROGRESS()                                                           \
    } else {                                                                   \
      using GraphUpdater = QuerySerialGraphUpdaterHiMem<Distance>;             \
      NND_PROGRESS()                                                           \
    }                                                                          \
  }

struct NNDSerial {
  template <template <typename> class GraphUpdater, typename Distance,
            typename Progress, typename Rand>
  void
  build_knn(NeighborHeap &current_graph, GraphUpdater<Distance> &graph_updater,
            const std::size_t max_candidates, const std::size_t n_iters,
            Rand &rand, const double tol, Progress &progress, bool verbose) {
    nnd_full(current_graph, graph_updater, max_candidates, n_iters, rand,
             progress, tol, verbose);
  }
  void create_heap(NeighborHeap &current_graph, Rcpp::IntegerMatrix nn_idx,
                   Rcpp::NumericMatrix nn_dist) {
    r_to_heap<HeapAddSymmetric, NeighborHeap>(current_graph, nn_idx, nn_dist,
                                              current_graph.n_points - 1);
  }
};

struct NNDParallel {
  std::size_t block_size;
  std::size_t grain_size;

  NNDParallel(std::size_t block_size, std::size_t grain_size)
      : block_size(block_size), grain_size(grain_size) {}

  template <template <typename> class GraphUpdater, typename Distance,
            typename Progress, typename Rand>
  void
  build_knn(NeighborHeap &current_graph, GraphUpdater<Distance> &graph_updater,
            const std::size_t max_candidates, const std::size_t n_iters,
            Rand &rand, const double tol, Progress &progress, bool verbose) {
    nnd_parallel(current_graph, graph_updater, max_candidates, n_iters, rand,
                 progress, tol, block_size, grain_size, verbose);
  }

  void create_heap(NeighborHeap &current_graph, Rcpp::IntegerMatrix nn_idx,
                   Rcpp::NumericMatrix nn_dist) {
    r_to_heap_parallel<HeapAddSymmetric, NeighborHeap>(
        current_graph, nn_idx, nn_dist, block_size, grain_size,
        nn_idx.nrow() - 1);
  }
};

struct NNDQuerySerial {
  Rcpp::IntegerMatrix ref_idx;
  const std::size_t n_ref_points;

  NNDQuerySerial(Rcpp::IntegerMatrix ref_idx)
      : ref_idx(ref_idx), n_ref_points(ref_idx.nrow()) {}

  template <template <typename> class GraphUpdater, typename Distance,
            typename Progress, typename Rand>
  void
  build_knn(NeighborHeap &current_graph, GraphUpdater<Distance> &graph_updater,
            const std::size_t max_candidates, const std::size_t n_iters,
            Rand &rand, const double tol, Progress &progress, bool verbose) {

    auto ref_idx_vec =
        Rcpp::as<std::vector<std::size_t>>(Rcpp::transpose(ref_idx));

    nnd_query(current_graph, graph_updater, ref_idx_vec, n_ref_points,
              max_candidates, n_iters, rand, progress, tol, verbose);
  }
  void create_heap(NeighborHeap &current_graph, Rcpp::IntegerMatrix nn_idx,
                   Rcpp::NumericMatrix nn_dist) {
    r_to_heap<HeapAddQuery>(current_graph, nn_idx, nn_dist, n_ref_points - 1);
  }
};

struct NNDQueryParallel {

  Rcpp::IntegerMatrix ref_idx;
  const std::size_t n_ref_points;
  std::size_t block_size;
  std::size_t grain_size;

  NNDQueryParallel(Rcpp::IntegerMatrix ref_idx, std::size_t block_size,
                   std::size_t grain_size)
      : ref_idx(ref_idx), n_ref_points(ref_idx.nrow()), block_size(block_size),
        grain_size(grain_size) {}

  template <template <typename> class GraphUpdater, typename Distance,
            typename Progress, typename Rand>
  void
  build_knn(NeighborHeap &current_graph, GraphUpdater<Distance> &graph_updater,
            const std::size_t max_candidates, const std::size_t n_iters,
            Rand &rand, const double tol, Progress &progress, bool verbose) {
    auto ref_idx_vec =
        Rcpp::as<std::vector<std::size_t>>(Rcpp::transpose(ref_idx));

    nnd_query_parallel(current_graph, graph_updater, ref_idx_vec, n_ref_points,
                       max_candidates, n_iters, rand, progress, tol, block_size,
                       grain_size, verbose);
  }

  void create_heap(NeighborHeap &current_graph, Rcpp::IntegerMatrix nn_idx,
                   Rcpp::NumericMatrix nn_dist) {
    r_to_heap_parallel<HeapAddQuery>(current_graph, nn_idx, nn_dist, block_size,
                                     grain_size, n_ref_points - 1);
  }
};

template <typename KnnFactory, typename NNDImpl, typename GraphUpdater,
          typename Distance, typename Progress>
Rcpp::List nn_descent_impl(KnnFactory &factory, Rcpp::IntegerMatrix nn_idx,
                           Rcpp::NumericMatrix nn_dist, NNDImpl &nnd_impl,
                           const std::size_t max_candidates = 50,
                           const std::size_t n_iters = 10,
                           const double delta = 0.001, bool verbose = false) {
  const std::size_t n_points = nn_idx.nrow();
  const std::size_t n_nbrs = nn_idx.ncol();
  const double tol = delta * n_nbrs * n_points;

  auto distance = factory.create_distance();

  NeighborHeap current_graph(n_points, n_nbrs);
  nnd_impl.create_heap(current_graph, nn_idx, nn_dist);
  GraphUpdater graph_updater(current_graph, distance);

  Progress progress(current_graph, n_iters, verbose);
  RRand rand;

  nnd_impl.build_knn(current_graph, graph_updater, max_candidates, n_iters,
                     rand, tol, progress, verbose);

  return heap_to_r(current_graph);
}

// [[Rcpp::export]]
Rcpp::List nn_descent(Rcpp::NumericMatrix data, Rcpp::IntegerMatrix nn_idx,
                      Rcpp::NumericMatrix nn_dist,
                      const std::string metric = "euclidean",
                      const std::size_t max_candidates = 50,
                      const std::size_t n_iters = 10,
                      const double delta = 0.001, bool low_memory = true,
                      bool parallelize = false, std::size_t block_size = 16384,
                      std::size_t grain_size = 1, bool verbose = false,
                      const std::string &progress = "bar") {
  DISPATCH_ON_DISTANCES(NND_UPDATER);
}

// [[Rcpp::export]]
Rcpp::List
nn_descent_query(Rcpp::NumericMatrix reference,
                 Rcpp::IntegerMatrix reference_idx, Rcpp::NumericMatrix query,
                 Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
                 const std::string metric = "euclidean",
                 const std::size_t max_candidates = 50,
                 const std::size_t n_iters = 10, const double delta = 0.001,
                 bool low_memory = true, bool parallelize = false,
                 std::size_t block_size = 16384, std::size_t grain_size = 1,
                 bool verbose = false, const std::string &progress = "bar") {
  DISPATCH_ON_DISTANCES(NND_QUERY_UPDATER)
}

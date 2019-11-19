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
#include "tdoann/distance.h"
#include "tdoann/graphupdate.h"
#include "tdoann/heap.h"
#include "tdoann/nndescent.h"

using namespace tdoann;

#define NND_IMPL()                                                             \
  return nn_descent_impl<NNDImpl, GraphUpdater, Distance>(                     \
      data, nn_idx, nn_dist, nnd_impl, max_candidates, n_iters, delta,         \
      verbose);

#define NND_UPDATER()                                                          \
  if (parallelize) {                                                           \
    using NNDImpl = NNDParallel;                                               \
    NNDImpl nnd_impl(block_size, grain_size);                                  \
    if (low_memory) {                                                          \
      using GraphUpdater = BatchGraphUpdater<Distance>;                        \
      NND_IMPL()                                                               \
    } else {                                                                   \
      using GraphUpdater = BatchGraphUpdaterHiMem<Distance>;                   \
      NND_IMPL()                                                               \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDSerial;                                                 \
    NNDImpl nnd_impl;                                                          \
    if (low_memory) {                                                          \
      using GraphUpdater = SerialGraphUpdater<Distance>;                       \
      NND_IMPL()                                                               \
    } else {                                                                   \
      using GraphUpdater = SerialGraphUpdaterHiMem<Distance>;                  \
      NND_IMPL()                                                               \
    }                                                                          \
  }

struct NNDSerial {
  template <template <typename> class GraphUpdater, typename Distance,
            typename Progress, typename Rand>
  void
  operator()(NeighborHeap &current_graph, GraphUpdater<Distance> &graph_updater,
             const std::size_t max_candidates, const std::size_t n_iters,
             Rand &rand, const double tol, Progress &progress, bool verbose) {
    nnd_full(current_graph, graph_updater, max_candidates, n_iters, rand,
             progress, tol, verbose);
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
  operator()(NeighborHeap &current_graph, GraphUpdater<Distance> &graph_updater,
             const std::size_t max_candidates, const std::size_t n_iters,
             Rand &rand, const double tol, Progress &progress, bool verbose) {
    nnd_parallel(current_graph, graph_updater, max_candidates, n_iters, rand,
                 progress, tol, block_size, grain_size, verbose);
  }
};

template <typename NNDImpl, typename GraphUpdater, typename Distance>
Rcpp::List nn_descent_impl(Rcpp::NumericMatrix data, Rcpp::IntegerMatrix nn_idx,
                           Rcpp::NumericMatrix nn_dist, NNDImpl &nnd_impl,
                           const std::size_t max_candidates = 50,
                           const std::size_t n_iters = 10,
                           const double delta = 0.001, bool verbose = false) {
  const std::size_t n_points = nn_idx.nrow();
  const std::size_t n_nbrs = nn_idx.ncol();
  const double tol = delta * n_nbrs * n_points;

  KnnBuildFactory<Distance> factory(data);
  auto distance = factory.create_distance();

  NeighborHeap current_graph(n_points, n_nbrs);
  r_to_heap<HeapAddSymmetric, tdoann::NeighborHeap>(
      current_graph, nn_idx, nn_dist, static_cast<int>(n_points - 1));
  GraphUpdater graph_updater(current_graph, distance);

  HeapSumProgress progress(current_graph, n_iters, verbose);
  RRand rand;

  nnd_impl(current_graph, graph_updater, max_candidates, n_iters, rand, tol,
           progress, verbose);

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

#define NND_QUERY_IMPL()                                                       \
  return nn_descent_query_impl<NNDImpl, GraphUpdater, Distance>(               \
      reference, reference_idx, query, nn_idx, nn_dist, nnd_impl,              \
      max_candidates, n_iters, delta, verbose);

#define NND_QUERY_UPDATER()                                                    \
  if (parallelize) {                                                           \
    using NNDImpl = NNDQueryParallel;                                          \
    NNDImpl nnd_impl(block_size, grain_size);                                  \
    if (low_memory) {                                                          \
      using GraphUpdater = QuerySerialGraphUpdater<Distance>;                  \
      NND_QUERY_IMPL()                                                         \
    } else {                                                                   \
      using GraphUpdater = QuerySerialGraphUpdater<Distance>;                  \
      NND_QUERY_IMPL()                                                         \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDQuerySerial;                                            \
    NNDImpl nnd_impl;                                                          \
    if (low_memory) {                                                          \
      using GraphUpdater = QuerySerialGraphUpdater<Distance>;                  \
      NND_QUERY_IMPL()                                                         \
    } else {                                                                   \
      using GraphUpdater = QuerySerialGraphUpdaterHiMem<Distance>;             \
      NND_QUERY_IMPL()                                                         \
    }                                                                          \
  }

struct NNDQuerySerial {
  template <template <typename> class GraphUpdater, typename Distance,
            typename Progress, typename Rand>
  void
  operator()(NeighborHeap &current_graph, GraphUpdater<Distance> &graph_updater,
             const std::vector<std::size_t> &reference_idx_vec,
             const std::size_t n_ref_points, const std::size_t max_candidates,
             const std::size_t n_iters, Rand &rand, const double tol,
             Progress &progress, bool verbose) {
    nnd_query(current_graph, graph_updater, reference_idx_vec, n_ref_points,
              max_candidates, n_iters, rand, progress, tol, verbose);
  }
};

struct NNDQueryParallel {
  std::size_t block_size;
  std::size_t grain_size;

  NNDQueryParallel(std::size_t block_size, std::size_t grain_size)
      : block_size(block_size), grain_size(grain_size) {}

  template <template <typename> class GraphUpdater, typename Distance,
            typename Progress, typename Rand>
  void
  operator()(NeighborHeap &current_graph, GraphUpdater<Distance> &graph_updater,
             const std::vector<std::size_t> &reference_idx_vec,
             const std::size_t n_ref_points, const std::size_t max_candidates,
             const std::size_t n_iters, Rand &rand, const double tol,
             Progress &progress, bool verbose) {
    nnd_query_parallel(current_graph, graph_updater, reference_idx_vec,
                       n_ref_points, max_candidates, n_iters, rand, progress,
                       tol, block_size, grain_size, verbose);
  }
};

template <typename NNDImpl, typename GraphUpdater, typename Distance>
Rcpp::List nn_descent_query_impl(
    Rcpp::NumericMatrix reference, Rcpp::IntegerMatrix reference_idx,
    Rcpp::NumericMatrix query, Rcpp::IntegerMatrix nn_idx,
    Rcpp::NumericMatrix nn_dist, NNDImpl &nnd_impl,
    const std::size_t max_candidates = 50, const std::size_t n_iters = 10,
    const double delta = 0.001, bool verbose = false) {
  const std::size_t n_points = nn_idx.nrow();
  const std::size_t n_nbrs = nn_idx.ncol();
  const std::size_t n_ref_points = reference.nrow();
  const double tol = delta * n_nbrs * n_points;

  reference_idx = Rcpp::transpose(reference_idx);
  auto reference_idx_vec = Rcpp::as<std::vector<std::size_t>>(reference_idx);

  KnnQueryFactory<Distance> factory(reference, query);
  auto distance = factory.create_distance();

  NeighborHeap current_graph(n_points, n_nbrs);
  r_to_heap<HeapAddQuery>(current_graph, nn_idx, nn_dist,
                          static_cast<int>(n_ref_points - 1));
  GraphUpdater graph_updater(current_graph, distance);

  HeapSumProgress progress(current_graph, n_iters, verbose);
  RRand rand;

  nnd_impl(current_graph, graph_updater, reference_idx_vec, n_ref_points,
           max_candidates, n_iters, rand, tol, progress, verbose);

  return heap_to_r(current_graph);
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

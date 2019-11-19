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

#define NND_IMPL(NNDImpl, Distance, GraphUpdater)                              \
  return nn_descent_impl<NNDImpl, GraphUpdater, Distance>(                     \
      data, nn_idx, nn_dist, nnd_impl, max_candidates, n_iters, delta,         \
      verbose);

#define NND_UPDATER(Distance)                                                  \
  if (parallelize) {                                                           \
    using NNDImpl = NNDParallel;                                               \
    NNDImpl nnd_impl(block_size, grain_size);                                  \
    if (low_memory) {                                                          \
      using GraphUpdater = BatchGraphUpdater<Distance>;                        \
      NND_IMPL(NNDImpl, Distance, GraphUpdater)                                \
    } else {                                                                   \
      using GraphUpdater = BatchGraphUpdaterHiMem<Distance>;                   \
      NND_IMPL(NNDImpl, Distance, GraphUpdater)                                \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDSerial;                                                 \
    NNDImpl nnd_impl;                                                          \
    if (low_memory) {                                                          \
      using GraphUpdater = SerialGraphUpdater<Distance>;                       \
      NND_IMPL(NNDImpl, Distance, GraphUpdater)                                \
    } else {                                                                   \
      using GraphUpdater = SerialGraphUpdaterHiMem<Distance>;                  \
      NND_IMPL(NNDImpl, Distance, GraphUpdater)                                \
    }                                                                          \
  }

struct NNDSerial {
  template <template <typename> class GraphUpdater, typename Distance,
            typename Rand>
  void operator()(NeighborHeap &current_graph,
                  GraphUpdater<Distance> &graph_updater,
                  const std::size_t max_candidates, const std::size_t n_iters,
                  Rand &rand, const double tol, bool verbose) {
    HeapSumProgress progress(current_graph, n_iters, verbose);
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
            typename Rand>
  void operator()(NeighborHeap &current_graph,
                  GraphUpdater<Distance> &graph_updater,
                  const std::size_t max_candidates, const std::size_t n_iters,
                  Rand &rand, const double tol, bool verbose) {
    HeapSumProgress progress(current_graph, n_iters, verbose);
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
  const std::size_t ndim = data.ncol();

  data = Rcpp::transpose(data);
  auto data_vec = Rcpp::as<std::vector<typename Distance::in_type>>(data);
  RRand rand;
  Distance distance(data_vec, ndim);
  NeighborHeap current_graph(n_points, n_nbrs);
  r_to_heap<HeapAddSymmetric, tdoann::NeighborHeap>(
      current_graph, nn_idx, nn_dist, static_cast<int>(n_points - 1));
  GraphUpdater graph_updater(current_graph, distance);

  const double tol = delta * n_nbrs * n_points;

  nnd_impl(current_graph, graph_updater, max_candidates, n_iters, rand, tol,
           verbose);

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

#define NND_QUERY_IMPL(NNDImpl, Distance, Rand, GraphUpdater)                  \
  return nn_descent_query_impl<NNDImpl, GraphUpdater, Distance, Rand>(         \
      reference, reference_idx, query, nn_idx, nn_dist, nnd_impl,              \
      max_candidates, n_iters, delta, verbose);

#define NND_QUERY_UPDATER(Distance, Rand, low_memory, parallelize)             \
  if (parallelize) {                                                           \
    using NNDImpl = NNDQueryParallel;                                          \
    NNDImpl nnd_impl(block_size, grain_size);                                  \
    if (low_memory) {                                                          \
      using GraphUpdater = QuerySerialGraphUpdater<Distance>;                  \
      NND_QUERY_IMPL(NNDImpl, Distance, Rand, GraphUpdater)                    \
    } else {                                                                   \
      using GraphUpdater = QuerySerialGraphUpdater<Distance>;                  \
      NND_QUERY_IMPL(NNDImpl, Distance, Rand, GraphUpdater)                    \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDQuerySerial;                                            \
    NNDImpl nnd_impl;                                                          \
    if (low_memory) {                                                          \
      using GraphUpdater = QuerySerialGraphUpdater<Distance>;                  \
      NND_QUERY_IMPL(NNDImpl, Distance, Rand, GraphUpdater)                    \
    } else {                                                                   \
      using GraphUpdater = QuerySerialGraphUpdaterHiMem<Distance>;             \
      NND_QUERY_IMPL(NNDImpl, Distance, Rand, GraphUpdater)                    \
    }                                                                          \
  }

struct NNDQuerySerial {
  template <template <typename> class GraphUpdater, typename Distance,
            typename Rand>
  void operator()(NeighborHeap &current_graph,
                  GraphUpdater<Distance> &graph_updater,
                  const std::vector<std::size_t> &reference_idx_vec,
                  const std::size_t n_ref_points,
                  const std::size_t max_candidates, const std::size_t n_iters,
                  Rand &rand, const double tol, bool verbose) {
    HeapSumProgress progress(current_graph, n_iters, verbose);
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
            typename Rand>
  void operator()(NeighborHeap &current_graph,
                  GraphUpdater<Distance> &graph_updater,
                  const std::vector<std::size_t> &reference_idx_vec,
                  const std::size_t n_ref_points,
                  const std::size_t max_candidates, const std::size_t n_iters,
                  Rand &rand, const double tol, bool verbose) {
    HeapSumProgress progress(current_graph, n_iters, verbose);
    nnd_query_parallel(current_graph, graph_updater, reference_idx_vec,
                       n_ref_points, max_candidates, n_iters, rand, progress,
                       tol, block_size, grain_size, verbose);
  }
};

template <typename NNDImpl, typename GraphUpdater, typename Distance,
          typename Rand>
Rcpp::List nn_descent_query_impl(
    Rcpp::NumericMatrix reference, Rcpp::IntegerMatrix reference_idx,
    Rcpp::NumericMatrix query, Rcpp::IntegerMatrix nn_idx,
    Rcpp::NumericMatrix nn_dist, NNDImpl &nnd_impl,
    const std::size_t max_candidates = 50, const std::size_t n_iters = 10,
    const double delta = 0.001, bool verbose = false) {
  const std::size_t n_points = nn_idx.nrow();
  const std::size_t n_nbrs = nn_idx.ncol();
  const std::size_t n_ref_points = reference.nrow();

  const std::size_t ndim = reference.ncol();

  reference = Rcpp::transpose(reference);
  auto reference_vec =
      Rcpp::as<std::vector<typename Distance::in_type>>(reference);
  NeighborHeap current_graph(n_points, n_nbrs);
  r_to_heap<HeapAddQuery>(current_graph, nn_idx, nn_dist,
                          static_cast<int>(n_ref_points - 1));
  reference_idx = Rcpp::transpose(reference_idx);
  auto reference_idx_vec = Rcpp::as<std::vector<std::size_t>>(reference_idx);

  query = Rcpp::transpose(query);
  auto query_vec = Rcpp::as<std::vector<typename Distance::in_type>>(query);

  Rand rand;
  Distance distance(reference_vec, query_vec, ndim);

  GraphUpdater graph_updater(current_graph, distance);
  const double tol = delta * n_nbrs * n_points;

  nnd_impl(current_graph, graph_updater, reference_idx_vec, n_ref_points,
           max_candidates, n_iters, rand, tol, verbose);

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

  if (metric == "euclidean") {
    using Distance = Euclidean<float, float>;
    NND_QUERY_UPDATER(Distance, RRand, low_memory, parallelize)
  } else if (metric == "l2sqr") {
    using Distance = L2Sqr<float, float>;
    NND_QUERY_UPDATER(Distance, RRand, low_memory, parallelize)
  } else if (metric == "cosine") {
    using Distance = Cosine<float, float>;
    NND_QUERY_UPDATER(Distance, RRand, low_memory, parallelize)
  } else if (metric == "manhattan") {
    using Distance = Manhattan<float, float>;
    NND_QUERY_UPDATER(Distance, RRand, low_memory, parallelize)
  } else if (metric == "hamming") {
    using Distance = Hamming<uint8_t, std::size_t>;
    NND_QUERY_UPDATER(Distance, RRand, low_memory, parallelize)
  } else {
    Rcpp::stop("Bad metric: " + metric);
  }
}

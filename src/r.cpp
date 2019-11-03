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

#include "arrayheap.h"
#include "distance.h"
#include "nndescent.h"
#include "rnn.h"
#include "rnnd_parallel.h"
#include "setheap.h"

#define NNDS(DistType, RandType, low_memory, parallelize)                 \
if (parallelize) {                                                        \
  if (low_memory) {                                                       \
    return nn_descent_impl<ArrayHeap,                                     \
                           DistType,                                      \
                           RandType>                                      \
    (data, idx, dist, max_candidates, n_iters, delta, true, true,         \
     grain_size, block_size, verbose);                                    \
  }                                                                       \
  else {                                                                  \
    return nn_descent_impl<ArrayHeap,                                     \
                           DistType,                                      \
                           RandType>                                      \
      (data, idx, dist, max_candidates, n_iters, delta, false, true,      \
       grain_size, block_size, verbose);                                  \
  }                                                                       \
}                                                                         \
else {                                                                    \
  if (low_memory) {                                                       \
    return nn_descent_impl<ArrayHeap,                                     \
                           DistType,                                      \
                           RandType>                                      \
    (data, idx, dist, max_candidates, n_iters, delta, false, false,       \
     grain_size, block_size, verbose);                                    \
  }                                                                       \
  else {                                                                  \
    return nn_descent_impl<SetHeap,                                       \
                           DistType,                                      \
                           RandType>                                      \
    (data, idx, dist, max_candidates, n_iters, delta, false, false,       \
     grain_size, block_size, verbose);                                    \
  }                                                                       \
}

#define NNDR(DistType, low_memory, use_fast_rand, parallelize)    \
if (use_fast_rand) {                                              \
  NNDS(DistType, TauRand, low_memory, parallelize)                \
}                                                                 \
else {                                                            \
  NNDS(DistType, RRand, low_memory, parallelize)                  \
}

template <template<typename> class Heap,
          typename Distance,
          typename Rand>
Rcpp::List nn_descent_impl(
    Rcpp::NumericMatrix data,
    Rcpp::IntegerMatrix idx,
    Rcpp::NumericMatrix dist,
    const std::size_t max_candidates = 50,
    const std::size_t n_iters = 10,
    const double delta = 0.001,
    bool low_memory = false,
    bool parallelize = false,
    std::size_t grain_size = 1,
    std::size_t block_size = 16384,
    bool verbose = false) {
  const std::size_t npoints = idx.nrow();
  const std::size_t nnbrs = idx.ncol();

  const std::size_t ndim = data.ncol();
  data = Rcpp::transpose(data);
  auto data_vec = Rcpp::as<std::vector<typename Distance::in_type>>(data);

  Rand rand;
  Distance distance(data_vec, ndim);
  Heap<Distance> current_graph = r_to_heap<Heap, Distance>(distance, idx, dist);
  HeapSumProgress progress(current_graph.neighbor_heap, n_iters, verbose);

  const double tol = delta * nnbrs * npoints;
  if (parallelize) {
    if (low_memory) {
      GraphUpdater graph_updater(current_graph.neighbor_heap);

      nnd_parallel(current_graph, max_candidates, n_iters, graph_updater, rand,
                   progress, tol, grain_size, block_size, verbose);
    }
    else {
      GraphUpdaterHiMem graph_updater(current_graph.neighbor_heap);

      nnd_parallel(current_graph, max_candidates, n_iters, graph_updater, rand,
                   progress, tol, grain_size, block_size, verbose);
    }
  }
  else {
    nnd_full(current_graph, max_candidates, n_iters, rand, progress, tol,
             verbose);
  }

  return heap_to_r(current_graph.neighbor_heap);
}

// [[Rcpp::export]]
Rcpp::List nn_descent(
    Rcpp::NumericMatrix data,
    Rcpp::IntegerMatrix idx,
    Rcpp::NumericMatrix dist,
    const std::string metric = "euclidean",
    const std::size_t max_candidates = 50,
    const std::size_t n_iters = 10,
    const double delta = 0.001,
    bool low_memory = true,
    bool fast_rand = false,
    bool parallelize = false,
    std::size_t grain_size = 1,
    std::size_t block_size = 16384,
    bool verbose = false) {

  if (metric == "euclidean") {
    using DistType = Euclidean<float, float>;
    NNDR(DistType, low_memory, fast_rand, parallelize)
  }
  else if (metric == "l2") {
    using DistType = L2<float, float>;
    NNDR(DistType, low_memory, fast_rand, parallelize)
  }
  else if (metric == "cosine") {
    using DistType = Cosine<float, float>;
    NNDR(DistType, low_memory, fast_rand, parallelize)
  }
  else if (metric == "manhattan") {
    using DistType = Manhattan<float, float>;
    NNDR(DistType, low_memory, fast_rand, parallelize)
  }
  else if (metric == "hamming") {
    using DistType = Hamming<uint8_t, std::size_t>;
    NNDR(DistType, low_memory, fast_rand, parallelize)
  }
  else {
    Rcpp::stop("Bad metric: " + metric);
  }
}

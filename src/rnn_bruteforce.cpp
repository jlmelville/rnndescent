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

#include "tdoann/bruteforce.h"
#include "tdoann/heap.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"

using namespace Rcpp;
using namespace tdoann;

#define BRUTE_FORCE_BUILD()                                                    \
  auto distance = create_build_distance<Distance>(data);                       \
  return rnn_brute_force_impl<NNBruteForceBuild, Distance, RPProgress,         \
                              RParallel>(distance, k, n_threads, block_size,   \
                                         grain_size, verbose);

#define BRUTE_FORCE_QUERY()                                                    \
  auto distance = create_query_distance<Distance>(reference, query);           \
  return rnn_brute_force_impl<NNBruteForceQuery, Distance, RPProgress,         \
                              RParallel>(distance, k, n_threads, block_size,   \
                                         grain_size, verbose);

template <typename Distance, typename Progress, typename Parallel>
struct NNBruteForceBuild {
  static SimpleNeighborHeap calculate(Distance distance, std::size_t k,
                                      std::size_t n_threads = 0,
                                      std::size_t block_size = 64,
                                      std::size_t grain_size = 1,
                                      bool verbose = false) {

    if (n_threads > 0) {
      return nnbf_parallel<Distance, Progress, Parallel>(
          distance, k, n_threads, block_size, grain_size, verbose);
    } else {
      return nnbf<Distance, Progress>(distance, k, verbose);
    }
  }
};

template <typename Distance, typename Progress, typename Parallel>
struct NNBruteForceQuery {
  static SimpleNeighborHeap calculate(Distance distance, std::size_t k,
                                      std::size_t n_threads = 0,
                                      std::size_t block_size = 64,
                                      std::size_t grain_size = 1,
                                      bool verbose = false) {

    if (n_threads > 0) {
      return nnbf_parallel_query<Distance, Progress, Parallel>(
          distance, k, n_threads, block_size, grain_size, verbose);
    } else {
      return nnbf_query<Distance, Progress>(distance, k, verbose);
    }
  }
};

template <template <typename, typename, typename> class NNBruteForce,
          typename Distance, typename Progress, typename Parallel>
auto rnn_brute_force_impl(Distance &distance, std::size_t k,
                          std::size_t n_threads = 0,
                          std::size_t block_size = 64,
                          std::size_t grain_size = 1, bool verbose = false)
    -> List {

  SimpleNeighborHeap neighbor_heap =
      NNBruteForce<Distance, Progress, Parallel>::calculate(
          distance, k, n_threads, block_size, grain_size, verbose);
  return heap_to_r(neighbor_heap);
}

// [[Rcpp::export]]
List rnn_brute_force(NumericMatrix data, int k,
                     const std::string &metric = "euclidean",
                     std::size_t n_threads = 0, std::size_t block_size = 64,
                     std::size_t grain_size = 1, bool verbose = false){
    DISPATCH_ON_DISTANCES(BRUTE_FORCE_BUILD)}

// [[Rcpp::export]]
List rnn_brute_force_query(NumericMatrix reference, NumericMatrix query, int k,
                           const std::string &metric = "euclidean",
                           std::size_t n_threads = 0,
                           std::size_t block_size = 64,
                           std::size_t grain_size = 1, bool verbose = false) {
  DISPATCH_ON_QUERY_DISTANCES(BRUTE_FORCE_QUERY)
}

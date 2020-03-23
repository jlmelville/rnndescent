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
  return rnn_brute_force_impl<Distance>(data, k, n_threads, block_size,        \
                                        grain_size, verbose);

#define BRUTE_FORCE_QUERY()                                                    \
  return rnn_brute_force_query_impl<Distance>(x, y, k, n_threads, block_size,  \
                                              grain_size, verbose);

template <typename Distance>
auto rnn_brute_force_impl(NumericMatrix data, std::size_t k,
                          std::size_t n_threads = 0,
                          std::size_t block_size = 64,
                          std::size_t grain_size = 1, bool verbose = false)
    -> List {

  auto distance = create_build_distance<Distance>(data);

  if (n_threads > 0) {
    SimpleNeighborHeap neighbor_heap =
        nnbf_parallel<Distance, RPProgress, RParallel>(
            distance, k, n_threads, block_size, grain_size, verbose);
    return heap_to_r(neighbor_heap);

  } else {
    SimpleNeighborHeap neighbor_heap =
        nnbf<Distance, RPProgress>(distance, k, verbose);
    return heap_to_r(neighbor_heap);
  }
}

template <typename Distance>
auto rnn_brute_force_query_impl(NumericMatrix x, NumericMatrix y, std::size_t k,
                                std::size_t n_threads = 0,
                                std::size_t block_size = 64,
                                std::size_t grain_size = 1,
                                bool verbose = false) -> List {

  auto distance = create_query_distance<Distance>(x, y);

  if (n_threads > 0) {
    SimpleNeighborHeap neighbor_heap =
        nnbf_parallel_query<Distance, RPProgress, RParallel>(
            distance, k, n_threads, block_size, grain_size, verbose);
    return heap_to_r(neighbor_heap);
  } else {
    SimpleNeighborHeap neighbor_heap =
        nnbf_query<Distance, RPProgress>(distance, k, verbose);
    return heap_to_r(neighbor_heap);
  }
}

// [[Rcpp::export]]
List rnn_brute_force(NumericMatrix data, int k,
                     const std::string &metric = "euclidean",
                     std::size_t n_threads = 0, std::size_t block_size = 64,
                     std::size_t grain_size = 1, bool verbose = false){
    DISPATCH_ON_DISTANCES(BRUTE_FORCE_BUILD)}

// [[Rcpp::export]]
List rnn_brute_force_query(NumericMatrix x, NumericMatrix y, int k,
                           const std::string &metric = "euclidean",
                           std::size_t n_threads = 0,
                           std::size_t block_size = 64,
                           std::size_t grain_size = 1, bool verbose = false) {
  DISPATCH_ON_QUERY_DISTANCES(BRUTE_FORCE_QUERY)
}

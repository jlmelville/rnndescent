//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2021 James Melville
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

#include "tdoann/search.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_macros.h"
#include "rnn_progress.h"
#include "rnn_rtoheap.h"

using namespace Rcpp;

#define NN_QUERY_IMPL()                                                        \
  return nn_impl.get_nn<Distance, RPProgress>(nn_idx, nn_dist, epsilon,        \
                                              verbose);

#define NN_QUERY_UPDATER()                                                     \
  if (n_threads > 0) {                                                         \
    using NNImpl = NNQueryParallel;                                            \
    NNImpl nn_impl(reference, query, reference_graph_list, n_threads);         \
    NN_QUERY_IMPL()                                                            \
  } else {                                                                     \
    using NNImpl = NNQuerySerial;                                              \
    NNImpl nn_impl(reference, query, reference_graph_list);                    \
    NN_QUERY_IMPL()                                                            \
  }

struct NNQuerySerial {
  NumericMatrix reference;
  NumericMatrix query;

  List reference_graph_list;

  NNQuerySerial(NumericMatrix reference, NumericMatrix query,
                List reference_graph_list)
      : reference(reference), query(query),
        reference_graph_list(reference_graph_list) {}

  template <typename Distance, typename Progress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist, double epsilon = 0.1,
              bool verbose = false) -> List {
    using Out = typename Distance::Output;
    using Index = typename Distance::Index;

    auto nn_heap =
        r_to_heap_missing_ok<tdoann::HeapAddQuery, tdoann::NNHeap<Out, Index>>(
            nn_idx, nn_dist);
    auto distance = r_to_dist<Distance>(reference, query);
    auto reference_graph = r_to_sparse_graph<Distance>(reference_graph_list);
    tdoann::nn_query<Progress>(reference_graph, nn_heap, distance, epsilon,
                               verbose);

    return heap_to_r(nn_heap);
  }
};

struct NNQueryParallel {
  NumericMatrix reference;
  NumericMatrix query;

  List reference_graph_list;

  std::size_t n_threads;

  NNQueryParallel(NumericMatrix reference, NumericMatrix query,
                  List reference_graph_list, std::size_t n_threads)
      : reference(reference), query(query),
        reference_graph_list(reference_graph_list), n_threads(n_threads) {}

  template <typename Distance, typename Progress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist, double epsilon = 0.1,
              bool verbose = false) -> List {
    using Out = typename Distance::Output;
    using Index = typename Distance::Index;

    auto nn_heap =
        r_to_heap_missing_ok<tdoann::HeapAddQuery, tdoann::NNHeap<Out, Index>>(
            nn_idx, nn_dist);
    auto distance = r_to_dist<Distance>(reference, query);
    auto reference_graph = r_to_sparse_graph<Distance>(reference_graph_list);
    tdoann::nn_query<RParallel, Progress>(reference_graph, nn_heap, distance,
                                          epsilon, n_threads, verbose);

    return heap_to_r(nn_heap, n_threads);
  }
};

// [[Rcpp::export]]
List nn_query(NumericMatrix reference, List reference_graph_list,
              NumericMatrix query, IntegerMatrix nn_idx, NumericMatrix nn_dist,
              const std::string &metric = "euclidean", double epsilon = 0.1,
              std::size_t n_threads = 0, bool verbose = false) {
  DISPATCH_ON_QUERY_DISTANCES(NN_QUERY_UPDATER)
}

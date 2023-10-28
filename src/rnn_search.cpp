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

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,modernize-use-trailing-return-type,readability-magic-numbers)

#include <Rcpp.h>

#include "tdoann/search.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_init.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_rtoheap.h"

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

// [[Rcpp::export]]
List nn_query(const NumericMatrix &reference, const List &reference_graph_list,
              const NumericMatrix &query, const IntegerMatrix &nn_idx,
              const NumericMatrix &nn_dist,
              const std::string &metric = "euclidean", double epsilon = 0.1,
              std::size_t n_threads = 0, bool verbose = false) {
  auto distance_ptr = create_query_distance(reference, query, metric);
  using Out = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Output;
  using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;

  const auto reference_graph =
      r_to_sparse_graph<Out, Idx>(reference_graph_list);
  auto nn_heap = r_to_query_heap<tdoann::NNHeap<Out, Idx>>(nn_idx, nn_dist);

  // replace missing data with randomly chosen neighbors so all points have
  // k initial guesses
  fill_random(nn_heap, *distance_ptr, n_threads, verbose);

  RParallelExecutor executor;
  RPProgress progress(verbose);
  tdoann::nn_query(reference_graph, nn_heap, *distance_ptr, epsilon, n_threads,
                   progress, executor);

  return heap_to_r(nn_heap, n_threads, progress, executor);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,modernize-use-trailing-return-type,readability-magic-numbers)

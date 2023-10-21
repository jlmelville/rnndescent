//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2023 James Melville
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

#include <sstream>

#include <Rcpp.h>

#include "rnndescent/random.h"
#include "tdoann/rptree.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

using Rcpp::List;
using Rcpp::NumericMatrix;

// [[Rcpp::export]]
List rp_tree_knn_cpp(const NumericMatrix &data, uint32_t nnbrs,
                     const std::string &metric, unsigned int leaf_size = 30,
                     bool angular = false, std::size_t n_threads = 0,
                     bool verbose = false) {
  auto distance_ptr = create_self_distance(data, metric);
  using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;
  using In = RNN_DEFAULT_IN;

  auto data_vec = r_to_vec<In>(data);

  RPProgress progress(verbose);
  RParallelExecutor executor;
  rnndescent::DQIntSampler<Idx> rng;

  constexpr unsigned int n_trees = 1;
  if (verbose) {
    std::ostringstream oss;
    oss << "Creating RP forest with " << n_trees << " trees";
    progress.log(oss.str());
  }
  tdoann::RPTree<Idx, In> rp_tree =
      tdoann::make_dense_tree(data_vec, data.nrow(), rng, leaf_size, angular);

  std::vector<Idx> leaf_array = get_leaves_from_tree(rp_tree);

  if (verbose) {
    std::ostringstream oss;
    oss << "Creating knn using " << leaf_array.size() / rp_tree.max_leaf_size
        << " leaves";
    progress.log(oss.str());
  }

  auto neighbor_heap =
      init_rp_tree(*distance_ptr, leaf_array, rp_tree.max_leaf_size, nnbrs,
                   n_threads, progress, executor);

  return heap_to_r(neighbor_heap, n_threads, progress, executor);
}

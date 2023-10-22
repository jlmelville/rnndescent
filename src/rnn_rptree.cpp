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
using Rcpp::Rcerr;

// [[Rcpp::export]]
List rp_tree_knn_cpp(const NumericMatrix &data, uint32_t nnbrs,
                     const std::string &metric, unsigned int n_trees,
                     unsigned int leaf_size, bool include_self,
                     std::size_t n_threads = 0, bool verbose = false) {
  auto distance_ptr = create_self_distance(data, metric);
  using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;
  using In = RNN_DEFAULT_IN;

  auto data_vec = r_to_vec<In>(data);

  RParallelExecutor executor;
  rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler> rng_provider;
  constexpr bool angular = false;
  if (verbose) {
    tsmessage() << "Creating RP forest with " << n_trees << " trees"
                << std::endl;
  }

  RPProgress forest_progress(verbose);
  std::vector<tdoann::RPTree<Idx, In>> rp_forest = tdoann::make_forest(
      data_vec, data.nrow(), n_trees, leaf_size, rng_provider, angular,
      n_threads, forest_progress, executor);
  if (verbose) {
    tsmessage() << "Extracting leaf array from forest" << std::endl;
  }
  // Find the largest leaf size in the forest
  auto max_leaf_size_it =
      std::max_element(rp_forest.begin(), rp_forest.end(),
                       [](const tdoann::RPTree<Idx, In> &a, decltype(a) b) {
                         return a.leaf_size < b.leaf_size;
                       });
  Idx max_leaf_size = max_leaf_size_it->leaf_size;
  std::vector<Idx> leaf_array =
      tdoann::get_leaves_from_forest(rp_forest, max_leaf_size);

  if (verbose) {
    tsmessage() << "Creating knn using " << leaf_array.size() / max_leaf_size
                << " leaves" << std::endl;
  }
  RPProgress knn_progress(verbose);
  auto neighbor_heap =
      tdoann::init_rp_tree(*distance_ptr, leaf_array, max_leaf_size, nnbrs,
                           include_self, n_threads, knn_progress, executor);

  return heap_to_r(neighbor_heap, n_threads, knn_progress, executor);
}

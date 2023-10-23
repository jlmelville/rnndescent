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

#include <cmath>

#include <Rcpp.h>

#include "rnndescent/random.h"
#include "tdoann/rptree.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

using Rcpp::_;
using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::Rcerr;

template <typename Idx, typename In>
List search_tree_to_r(const tdoann::SearchTree<Idx, In> &search_tree,
                      std::size_t ndim) {
  // Rcerr << "Search Tree #nodes = " << search_tree.offsets.size() <<
  // std::endl; for (std::size_t i = 0; i < search_tree.offsets.size(); ++i) {
  //   Rcerr << i;
  //   if (std::isnan(search_tree.offsets[i])) {
  //     Rcerr << " leaf ptrs: [" << search_tree.children[i].first << ", "
  //     << search_tree.children[i].second << "): ";
  //     << search_tree.indices[i].size() << " items:";
  //     for (std::size_t j = search_tree.children[i].first;
  //          j < search_tree.children[i].second; ++j) {
  //       Rcerr << " " << search_tree.indices[j] + 1;
  //     }
  //   } else {
  //     Rcerr << "node-> " << search_tree.children[i].first << " "
  //           << search_tree.children[i].second
  //           << " off: " << search_tree.offsets[i] << " hyp:";
  //     for (std::size_t j = 0; j < ndim; j++) {
  //       Rcerr << " " << search_tree.hyperplanes[i][j];
  //     }
  //   }
  //   Rcerr << std::endl;
  // }

  std::size_t n_rows = search_tree.hyperplanes.size();
  NumericVector offsets(search_tree.offsets.size());
  std::size_t n_hyperplane_cols = ndim;
  NumericMatrix hyperplanes(n_rows, n_hyperplane_cols);

  IntegerMatrix children(search_tree.children.size(), 2);
  for (std::size_t i = 0; i < n_rows; ++i) {
    children(i, 0) = search_tree.children[i].first;
    children(i, 1) = search_tree.children[i].second;

    offsets[i] = search_tree.offsets[i];

    for (std::size_t j = 0; j < n_hyperplane_cols; ++j) {
      hyperplanes(i, j) = search_tree.hyperplanes[i][j];
    }
  }

  IntegerVector indices(search_tree.indices.size());
  for (std::size_t i = 0; i < search_tree.indices.size(); ++i) {
    indices[i] = search_tree.indices[i];
  }

  return List::create(_("hyperplanes") = hyperplanes, _("offsets") = offsets,
                      _("children") = children, _("indices") = indices,
                      _("leaf_size") = search_tree.leaf_size);
}

template <typename Idx, typename In>
List search_forest_to_r(
    const std::vector<tdoann::SearchTree<Idx, In>> &search_forest,
    std::size_t ndim) {
  std::size_t n_trees = search_forest.size();
  List forest_list(n_trees);

  for (std::size_t i = 0; i < n_trees; ++i) {
    // Convert each tree to a list of matrices using your existing function
    List tree_list = search_tree_to_r(search_forest[i], ndim);
    forest_list[i] = tree_list;
  }

  return forest_list;
}

// [[Rcpp::export]]
List rp_tree_knn_cpp(const NumericMatrix &data, uint32_t nnbrs,
                     const std::string &metric, unsigned int n_trees,
                     unsigned int leaf_size, bool include_self,
                     std::size_t n_threads = 0, bool verbose = false) {
  auto distance_ptr = create_self_distance(data, metric);
  using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;
  using In = RNN_DEFAULT_IN;

  auto data_vec = r_to_vec<In>(data);

  bool angular = distance_ptr->is_angular();
  RParallelExecutor executor;
  rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler> rng_provider;
  if (verbose) {
    tsmessage() << "Creating" << (angular ? " angular " : "")
                << " RP forest with " << n_trees << " trees" << std::endl;
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

// [[Rcpp::export]]
List rnn_build_search_forest(const NumericMatrix &data,
                             const std::string &metric, unsigned int n_trees,
                             unsigned int leaf_size, std::size_t n_threads = 0,
                             bool verbose = false) {
  auto distance_ptr = create_self_distance(data, metric);
  using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;
  using In = RNN_DEFAULT_IN;

  auto data_vec = r_to_vec<In>(data);

  bool angular = distance_ptr->is_angular();
  RParallelExecutor executor;
  rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler> rng_provider;
  if (verbose) {
    tsmessage() << "Creating" << (angular ? " angular " : "")
                << " RP forest with " << n_trees << " trees" << std::endl;
  }

  const std::size_t ndim = data.nrow();
  RPProgress forest_progress(verbose);
  std::vector<tdoann::RPTree<Idx, In>> rp_forest =
      tdoann::make_forest(data_vec, ndim, n_trees, leaf_size, rng_provider,
                          angular, n_threads, forest_progress, executor);

  auto search_forest = tdoann::convert_rp_forest(rp_forest, data.ncol(), ndim);

  return search_forest_to_r(search_forest, ndim);
}

// [[Rcpp::export]]
List rnn_tree_search(const NumericMatrix &data, uint32_t nnbrs,
                     const std::string &metric, unsigned int idx,
                     unsigned int leaf_size = 30, bool verbose = false) {
  auto distance_ptr = create_self_distance(data, metric);
  using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;
  using In = RNN_DEFAULT_IN;

  auto data_vec = r_to_vec<In>(data);

  bool angular = distance_ptr->is_angular();
  RParallelExecutor executor;
  rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler> rng_provider;

  constexpr std::size_t n_threads = 0;
  constexpr unsigned int n_trees = 1;
  RPProgress forest_progress(verbose);
  std::vector<tdoann::RPTree<Idx, In>> rp_forest = tdoann::make_forest(
      data_vec, data.nrow(), n_trees, leaf_size, rng_provider, angular,
      n_threads, forest_progress, executor);

  if (verbose) {
    for (const auto &tree : rp_forest) {
      Rcerr << "RP Tree #nodes = " << tree.offsets.size() << std::endl;
      for (std::size_t i = 0; i < tree.offsets.size(); ++i) {
        Rcerr << i;
        if (tree.children[i].first == static_cast<std::size_t>(-1)) {
          Rcerr << " leaf " << tree.indices[i].size() << " items:";
          for (std::size_t j = 0; j < tree.indices[i].size(); ++j) {
            Rcerr << " " << tree.indices[i][j] + 1;
          }
        } else {
          Rcerr << " node  " << tree.offsets[i]
                << " children: " << tree.children[i].first << " "
                << tree.children[i].second;
        }
        Rcerr << std::endl;
      }
    }
  }

  std::vector<std::pair<std::size_t, std::size_t>> children;
  std::vector<Idx> indices;
  const std::size_t ndim = data.nrow();
  tdoann::SearchTree<Idx, In> search_tree =
      tdoann::convert_tree_format(rp_forest[0], data.ncol(), ndim);

  if (verbose) {
    Rcerr << "Search Tree #nodes = " << search_tree.offsets.size() << std::endl;
    for (std::size_t i = 0; i < search_tree.offsets.size(); ++i) {
      Rcerr << i;
      if (std::isnan(search_tree.offsets[i])) {
        Rcerr << " leaf ptrs: [" << search_tree.children[i].first << ", "
              << search_tree.children[i].second << "): ";
        // << search_tree.indices[i].size() << " items:";
        for (std::size_t j = search_tree.children[i].first;
             j < search_tree.children[i].second; ++j) {
          Rcerr << " " << search_tree.indices[j] + 1;
        }
      } else {
        Rcerr << " node off: " << search_tree.offsets[i]
              << " children: " << search_tree.children[i].first << " "
              << search_tree.children[i].second;
      }
      Rcerr << std::endl;
    }
    for (std::size_t i = 0; i < search_tree.indices.size(); ++i) {
      Rcerr << search_tree.indices[i] << " ";
    }
    Rcerr << std::endl;
  }

  auto rng_ptr = rng_provider.get_parallel_instance(data.ncol());
  List search_list(data.ncol());
  for (std::size_t i = 0; i < static_cast<std::size_t>(data.ncol()); i++) {
    std::vector<In> observation(data_vec.begin() + i * ndim,
                                data_vec.begin() + (i + 1) * ndim);
    std::pair<std::size_t, std::size_t> range =
        tdoann::tree_search(search_tree, observation, *rng_ptr);
    std::vector<Idx> leaf_indices(search_tree.indices.begin() + range.first,
                                  search_tree.indices.begin() + range.second);

    std::transform(leaf_indices.begin(), leaf_indices.end(),
                   leaf_indices.begin(), [](Idx value) { return value + 1; });
    IntegerVector leaf_r(leaf_indices.begin(), leaf_indices.end());

    search_list[i] = leaf_r;
  }

  return search_list;
}

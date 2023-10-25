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
void print_rp_forest(const std::vector<tdoann::RPTree<Idx, In>> &rp_forest) {
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

template <typename Idx, typename In>
void print_search_tree(const tdoann::SearchTree<Idx, In> &search_tree,
                       std::size_t ndim) {
  Rcerr << "Search Tree #nodes = " << search_tree.offsets.size() << std::endl;
  for (std::size_t i = 0; i < search_tree.offsets.size(); ++i) {
    Rcerr << i;
    if (std::isnan(search_tree.offsets[i])) {
      Rcerr << " leaf ptrs: [" << search_tree.children[i].first << ", "
            << search_tree.children[i].second << "): ";
      for (std::size_t j = search_tree.children[i].first;
           j < search_tree.children[i].second; ++j) {
        Rcerr << " " << search_tree.indices[j] + 1;
      }
    } else {
      Rcerr << "node-> " << search_tree.children[i].first << " "
            << search_tree.children[i].second
            << " off: " << search_tree.offsets[i] << " hyp:";
      for (std::size_t j = 0; j < ndim; j++) {
        Rcerr << " " << search_tree.hyperplanes[i][j];
      }
    }
    Rcerr << std::endl;
  }
}

template <typename Idx, typename In>
List search_tree_to_r(const tdoann::SearchTree<Idx, In> &search_tree,
                      std::size_t ndim) {

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
    List tree_list = search_tree_to_r(search_forest[i], ndim);
    forest_list[i] = tree_list;
  }

  return forest_list;
}

template <typename Idx, typename In>
tdoann::SearchTree<Idx, In> r_to_search_tree(List tree_list) {
  NumericMatrix hyperplanes = tree_list["hyperplanes"];
  NumericVector offsets = tree_list["offsets"];
  IntegerMatrix children = tree_list["children"];
  IntegerVector indices = tree_list["indices"];
  int leaf_size = tree_list["leaf_size"];

  std::size_t n_rows = hyperplanes.nrow();
  std::vector<std::vector<In>> cpp_hyperplanes(
      n_rows, std::vector<In>(hyperplanes.ncol()));
  std::vector<In> cpp_offsets(offsets.size());
  std::vector<std::pair<std::size_t, std::size_t>> cpp_children(
      children.nrow());
  // std::vector<Idx> cpp_indices(indices.size());

  for (std::size_t i = 0; i < n_rows; ++i) {
    for (std::size_t j = 0; j < hyperplanes.ncol(); ++j) {
      cpp_hyperplanes[i][j] = hyperplanes(i, j);
    }
    cpp_offsets[i] = offsets[i];
    cpp_children[i] = std::make_pair(children(i, 0), children(i, 1));
  }

  std::vector<Idx> cpp_indices = Rcpp::as<std::vector<Idx>>(indices);

  // for (std::size_t i = 0; i < indices.size(); ++i) {
  //   cpp_indices[i] = indices[i];
  // }

  return tdoann::SearchTree<Idx, In>(cpp_hyperplanes, cpp_offsets, cpp_children,
                                     cpp_indices, leaf_size);
}

template <typename Idx, typename In>
std::vector<tdoann::SearchTree<Idx, In>> r_to_search_forest(List forest_list) {
  std::size_t n_trees = forest_list.size();
  std::vector<tdoann::SearchTree<Idx, In>> search_forest(n_trees);

  for (std::size_t i = 0; i < n_trees; ++i) {
    List tree_list = forest_list[i];
    search_forest[i] = r_to_search_tree<Idx, In>(tree_list);
  }

  return search_forest;
}

template <typename Idx>
List init_rp_tree_binary(const NumericMatrix &data, uint32_t nnbrs,
                         const std::string &metric, bool include_self,
                         const std::vector<Idx> &leaf_array,
                         std::size_t max_leaf_size, std::size_t n_threads,
                         tdoann::ProgressBase &progress,
                         const tdoann::Executor &executor) {
  auto distance_ptr = create_self_distance(data, metric);

  auto neighbor_heap =
      tdoann::init_rp_tree(*distance_ptr, leaf_array, max_leaf_size, nnbrs,
                           include_self, n_threads, progress, executor);

  return heap_to_r(neighbor_heap, n_threads, progress, executor);
}

template <typename Idx, typename In>
std::vector<tdoann::RPTree<Idx, In>>
build_rp_forest(const std::vector<In> &data_vec, std::size_t ndim,
                const std::string &metric, unsigned int n_trees,
                unsigned int leaf_size, std::size_t n_threads, bool verbose,
                const tdoann::Executor &executor) {
  bool angular = is_angular_metric(metric);
  RPProgress forest_progress(verbose);
  rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler> rng_provider;
  if (verbose) {
    tsmessage() << "Creating" << (angular ? " angular " : "")
                << " RP forest with " << n_trees << " trees" << std::endl;
  }
  return tdoann::make_forest(data_vec, ndim, n_trees, leaf_size, rng_provider,
                             angular, n_threads, forest_progress, executor);
}

// [[Rcpp::export]]
List rp_tree_knn_cpp(const NumericMatrix &data, uint32_t nnbrs,
                     const std::string &metric, unsigned int n_trees,
                     unsigned int leaf_size, bool include_self,
                     std::size_t n_threads = 0, bool verbose = false) {
  using Idx = RNN_DEFAULT_IDX;
  using In = RNN_DEFAULT_IN;

  const std::size_t ndim = data.nrow();
  auto data_vec = r_to_vec<In>(data);

  RParallelExecutor executor;
  auto rp_forest = build_rp_forest<Idx>(
      data_vec, ndim, metric, n_trees, leaf_size, n_threads, verbose, executor);

  if (verbose) {
    tsmessage() << "Extracting leaf array from forest" << std::endl;
  }
  const std::size_t max_leaf_size = tdoann::find_max_leaf_size(rp_forest);
  std::vector<Idx> leaf_array =
      tdoann::get_leaves_from_forest(rp_forest, max_leaf_size);

  RPProgress knn_progress(verbose);
  if (verbose) {
    tsmessage() << "Creating knn using " << leaf_array.size() / max_leaf_size
                << " leaves" << std::endl;
  }
  if (metric == "bhamming") {
    // unfortunately data_vec is still in scope even though we no longer
    // need it
    return init_rp_tree_binary(data, nnbrs, metric, include_self, leaf_array,
                               max_leaf_size, n_threads, knn_progress,
                               executor);
  }
  auto distance_ptr = create_self_distance(std::move(data_vec), ndim, metric);
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
  using Idx = RNN_DEFAULT_IDX;
  using In = RNN_DEFAULT_IN;

  const std::size_t ndim = data.nrow();
  auto data_vec = r_to_vec<In>(data);

  RParallelExecutor executor;
  auto rp_forest = build_rp_forest<Idx>(
      data_vec, ndim, metric, n_trees, leaf_size, n_threads, verbose, executor);

  auto search_forest = tdoann::convert_rp_forest(rp_forest, data.ncol(), ndim);

  return search_forest_to_r(search_forest, ndim);
}

// [[Rcpp::export]]
List rnn_tree_search(const NumericMatrix &data, uint32_t nnbrs,
                     const std::string &metric,
                     unsigned int leaf_size, bool angular,
                     bool verbose = false) {
  using Idx = RNN_DEFAULT_IDX;
  using In = RNN_DEFAULT_IN;

  auto data_vec = r_to_vec<In>(data);

  const std::size_t n_obs = data.ncol();
  const std::size_t ndim = data.nrow();

  constexpr std::size_t n_threads = 0;
  constexpr unsigned int n_trees = 1;
  RParallelExecutor executor;
  auto rp_forest = build_rp_forest<Idx>(
      data_vec, ndim, metric, n_trees, leaf_size, n_threads, verbose, executor);
  if (verbose) {
    print_rp_forest(rp_forest);
  }

  tdoann::SearchTree<Idx, In> search_tree =
      tdoann::convert_tree_format(rp_forest[0], n_obs, ndim);
  if (verbose) {
    print_search_tree(search_tree, ndim);
  }

  rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler> rng_provider;
  auto rng_ptr = rng_provider.get_parallel_instance(data.ncol());
  List search_list(n_obs);
  for (std::size_t i = 0; i < n_obs; i++) {
    std::vector<Idx> leaf_indices =
        search_indices(search_tree, data_vec.begin() + i * ndim, *rng_ptr);

    std::transform(leaf_indices.begin(), leaf_indices.end(),
                   leaf_indices.begin(), [](Idx value) { return value + 1; });
    IntegerVector leaf_r(leaf_indices.begin(), leaf_indices.end());

    search_list[i] = leaf_r;
  }

  return search_list;
}

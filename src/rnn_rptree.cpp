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
#include "tdoann/rptreeimplicit.h"

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

enum class MarginType { EXPLICIT, IMPLICIT };

// Function to convert MarginType to a string
std::string margin_type_to_string(MarginType margin_type) {
  switch (margin_type) {
  case MarginType::EXPLICIT:
    return "explicit";
  case MarginType::IMPLICIT:
    return "implicit";
  }
  return "";
}

template <typename In, typename Idx>
void print_rp_forest(const std::vector<tdoann::RPTree<In, Idx>> &rp_forest) {
  for (const auto &tree : rp_forest) {
    Rcerr << "RP Tree #nodes = " << tree.offsets.size() << "\n";
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
      Rcerr << "\n";
    }
  }
}

template <typename In, typename Idx>
void print_search_forest(
    const std::vector<tdoann::SearchTree<In, Idx>> &search_forest,
    std::size_t ndim) {
  for (std::size_t j = 0; j < search_forest.size(); ++j) {
    const auto &search_tree = search_forest[j];

    Rcerr << "Search Tree " << j << "#nodes = " << search_tree.offsets.size()
          << "\n";
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
      Rcerr << "\n";
    }
  }
}

template <typename In, typename Idx>
List search_tree_to_r(const tdoann::SearchTree<In, Idx> &search_tree) {

  std::size_t n_nodes = search_tree.hyperplanes.size();
  NumericVector offsets(search_tree.offsets.size());
  std::size_t n_hyperplane_cols = search_tree.hyperplanes[0].size();
  NumericMatrix hyperplanes(n_nodes, n_hyperplane_cols);

  IntegerMatrix children(search_tree.children.size(), 2);
  for (std::size_t i = 0; i < n_nodes; ++i) {
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

template <typename In, typename Idx>
List search_forest_to_r(
    const std::vector<tdoann::SearchTree<In, Idx>> &search_forest) {
  std::size_t n_trees = search_forest.size();
  List forest_list(n_trees);

  for (std::size_t i = 0; i < n_trees; ++i) {
    List tree_list = search_tree_to_r(search_forest[i]);
    forest_list[i] = tree_list;
  }
  return List::create(_("trees") = forest_list,
                      _("margin") = margin_type_to_string(MarginType::EXPLICIT),
                      _("version") = "0.0.12");
}

template <typename Idx>
List search_tree_implicit_to_r(
    const tdoann::SearchTreeImplicit<Idx> &search_tree) {
  std::size_t n_nodes = search_tree.children.size();

  IntegerMatrix children(n_nodes, 2);
  IntegerMatrix normal_indices(n_nodes, 2);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    children(i, 0) = search_tree.children[i].first;
    children(i, 1) = search_tree.children[i].second;

    normal_indices(i, 0) = search_tree.normal_indices[i].first;
    normal_indices(i, 1) = search_tree.normal_indices[i].second;
  }

  IntegerVector indices(search_tree.indices.size());
  for (std::size_t i = 0; i < search_tree.indices.size(); ++i) {
    indices[i] = search_tree.indices[i];
  }

  return List::create(_("normal_indices") = normal_indices,
                      _("children") = children, _("indices") = indices,
                      _("leaf_size") = search_tree.leaf_size);
}

template <typename Idx>
List search_forest2_to_r(
    const std::vector<tdoann::SearchTreeImplicit<Idx>> &search_forest) {
  std::size_t n_trees = search_forest.size();
  List forest_list(n_trees);

  for (std::size_t i = 0; i < n_trees; ++i) {
    List tree_list = search_tree_implicit_to_r(search_forest[i]);
    forest_list[i] = tree_list;
  }

  return List::create(_("trees") = forest_list,
                      _("margin") = margin_type_to_string(MarginType::IMPLICIT),
                      _("version") = "0.0.12");
}

template <typename In, typename Idx>
tdoann::SearchTree<In, Idx> r_to_search_tree(List tree_list) {
  NumericMatrix hyperplanes = tree_list["hyperplanes"];
  NumericVector offsets = tree_list["offsets"];
  IntegerMatrix children = tree_list["children"];
  IntegerVector indices = tree_list["indices"];
  int leaf_size = tree_list["leaf_size"];

  std::size_t n_nodes = hyperplanes.ncol();
  std::size_t n_rows = hyperplanes.nrow();
  std::vector<std::vector<In>> cpp_hyperplanes(n_rows,
                                               std::vector<In>(n_nodes));
  std::vector<In> cpp_offsets(offsets.size());
  std::vector<std::pair<std::size_t, std::size_t>> cpp_children(
      children.nrow());

  for (std::size_t i = 0; i < n_rows; ++i) {
    for (std::size_t j = 0; j < n_nodes; ++j) {
      cpp_hyperplanes[i][j] = hyperplanes(i, j);
    }
    cpp_offsets[i] = offsets[i];
    cpp_children[i] = std::make_pair(children(i, 0), children(i, 1));
  }

  std::vector<Idx> cpp_indices = Rcpp::as<std::vector<Idx>>(indices);

  return tdoann::SearchTree<In, Idx>(cpp_hyperplanes, cpp_offsets, cpp_children,
                                     cpp_indices, leaf_size);
}

template <typename Idx>
tdoann::SearchTreeImplicit<Idx> r_to_search_tree_implicit(List tree_list) {
  IntegerMatrix normal_indices = tree_list["normal_indices"];
  IntegerMatrix children = tree_list["children"];
  IntegerVector indices = tree_list["indices"];
  int leaf_size = tree_list["leaf_size"];

  std::size_t n_nodes = children.nrow();

  std::vector<std::pair<Idx, Idx>> cpp_normal_indices(n_nodes);
  std::vector<std::pair<std::size_t, std::size_t>> cpp_children(n_nodes);

  for (std::size_t i = 0; i < n_nodes; ++i) {
    cpp_normal_indices[i] =
        std::make_pair(normal_indices(i, 0), normal_indices(i, 1));
    cpp_children[i] = std::make_pair(children(i, 0), children(i, 1));
  }

  std::vector<Idx> cpp_indices = Rcpp::as<std::vector<Idx>>(indices);

  return tdoann::SearchTreeImplicit<Idx>(cpp_normal_indices, cpp_children,
                                         cpp_indices, leaf_size);
}

template <typename In, typename Idx>
std::vector<tdoann::SearchTree<In, Idx>>
r_to_search_forest(List forest_list, std::size_t n_threads) {
  if (not forest_list.containsElementNamed("margin")) {
    Rcpp::stop("Bad forest object passed");
  }
  const std::string &margin_type = forest_list["margin"];
  if (margin_type != margin_type_to_string(MarginType::EXPLICIT)) {
    Rcpp::stop("Unsupported margin type: ", margin_type);
  }

  const List &trees = forest_list["trees"];
  const auto n_trees = trees.size();
  std::vector<tdoann::SearchTree<In, Idx>> search_forest(n_trees);

  auto worker = [&](std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; ++i) {
      search_forest[i] = r_to_search_tree<In, Idx>(trees[i]);
    }
  };

  RParallelExecutor executor;
  RPProgress progress(false);
  tdoann::ExecutionParams exec_params{n_threads};
  dispatch_work(worker, n_trees, n_threads, exec_params, progress, executor);

  return search_forest;
}

template <typename Idx>
std::vector<tdoann::SearchTreeImplicit<Idx>>
r_to_search_forest2(List forest_list, std::size_t n_threads) {
  if (not forest_list.containsElementNamed("margin")) {
    Rcpp::stop("Bad forest object passed");
  }
  const std::string margin_type = forest_list["margin"];
  if (margin_type != margin_type_to_string(MarginType::IMPLICIT)) {
    Rcpp::stop("Unsupported forest type: ", margin_type);
  }

  const List &trees = forest_list["trees"];
  const auto n_trees = trees.size();
  std::vector<tdoann::SearchTreeImplicit<Idx>> search_forest(n_trees);

  auto worker = [&](std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; ++i) {
      search_forest[i] = r_to_search_tree_implicit<Idx>(trees[i]);
    }
  };

  RParallelExecutor executor;
  RPProgress progress(false);
  tdoann::ExecutionParams exec_params{n_threads};
  dispatch_work(worker, n_trees, n_threads, exec_params, progress, executor);

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

template <typename In, typename Idx>
std::vector<tdoann::RPTree<In, Idx>>
build_rp_forest(const std::vector<In> &data_vec, std::size_t ndim,
                const std::string &metric, uint32_t n_trees, uint32_t leaf_size,

                std::size_t n_threads, bool verbose,
                const tdoann::Executor &executor) {
  bool angular = is_angular_metric(metric);
  rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler> rng_provider;
  if (verbose) {
    tsmessage() << "Using" << (angular ? " angular " : " euclidean ")
                << "margin calculation\n";
  }
  RPProgress forest_progress(verbose);
  return tdoann::make_forest(data_vec, ndim, n_trees, leaf_size, rng_provider,
                             angular, n_threads, forest_progress, executor);
}

// [[Rcpp::export]]
List rp_tree_knn_cpp(const NumericMatrix &data, uint32_t nnbrs,
                     const std::string &metric, uint32_t n_trees,
                     uint32_t leaf_size, bool include_self, bool unzero = true,
                     bool ret_forest = false, std::size_t n_threads = 0,
                     bool verbose = false) {
  using Idx = RNN_DEFAULT_IDX;
  using In = RNN_DEFAULT_IN;

  const std::size_t ndim = data.nrow();
  auto data_vec = r_to_vec<In>(data);

  RParallelExecutor executor;
  auto rp_forest = build_rp_forest<In, Idx>(
      data_vec, ndim, metric, n_trees, leaf_size, n_threads, verbose, executor);

  if (verbose) {
    tsmessage() << "Extracting leaf array from forest\n";
  }
  const std::size_t max_leaf_size = tdoann::find_max_leaf_size(rp_forest);
  std::vector<Idx> leaf_array =
      tdoann::get_leaves_from_forest(rp_forest, max_leaf_size);

  if (verbose) {
    tsmessage() << "Creating knn using " << leaf_array.size() / max_leaf_size
                << " leaves\n";
  }
  RPProgress knn_progress(verbose);
  if (is_binary_metric(metric)) {
    // unfortunately data_vec is still in scope even though we no longer
    // need it
    auto nn_list =
        init_rp_tree_binary(data, nnbrs, metric, include_self, leaf_array,
                            max_leaf_size, n_threads, knn_progress, executor);
    if (ret_forest) {
      auto search_forest =
          tdoann::convert_rp_forest(rp_forest, data.ncol(), ndim);
      List search_forest_r = search_forest_to_r(search_forest);
      nn_list["forest"] = search_forest_r;
    }
    return nn_list;
  }
  auto distance_ptr = create_self_distance(std::move(data_vec), ndim, metric);
  auto neighbor_heap =
      tdoann::init_rp_tree(*distance_ptr, leaf_array, max_leaf_size, nnbrs,
                           include_self, n_threads, knn_progress, executor);
  auto nn_list =
      heap_to_r(neighbor_heap, n_threads, knn_progress, executor, unzero);
  if (ret_forest) {
    auto search_forest =
        tdoann::convert_rp_forest(rp_forest, data.ncol(), ndim);
    List search_forest_r = search_forest_to_r(search_forest);
    nn_list["forest"] = search_forest_r;
  }
  return nn_list;
}

// [[Rcpp::export]]
List rp_tree_knn_cpp2(const NumericMatrix &data, uint32_t nnbrs,
                      const std::string &metric, uint32_t n_trees,
                      uint32_t leaf_size, bool include_self, bool unzero = true,
                      bool ret_forest = false, std::size_t n_threads = 0,
                      bool verbose = false) {
  const std::size_t ndim = data.nrow();

  auto distance_ptr = create_self_distance(data, metric);
  using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;

  RParallelExecutor executor;
  rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler> rng_provider;
  RPProgress forest_progress(verbose);
  auto rp_forest =
      tdoann::make_forest(*distance_ptr, ndim, n_trees, leaf_size, rng_provider,
                          n_threads, forest_progress, executor);

  if (verbose) {
    tsmessage() << "Extracting leaf array from forest\n";
  }
  const std::size_t max_leaf_size = tdoann::find_max_leaf_size(rp_forest);
  std::vector<Idx> leaf_array =
      tdoann::get_leaves_from_forest(rp_forest, max_leaf_size);

  if (verbose) {
    tsmessage() << "Creating knn using " << leaf_array.size() / max_leaf_size
                << " leaves\n";
  }
  RPProgress knn_progress(verbose);

  auto neighbor_heap =
      tdoann::init_rp_tree(*distance_ptr, leaf_array, max_leaf_size, nnbrs,
                           include_self, n_threads, knn_progress, executor);

  auto nn_list =
      heap_to_r(neighbor_heap, n_threads, knn_progress, executor, unzero);

  if (ret_forest) {
    auto search_forest =
        tdoann::convert_rp_forest(rp_forest, data.ncol(), ndim);
    List search_forest_r = search_forest2_to_r(search_forest);
    nn_list["forest"] = search_forest_r;
  }
  return nn_list;
}

// [[Rcpp::export]]
List rnn_rp_forest_build(const NumericMatrix &data, const std::string &metric,
                         uint32_t n_trees, uint32_t leaf_size,
                         std::size_t n_threads = 0, bool verbose = false) {
  using Idx = RNN_DEFAULT_IDX;
  using In = RNN_DEFAULT_IN;

  const std::size_t ndim = data.nrow();
  auto data_vec = r_to_vec<In>(data);

  RParallelExecutor executor;
  auto rp_forest = build_rp_forest<In, Idx>(
      data_vec, ndim, metric, n_trees, leaf_size, n_threads, verbose, executor);

  auto search_forest = tdoann::convert_rp_forest(rp_forest, data.ncol(), ndim);

  return search_forest_to_r(search_forest);
}

// [[Rcpp::export]]
List rnn_rp_forest_implicit_build(const NumericMatrix &data,
                                  const std::string &metric, uint32_t n_trees,
                                  uint32_t leaf_size, std::size_t n_threads = 0,
                                  bool verbose = false) {
  const std::size_t ndim = data.nrow();

  auto distance_ptr = create_self_distance(data, metric);
  using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;

  RParallelExecutor executor;
  rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler> rng_provider;
  RPProgress forest_progress(verbose);
  auto rp_forest =
      tdoann::make_forest(*distance_ptr, ndim, n_trees, leaf_size, rng_provider,
                          n_threads, forest_progress, executor);
  auto search_forest = tdoann::convert_rp_forest(rp_forest, data.ncol(), ndim);
  List search_forest_r = search_forest2_to_r(search_forest);

  return search_forest2_to_r(search_forest);
}

// [[Rcpp::export]]
List rnn_rp_forest_search(const NumericMatrix &query,
                          const NumericMatrix &reference, List search_forest,
                          uint32_t n_nbrs, const std::string &metric,
                          bool cache, std::size_t n_threads,
                          bool verbose = false) {
  RParallelExecutor executor;
  std::string margin_type = search_forest["margin"];

  if (margin_type == margin_type_to_string(MarginType::EXPLICIT)) {
    auto distance_ptr = create_query_vector_distance(reference, query, metric);

    using In = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Input;
    using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;

    auto search_forest_cpp =
        r_to_search_forest<In, Idx>(search_forest, n_threads);

    rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler>
        rng_provider;
    RPProgress progress(verbose);
    auto nn_heap = tdoann::search_forest(search_forest_cpp, *distance_ptr,
                                         n_nbrs, rng_provider, cache, n_threads,
                                         progress, executor);
    return heap_to_r(nn_heap);
  } else if (margin_type == margin_type_to_string(MarginType::IMPLICIT)) {
    auto distance_ptr = create_query_distance(reference, query, metric);
    using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;

    auto search_forest_cpp = r_to_search_forest2<Idx>(search_forest, n_threads);

    rnndescent::ParallelIntRNGAdapter<Idx, rnndescent::DQIntSampler>
        rng_provider;
    RPProgress progress(verbose);
    auto nn_heap = tdoann::search_forest(search_forest_cpp, *distance_ptr,
                                         n_nbrs, rng_provider, cache, n_threads,
                                         progress, executor);
    return heap_to_r(nn_heap);
  } else {
    Rcpp::stop("Bad search forest type ", margin_type);
  }
}

template <typename Tree>
std::vector<Tree> rnn_score_forest_impl(const IntegerMatrix &idx,
                                        const std::vector<Tree> &search_forest,
                                        uint32_t n_trees, std::size_t n_threads,
                                        bool verbose = false) {
  using Idx = typename Tree::Index;

  std::vector<Idx> idx_vec = r_to_idxt<Idx>(idx);

  uint32_t k = idx.ncol();

  RPProgress progress(verbose);
  RParallelExecutor executor;
  std::vector<double> scores = tdoann::score_forest(
      search_forest, idx_vec, k, n_threads, progress, executor);

  if (verbose) {
    auto min_it = std::min_element(scores.begin(), scores.end());
    auto max_it = std::max_element(scores.begin(), scores.end());

    double total = std::accumulate(scores.begin(), scores.end(), 0.0);
    double mean = total / scores.size();

    Rcpp::Rcerr << "Min score: " << *min_it << "\n"
                << "Max score: " << *max_it << "\n"
                << "Mean score: " << mean << "\n";
  }

  return tdoann::filter_top_n_trees(search_forest, scores, n_trees);
}

// [[Rcpp::export]]
List rnn_score_forest(const IntegerMatrix &idx, List search_forest,
                      uint32_t n_trees, std::size_t n_threads,
                      bool verbose = false) {
  using Idx = RNN_DEFAULT_IDX;
  using In = RNN_DEFAULT_IN;

  if (not search_forest.containsElementNamed("margin")) {
    Rcpp::stop("Bad forest object passed");
  }
  const std::string margin_type = search_forest["margin"];
  if (margin_type == margin_type_to_string(MarginType::EXPLICIT)) {
    auto search_forest_cpp =
        r_to_search_forest<In, Idx>(search_forest, n_threads);

    auto filtered_forest = rnn_score_forest_impl(idx, search_forest_cpp,
                                                 n_trees, n_threads, verbose);

    return search_forest_to_r(filtered_forest);
  } else if (margin_type == margin_type_to_string(MarginType::IMPLICIT)) {
    auto search_forest_cpp = r_to_search_forest2<Idx>(search_forest, n_threads);

    auto filtered_forest = rnn_score_forest_impl(idx, search_forest_cpp,
                                                 n_trees, n_threads, verbose);

    return search_forest2_to_r(filtered_forest);
  } else {
    Rcpp::stop("Unknown forest type: ", margin_type);
  }
}

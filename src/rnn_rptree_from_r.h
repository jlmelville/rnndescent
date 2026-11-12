//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#ifndef RNN_RPTREE_FROM_R_H
#define RNN_RPTREE_FROM_R_H

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <Rcpp.h>

#include "rnn_util.h"
#include "tdoann/rptree.h"
#include "tdoann/rptreeimplicit.h"
#include "tdoann/rptreesparse.h"

enum class MarginType { EXPLICIT, IMPLICIT };

inline std::string margin_type_to_string(MarginType margin_type) {
  switch (margin_type) {
  case MarginType::EXPLICIT:
    return "explicit";
  case MarginType::IMPLICIT:
    return "implicit";
  }
  return "";
}

template <typename In, typename Idx>
tdoann::SearchTree<In, Idx> r_to_search_tree(Rcpp::List tree_list) {
  // n_nodes x ndim
  Rcpp::NumericMatrix hyperplanes = tree_list["hyperplanes"];
  // n_nodes
  Rcpp::NumericVector offsets = tree_list["offsets"];
  // n_nodes x 2
  Rcpp::IntegerMatrix children = tree_list["children"];
  // n_obs
  Rcpp::IntegerVector indices = tree_list["indices"];
  int leaf_size = tree_list["leaf_size"];

  const std::size_t ndim = hyperplanes.ncol();
  const std::size_t n_nodes = hyperplanes.nrow();
  std::vector<std::vector<In>> cpp_hyperplanes(n_nodes, std::vector<In>(ndim));
  std::vector<In> cpp_offsets(n_nodes);
  std::vector<std::pair<std::size_t, std::size_t>> cpp_children(n_nodes);

  for (std::size_t i = 0; i < n_nodes; ++i) {
    for (std::size_t j = 0; j < ndim; ++j) {
      cpp_hyperplanes[i][j] = hyperplanes(i, j);
    }
    cpp_offsets[i] = offsets[i];
    cpp_children[i] = std::make_pair(children(i, 0), children(i, 1));
  }

  auto cpp_indices = r0_to_idx<Idx>(indices);

  return tdoann::SearchTree<In, Idx>(
      std::move(cpp_hyperplanes), std::move(cpp_offsets),
      std::move(cpp_children), std::move(cpp_indices), leaf_size);
}

template <typename Idx>
tdoann::SearchTreeImplicit<Idx>
r_to_search_tree_implicit(Rcpp::List tree_list) {
  // n_nodes x 2
  Rcpp::IntegerMatrix normal_indices = tree_list["normal_indices"];
  // n_nodes x 2
  Rcpp::IntegerMatrix children = tree_list["children"];
  // n_obs
  Rcpp::IntegerVector indices = tree_list["indices"];
  Idx leaf_size = tree_list["leaf_size"];

  const std::size_t n_nodes = children.nrow();

  std::vector<std::pair<Idx, Idx>> cpp_normal_indices(n_nodes);
  std::vector<std::pair<std::size_t, std::size_t>> cpp_children(n_nodes);

  for (std::size_t i = 0; i < n_nodes; ++i) {
    cpp_normal_indices[i] =
        std::make_pair(normal_indices(i, 0), normal_indices(i, 1));
    cpp_children[i] = std::make_pair(children(i, 0), children(i, 1));
  }

  auto cpp_indices = r0_to_idx<Idx>(indices);

  return tdoann::SearchTreeImplicit<Idx>(std::move(cpp_normal_indices),
                                         std::move(cpp_children),
                                         std::move(cpp_indices), leaf_size);
}

template <typename In, typename Idx>
std::vector<tdoann::SearchTree<In, Idx>>
r_to_search_forest(Rcpp::List forest_list) {
  if (not forest_list.containsElementNamed("margin")) {
    Rcpp::stop("Bad forest object passed");
  }
  const std::string &margin_type = forest_list["margin"];
  if (margin_type != margin_type_to_string(MarginType::EXPLICIT)) {
    Rcpp::stop("Unsupported margin type: ", margin_type);
  }

  const Rcpp::List &trees = forest_list["trees"];
  const std::size_t n_trees = trees.size();
  std::vector<tdoann::SearchTree<In, Idx>> search_forest(n_trees);

  for (std::size_t i = 0; i < n_trees; ++i) {
    search_forest[i] = r_to_search_tree<In, Idx>(trees[i]);
  }

  return search_forest;
}

template <typename Idx>
std::vector<tdoann::SearchTreeImplicit<Idx>>
r_to_search_forest_implicit(Rcpp::List forest_list) {
  if (not forest_list.containsElementNamed("margin")) {
    Rcpp::stop("Bad forest object passed");
  }
  const std::string margin_type = forest_list["margin"];
  if (margin_type != margin_type_to_string(MarginType::IMPLICIT)) {
    Rcpp::stop("Unsupported forest type: ", margin_type);
  }

  const Rcpp::List &trees = forest_list["trees"];
  const std::size_t n_trees = trees.size();
  std::vector<tdoann::SearchTreeImplicit<Idx>> search_forest(n_trees);
  for (std::size_t i = 0; i < n_trees; ++i) {
    search_forest[i] = r_to_search_tree_implicit<Idx>(trees[i]);
  }

  return search_forest;
}

template <typename In, typename Idx>
tdoann::SparseSearchTree<In, Idx>
r_to_sparse_search_tree(Rcpp::List tree_list) {
  Rcpp::NumericVector hyperplanes_data = tree_list["hyperplanes_data"];
  Rcpp::IntegerVector hyperplanes_ind = tree_list["hyperplanes_ind"];
  Rcpp::IntegerVector hyperplanes_ptr = tree_list["hyperplanes_ptr"];
  Rcpp::NumericVector offsets = tree_list["offsets"];
  Rcpp::IntegerMatrix children = tree_list["children"];
  Rcpp::IntegerVector indices = tree_list["indices"];
  int leaf_size = tree_list["leaf_size"];

  const std::size_t n_nodes = offsets.size();

  std::vector<In> hyperplanes_data_cpp(hyperplanes_data.begin(),
                                       hyperplanes_data.end());
  std::vector<std::size_t> hyperplanes_ind_cpp(hyperplanes_ind.begin(),
                                               hyperplanes_ind.end());
  std::vector<std::size_t> hyperplanes_ptr_cpp(hyperplanes_ptr.begin(),
                                               hyperplanes_ptr.end());
  std::vector<In> offsets_cpp(offsets.begin(), offsets.end());
  std::vector<std::pair<std::size_t, std::size_t>> children_cpp(n_nodes);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    children_cpp[i] = {static_cast<std::size_t>(children(i, 0)),
                       static_cast<std::size_t>(children(i, 1))};
  }
  std::vector<Idx> indices_cpp(indices.begin(), indices.end());

  std::vector<std::vector<std::size_t>> hyperplanes_ind_nested(n_nodes);
  std::vector<std::vector<In>> hyperplanes_data_nested(n_nodes);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    auto start_idx = hyperplanes_ptr_cpp[i];
    auto end_idx = hyperplanes_ptr_cpp[i + 1];
    hyperplanes_ind_nested[i].assign(hyperplanes_ind_cpp.begin() + start_idx,
                                     hyperplanes_ind_cpp.begin() + end_idx);
    hyperplanes_data_nested[i].assign(hyperplanes_data_cpp.begin() + start_idx,
                                      hyperplanes_data_cpp.begin() + end_idx);
  }

  return tdoann::SparseSearchTree<In, Idx>(
      std::move(hyperplanes_ind_nested), std::move(hyperplanes_data_nested),
      std::move(offsets_cpp), std::move(children_cpp), std::move(indices_cpp),
      leaf_size);
}

template <typename In, typename Idx>
std::vector<tdoann::SparseSearchTree<In, Idx>>
r_to_sparse_search_forest(Rcpp::List forest_list) {
  if (not forest_list.containsElementNamed("margin")) {
    Rcpp::stop("Bad forest object passed");
  }
  const std::string &margin_type = forest_list["margin"];
  if (margin_type != margin_type_to_string(MarginType::EXPLICIT)) {
    Rcpp::stop("Unsupported margin type: ", margin_type);
  }

  const Rcpp::List &trees = forest_list["trees"];
  const std::size_t n_trees = trees.size();
  std::vector<tdoann::SparseSearchTree<In, Idx>> search_forest(n_trees);
  for (std::size_t i = 0; i < n_trees; ++i) {
    search_forest[i] = r_to_sparse_search_tree<In, Idx>(trees[i]);
  }
  return search_forest;
}

#endif // RNN_RPTREE_FROM_R_H

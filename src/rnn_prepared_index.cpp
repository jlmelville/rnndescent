//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Rcpp.h>

#include "rnn_prepared_index.h"
#include "rnn_rptree_from_r.h"

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::LogicalMatrix;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

namespace {
// The weak references permit collection of handles and let the unload hook
// run their finalizers before the native library is unmapped.
std::unordered_map<SEXP, SEXP> prepared_index_refs;

template <typename Index> void finalize_prepared_index(SEXP handle) {
  delete static_cast<Index *>(R_ExternalPtrAddr(handle));
  R_ClearExternalPtr(handle);
  prepared_index_refs.erase(handle);
}

template <typename Index>
SEXP register_prepared_index(std::unique_ptr<Index> index, SEXP tag) {
  Rcpp::XPtr<Index> handle(nullptr, false, tag);
  const auto registration =
      prepared_index_refs.emplace(handle, R_NilValue).first;
  try {
    registration->second = Rcpp::unwindProtect([&] {
      return R_MakeWeakRefC(handle, R_NilValue, finalize_prepared_index<Index>,
                            TRUE);
    });
  } catch (...) {
    prepared_index_refs.erase(registration);
    throw;
  }
  R_SetExternalPtrAddr(handle, index.release());
  return handle;
}

template <typename IsLeaf>
void check_prepared_tree(const IntegerMatrix &children,
                         const IntegerVector &indices, int n_reference,
                         IsLeaf is_leaf) {
  const int n_nodes = children.nrow();
  if (n_nodes == 0 || children.ncol() != 2 || indices.size() != n_reference) {
    Rcpp::stop("Invalid prepared search tree dimensions");
  }
  for (int i = 0; i < n_nodes; ++i) {
    const int left = children(i, 0);
    const int right = children(i, 1);
    if (is_leaf(i)) {
      if (left < 0 || right < left || right > n_reference) {
        Rcpp::stop("Invalid leaf range in prepared search tree");
      }
    } else if (left <= i || right <= i || left >= n_nodes || right >= n_nodes) {
      Rcpp::stop("Invalid child in prepared search tree");
    }
  }
  if (std::any_of(indices.begin(), indices.end(), [n_reference](int idx) {
        return idx < 0 || idx >= n_reference;
      })) {
    Rcpp::stop("Invalid reference index in prepared search tree");
  }
}

void check_prepared_forest(const List &forest, int n_reference, int ndim) {
  const List trees = forest["trees"];
  if (trees.size() == 0) {
    Rcpp::stop("Prepared index requires at least one search tree");
  }
  for (const auto &entry : trees) {
    const List tree = entry;
    const NumericMatrix hyperplanes = tree["hyperplanes"];
    const NumericVector offsets = tree["offsets"];
    const IntegerMatrix children = tree["children"];
    const IntegerVector indices = tree["indices"];
    const int n_nodes = hyperplanes.nrow();
    if (hyperplanes.ncol() != ndim || offsets.size() != n_nodes ||
        children.nrow() != n_nodes) {
      Rcpp::stop("Invalid prepared search tree dimensions");
    }
    check_prepared_tree(children, indices, n_reference,
                        [&](int i) { return std::isnan(offsets[i]); });
  }
}

auto prepared_search_graph(const List &graph, std::size_t n_reference)
    -> rnndescent::PreparedDenseIndex::Graph {
  auto search_graph = r_to_sparse_graph(graph);
  if (search_graph.n_points != n_reference ||
      search_graph.row_ptr.front() != 0 ||
      search_graph.row_ptr.back() != search_graph.col_idx.size() ||
      search_graph.dist.size() != search_graph.col_idx.size() ||
      !std::is_sorted(search_graph.row_ptr.begin(),
                      search_graph.row_ptr.end()) ||
      std::any_of(
          search_graph.col_idx.begin(), search_graph.col_idx.end(),
          [n_reference](RNN_DEFAULT_IDX idx) { return idx >= n_reference; })) {
    Rcpp::stop("Invalid prepared search graph");
  }
  return search_graph;
}

template <typename Idx>
auto prepare_implicit_search_forest(const List &forest, int n_reference)
    -> std::vector<tdoann::SearchTreeImplicit<Idx>> {
  const List trees = forest["trees"];
  if (trees.size() == 0) {
    Rcpp::stop("Prepared index requires at least one search tree");
  }
  for (const auto &entry : trees) {
    const List tree = entry;
    const IntegerMatrix children = tree["children"];
    const IntegerMatrix normals = tree["normal_indices"];
    if (normals.nrow() != children.nrow() || normals.ncol() != 2) {
      Rcpp::stop("Invalid prepared search tree dimensions");
    }
    check_prepared_tree(children, tree["indices"], n_reference, [&](int i) {
      const int left = normals(i, 0);
      const int right = normals(i, 1);
      if (left == -1) {
        return true;
      }
      if (left < 0 || left >= n_reference || right < 0 ||
          right >= n_reference) {
        Rcpp::stop("Invalid reference index in prepared search tree");
      }
      return false;
    });
  }
  return r_to_search_forest_implicit<Idx>(forest);
}

auto prepare_sparse_search_forest(const List &forest, int n_reference, int ndim)
    -> rnndescent::PreparedSparseIndex::Forest {
  using Index = rnndescent::PreparedSparseIndex;
  const std::string margin = forest["margin"];
  if (margin == "implicit") {
    return prepare_implicit_search_forest<Index::Idx>(forest, n_reference);
  }
  const List trees = forest["trees"];
  if (trees.size() == 0) {
    Rcpp::stop("Prepared index requires at least one search tree");
  }
  for (const auto &entry : trees) {
    const List tree = entry;
    const IntegerMatrix children = tree["children"];
    const NumericVector offsets = tree["offsets"];
    const NumericVector hyperplanes = tree["hyperplanes_data"];
    const IntegerVector ind = tree["hyperplanes_ind"];
    const IntegerVector ptr = tree["hyperplanes_ptr"];
    if (offsets.size() != children.nrow() || ptr.size() - 1 != offsets.size()) {
      Rcpp::stop("Invalid prepared search tree dimensions");
    }
    rnndescent::check_prepared_sparse_storage(ind, ptr, hyperplanes.size(),
                                              ndim);
    check_prepared_tree(children, tree["indices"], n_reference,
                        [&](int i) { return std::isnan(offsets[i]); });
  }
  return r_to_sparse_search_forest<Index::In, Index::Idx>(forest);
}
} // namespace

// [[Rcpp::export(rng = false)]]
void rnn_release_prepared_indexes() {
  while (!prepared_index_refs.empty()) {
    R_RunWeakRefFinalizer(prepared_index_refs.begin()->second);
  }
}

// [[Rcpp::export(rng = false)]]
SEXP rnn_prepare_dense_index(const NumericMatrix &reference, const List &graph,
                             const List &forest, const std::string &metric) {
  using Index = rnndescent::PreparedDenseIndex;
  if (reference.nrow() == 0 || reference.ncol() == 0) {
    Rcpp::stop("Expected a nonempty dense numeric reference");
  }
  auto [calculate, preprocess] =
      get_dense_distance_funcs<Index::In, Index::Out>(metric);
  const auto ndim = static_cast<std::size_t>(reference.nrow());
  validate_haversine_ndim(ndim, metric);
  check_prepared_forest(forest, reference.ncol(), reference.nrow());
  auto search_graph = prepared_search_graph(graph, reference.ncol());
  auto data = is_probability_metric(metric)
                  ? r_to_vec_for_probability<Index::In>(reference, metric,
                                                        "reference", true)
                  : copy_finite_reference<Index::In>(reference);
  validate_haversine_radians(data, ndim, metric);
  if (preprocess) {
    preprocess(data, ndim);
  } else if (metric == "spearmanr") {
    rnndescent::rank_dense_observations(data, ndim);
    calculate = tdoann::correlation<Index::Out, tdoann::DataIt<Index::In>>;
  }
  auto index = std::make_unique<Index>(
      Index{std::move(data), std::move(search_graph),
            r_to_search_forest<Index::In, Index::Idx>(forest), ndim, metric,
            calculate, preprocess});
  return register_prepared_index(std::move(index),
                                 rnndescent::prepared_index_tag());
}

// [[Rcpp::export(rng = false)]]
SEXP rnn_prepare_logical_index(const LogicalMatrix &reference,
                               const List &graph, const List &forest,
                               const std::string &metric) {
  using Index = rnndescent::PreparedLogicalIndex;
  if (reference.nrow() == 0 || reference.ncol() == 0) {
    Rcpp::stop("Expected a nonempty dense logical reference");
  }
  const auto calculate =
      find_packed_distance_func<Index::Out, Index::Idx>(metric);
  if (!calculate) {
    Rcpp::stop(
        "rnnd_prepare() requires a packed metric for dense logical indexes");
  }
  auto search_graph = prepared_search_graph(graph, reference.ncol());
  auto search_forest =
      prepare_implicit_search_forest<Index::Idx>(forest, reference.ncol());
  auto index = std::make_unique<Index>(
      Index{pack_logical_matrix(reference), std::move(search_graph),
            std::move(search_forest),
            static_cast<std::size_t>(reference.nrow()), calculate});
  return register_prepared_index(std::move(index),
                                 rnndescent::prepared_logical_index_tag());
}

// [[Rcpp::export(rng = false)]]
SEXP rnn_prepare_sparse_index(const IntegerVector &ind,
                              const IntegerVector &ptr,
                              const NumericVector &data, int ndim,
                              const List &graph, const List &forest,
                              const std::string &metric) {
  using Index = rnndescent::PreparedSparseIndex;
  if (ndim <= 0 || ptr.size() < 2 ||
      ptr.size() - 1 > std::numeric_limits<int>::max()) {
    Rcpp::stop("Expected a nonempty sparse reference");
  }
  rnndescent::check_prepared_sparse_storage(ind, ptr, data.size(), ndim);
  const int n_reference = ptr.size() - 1;
  auto search_graph = prepared_search_graph(graph, n_reference);
  auto search_forest = prepare_sparse_search_forest(forest, n_reference, ndim);
  auto reference = create_sparse_self_distance_impl<Index::Distance>(
      ind, ptr, data, ndim, metric);
  auto index = std::make_unique<Index>(
      Index{std::move(reference), std::move(search_graph),
            std::move(search_forest), static_cast<std::size_t>(ndim), metric,
            get_sparse_distance_funcs<Index::In, Index::Out>(metric).first});
  return register_prepared_index(std::move(index),
                                 rnndescent::prepared_sparse_index_tag());
}

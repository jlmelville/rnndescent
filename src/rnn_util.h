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

#ifndef RNN_UTIL_H
#define RNN_UTIL_H

#include <limits>

#include <Rcpp.h>

#include "tdoann/nngraph.h"

using RNN_DEFAULT_DIST = float;
using RNN_DEFAULT_IDX = uint32_t;

#define RNND_MAX_IDX (std::numeric_limits<int>::max)()

void print_time(bool print_date = false);
void ts(const std::string &);
void zero_index(Rcpp::IntegerMatrix &, int max_idx = RNND_MAX_IDX,
                bool missing_ok = false);

template <typename Out>
auto graph_to_r(const tdoann::NNGraph<Out> &graph, bool unzero = false)
    -> Rcpp::List {
  Rcpp::IntegerMatrix indices(graph.n_nbrs, graph.n_points, graph.idx.begin());
  Rcpp::NumericMatrix dist(graph.n_nbrs, graph.n_points, graph.dist.begin());

  return Rcpp::List::create(Rcpp::_("idx") =
                                Rcpp::transpose(unzero ? indices + 1 : indices),
                            Rcpp::_("dist") = Rcpp::transpose(dist));
}

template <typename T>
auto r_to_vec(const Rcpp::NumericVector &data) -> std::vector<T> {
  return Rcpp::as<std::vector<T>>(data);
}

template <typename T>
auto r_to_vec(const Rcpp::NumericMatrix &data) -> std::vector<T> {
  return Rcpp::as<std::vector<T>>(data);
}

template <typename T>
auto r_to_vect(const Rcpp::NumericMatrix &data) -> std::vector<T> {
  return Rcpp::as<std::vector<T>>(Rcpp::transpose(data));
}

template <typename Int>
inline auto r_to_idx(const Rcpp::IntegerMatrix &nn_idx,
                     int max_idx = RNND_MAX_IDX) -> std::vector<Int> {
  Rcpp::IntegerMatrix nn_idx_copy = Rcpp::clone(nn_idx);
  constexpr bool missing_ok = true;
  zero_index(nn_idx_copy, max_idx, missing_ok);
  return Rcpp::as<std::vector<Int>>(nn_idx_copy);
}

template <typename Int>
inline auto r_to_idxt(const Rcpp::IntegerMatrix &nn_idx,
                      int max_idx = RNND_MAX_IDX) -> std::vector<Int> {
  auto nn_idx_copy = Rcpp::clone(nn_idx);
  constexpr bool missing_ok = true;
  zero_index(nn_idx_copy, max_idx, missing_ok);
  return Rcpp::as<std::vector<Int>>(Rcpp::transpose(nn_idx_copy));
}

template <typename Out = RNN_DEFAULT_DIST, typename Idx = RNN_DEFAULT_IDX>
auto r_to_sparse_graph(const Rcpp::IntegerMatrix &idx,
                       const Rcpp::NumericMatrix &dist)
    -> tdoann::SparseNNGraph<Out, Idx> {

  auto idx_vec = r_to_idxt<Idx>(idx);
  auto dist_vec = r_to_vect<Out>(dist);

  const std::size_t nrow = idx.nrow();
  std::vector<std::size_t> ptr(nrow + 1);
  const std::size_t ncol = idx.ncol();
  for (std::size_t i = 0; i < nrow + 1; i++) {
    ptr[i] = i * ncol;
  }

  return tdoann::SparseNNGraph<Out, Idx>(ptr, idx_vec, dist_vec);
}

template <typename Out = RNN_DEFAULT_DIST, typename Idx = RNN_DEFAULT_IDX>
auto r_to_sparse_graph(const Rcpp::List &reference_graph)
    -> tdoann::SparseNNGraph<Out, Idx> {
  return tdoann::SparseNNGraph<Out, Idx>(reference_graph["row_ptr"],
                                         reference_graph["col_idx"],
                                         reference_graph["dist"]);
}

template <typename SparseNNGraph>
auto sparse_graph_to_r(const SparseNNGraph &sparse_graph) -> Rcpp::List {
  return Rcpp::List::create(Rcpp::_("row_ptr") = sparse_graph.row_ptr,
                            Rcpp::_("col_idx") = sparse_graph.col_idx,
                            Rcpp::_("dist") = sparse_graph.dist);
}

#endif // RNN_UTIL_H

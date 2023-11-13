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

// NOLINTBEGIN(modernize-use-trailing-return-type)

#include "rnndescent/random.h"
#include "tdoann/nngraph.h"
#include "tdoann/prepare.h"

#include <Rcpp.h>

#include "rnn_distance.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"

using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::LogicalMatrix;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

template <typename Out, typename Idx>
List diversify_impl(const tdoann::BaseDistance<Out, Idx> &distance,
                    List graph_list, double prune_probability,
                    std::size_t n_threads) {
  const auto graph = r_to_sparse_graph<Out, Idx>(graph_list);

  std::optional<tdoann::SparseNNGraph<Out, Idx>> diversified;
  if (n_threads == 0) {
    rnndescent::RRand rand;
    diversified =
        tdoann::remove_long_edges(graph, distance, rand, prune_probability);
  } else {
    RParallelExecutor executor;
    RPProgress progress(1, false);
    rnndescent::ParallelRNGAdapter<rnndescent::PcgRand> parallel_rand;
    diversified = tdoann::remove_long_edges(graph, distance, parallel_rand,
                                            prune_probability, n_threads,
                                            progress, executor);
  }
  return sparse_graph_to_r(*diversified);
}

// [[Rcpp::export]]
List rnn_sparse_diversify(const IntegerVector &ind, const IntegerVector &ptr,
                          const NumericVector &data, std::size_t ndim,
                          const List &graph_list, const std::string &metric,
                          double prune_probability, std::size_t n_threads) {
  auto distance_ptr = create_sparse_self_distance(ind, ptr, data, ndim, metric);
  return diversify_impl(*distance_ptr, graph_list, prune_probability,
                        n_threads);
}

// [[Rcpp::export]]
List rnn_diversify(const NumericMatrix &data, const List &graph_list,
                   const std::string &metric, double prune_probability,
                   std::size_t n_threads) {
  auto distance_ptr = create_self_distance(data, metric);
  return diversify_impl(*distance_ptr, graph_list, prune_probability,
                        n_threads);
}

// [[Rcpp::export]]
List rnn_logical_diversify(const LogicalMatrix &data, const List &graph_list,
                           const std::string &metric, double prune_probability,
                           std::size_t n_threads) {
  auto distance_ptr = create_self_distance(data, metric);
  return diversify_impl(*distance_ptr, graph_list, prune_probability,
                        n_threads);
}

// [[Rcpp::export]]
List rnn_merge_graph_lists(const List &graph_list1, const List &graph_list2) {
  auto graph1 = r_to_sparse_graph(graph_list1);
  auto graph2 = r_to_sparse_graph(graph_list2);

  auto graph_merged = tdoann::merge_graphs(graph1, graph2);

  return sparse_graph_to_r(graph_merged);
}

// [[Rcpp::export]]
List rnn_degree_prune(const List &graph_list, std::size_t max_degree,
                      std::size_t n_threads) {
  auto graph = r_to_sparse_graph(graph_list);

  RParallelExecutor executor;
  RPProgress progress(1, false);
  auto pruned =
      tdoann::degree_prune(graph, max_degree, n_threads, progress, executor);
  return sparse_graph_to_r(pruned);
}

// NOLINTEND(modernize-use-trailing-return-type)

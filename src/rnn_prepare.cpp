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

using Rcpp::List;
using Rcpp::NumericMatrix;

template <typename Out, typename Idx>
auto diversify_impl(const tdoann::SparseNNGraph<Out, Idx> &graph,
                    const tdoann::BaseDistance<Out, Idx> &distance,
                    double prune_probability, std::size_t n_threads)
    -> tdoann::SparseNNGraph<Out, Idx> {
  if (n_threads == 0) {
    rnndescent::RRand rand;
    return tdoann::remove_long_edges(graph, distance, rand, prune_probability);
  }

  RParallelExecutor executor;
  RPProgress progress(1, false);
  rnndescent::ParallelRNGAdapter<rnndescent::PcgRand> parallel_rand;
  return tdoann::remove_long_edges(graph, distance, parallel_rand,
                                   prune_probability, n_threads, progress,
                                   executor);
}

// [[Rcpp::export]]
List diversify_cpp(const NumericMatrix &data, const List &graph_list,
                   const std::string &metric, double prune_probability,
                   std::size_t n_threads) {
  auto distance_ptr = create_self_distance(data, metric);
  using Out = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Output;
  using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;
  const auto graph = r_to_sparse_graph<Out, Idx>(graph_list);

  auto diversified =
      diversify_impl(graph, *distance_ptr, prune_probability, n_threads);

  return sparse_graph_to_r(diversified);
}

// [[Rcpp::export]]
List merge_graph_lists_cpp(const List &graph_list1, const List &graph_list2) {
  auto graph1 = r_to_sparse_graph(graph_list1);
  auto graph2 = r_to_sparse_graph(graph_list2);

  auto graph_merged = tdoann::merge_graphs(graph1, graph2);

  return sparse_graph_to_r(graph_merged);
}

// [[Rcpp::export]]
List degree_prune_cpp(const List &graph_list, std::size_t max_degree,
                      std::size_t n_threads) {
  auto graph = r_to_sparse_graph(graph_list);

  RParallelExecutor executor;
  RPProgress progress(1, false);
  auto pruned =
      tdoann::degree_prune(graph, max_degree, n_threads, progress, executor);
  return sparse_graph_to_r(pruned);
}

// NOLINTEND(modernize-use-trailing-return-type)

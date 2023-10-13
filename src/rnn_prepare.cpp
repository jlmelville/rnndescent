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
#include "tdoann/prepare.h"

#include <Rcpp.h>

#include "rnn_distance.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"

// using namespace Rcpp;
using Rcpp::List;
using Rcpp::NumericMatrix;

#define DIVERSIFY_IMPL()                                                       \
  return diversify_impl<Distance>(data, graph_list, prune_probability,         \
                                  n_threads);

template <typename SparseNNGraph, typename Distance>
auto diversify_impl(const SparseNNGraph &graph, const Distance &distance,
                    double prune_probability, std::size_t n_threads)
    -> SparseNNGraph {
  if (n_threads == 0) {
    rnndescent::RRand rand;
    return tdoann::remove_long_edges(graph, distance, rand, prune_probability);
  }
  RPProgress progress(1, false);
  rnndescent::ParallelRNGAdapter<rnndescent::PcgRand> parallel_rand;
  return tdoann::remove_long_edges<RParallel>(
      graph, distance, parallel_rand, prune_probability, n_threads, progress);
}

template <typename Distance>
List diversify_impl(const NumericMatrix &data, const List &graph_list,
                    double prune_probability, std::size_t n_threads) {
  auto distance = tr_to_dist<Distance>(data);
  auto graph = r_to_sparse_graph<Distance>(graph_list);

  auto diversified =
      diversify_impl(graph, distance, prune_probability, n_threads);

  return sparse_graph_to_r(diversified);
}

// [[Rcpp::export]]
List diversify_cpp(const NumericMatrix &data, const List &graph_list,
                   const std::string &metric, double prune_probability,
                   std::size_t n_threads){DISPATCH_ON_DISTANCES(DIVERSIFY_IMPL)}

// [[Rcpp::export]]
List merge_graph_lists_cpp(const List &graph_list1, const List &graph_list2) {
  auto graph1 = r_to_sparse_graph<DummyDistance>(graph_list1);
  auto graph2 = r_to_sparse_graph<DummyDistance>(graph_list2);

  auto graph_merged = tdoann::merge_graphs(graph1, graph2);

  return sparse_graph_to_r(graph_merged);
}

template <typename SparseNNGraph>
auto degree_prune_impl(const SparseNNGraph &graph, std::size_t max_degree,
                       std::size_t n_threads) -> SparseNNGraph {
  RPProgress progress(1, false);
  if (n_threads == 0) {
    return tdoann::degree_prune(graph, max_degree, progress);
  }
  return tdoann::degree_prune<RParallel>(graph, max_degree, progress,
                                         n_threads);
}

// [[Rcpp::export]]
List degree_prune_cpp(const List &graph_list, std::size_t max_degree,
                      std::size_t n_threads) {
  auto graph = r_to_sparse_graph<DummyDistance>(graph_list);
  auto pruned = degree_prune_impl(graph, max_degree, n_threads);
  return sparse_graph_to_r(pruned);
}

// NOLINTEND(modernize-use-trailing-return-type)

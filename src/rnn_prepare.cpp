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

#include "tdoann/prepare.h"

#include <Rcpp.h>

#include "rnn_distance.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_rng.h"

using namespace Rcpp;

#define DIVERSIFY_IMPL()                                                       \
  return diversify_impl<Distance>(data, graph_list, prune_probability,         \
                                  n_threads);

template <typename SparseNNGraph, typename Distance>
auto diversify_impl(const SparseNNGraph &graph, const Distance &distance,
                    double prune_probability, std::size_t n_threads = 0)
    -> SparseNNGraph {
  if (n_threads > 0) {
    RPProgress progress(1, false);
    ParallelRand rand;
    return tdoann::remove_long_edges<RParallel>(
        graph, distance, rand, prune_probability, progress, n_threads);
  } else {
    RRand rand;
    return tdoann::remove_long_edges(graph, distance, rand, prune_probability);
  }
}

template <typename Distance>
List diversify_impl(NumericMatrix data, List graph_list,
                    double prune_probability, std::size_t n_threads = 0) {
  auto distance = r_to_dist<Distance>(data);
  auto graph = r_to_sparse_graph<Distance>(graph_list);

  auto diversified =
      diversify_impl(graph, distance, prune_probability, n_threads);

  return sparse_graph_to_r(diversified);
}

// [[Rcpp::export]]
List diversify_cpp(NumericMatrix data, List graph_list,
                   const std::string &metric = "euclidean",
                   double prune_probability = 1.0, std::size_t n_threads = 0) {
  DISPATCH_ON_DISTANCES(DIVERSIFY_IMPL)
}

struct Dummy {
  using Output = double;
  using Index = std::size_t;
};

// [[Rcpp::export]]
List merge_graph_lists_cpp(Rcpp::List gl1, Rcpp::List gl2) {
  auto g1 = r_to_sparse_graph<Dummy>(gl1);
  auto g2 = r_to_sparse_graph<Dummy>(gl2);

  auto g_merge = tdoann::merge_graphs(g1, g2);

  return sparse_graph_to_r(g_merge);
}

template <typename SparseNNGraph>
auto degree_prune_impl(const SparseNNGraph &graph, std::size_t max_degree,
                       std::size_t n_threads = 0) -> SparseNNGraph {
  RPProgress progress(1, false);
  if (n_threads > 0) {
    return tdoann::degree_prune<RParallel>(graph, max_degree, progress,
                                           n_threads);
  } else {
    return tdoann::degree_prune(graph, max_degree, progress);
  }
}

// [[Rcpp::export]]
List degree_prune_cpp(Rcpp::List graph_list, std::size_t max_degree,
                      std::size_t n_threads = 0) {
  auto graph = r_to_sparse_graph<Dummy>(graph_list);
  auto pruned = degree_prune_impl(graph, max_degree, n_threads);
  return sparse_graph_to_r(pruned);
}

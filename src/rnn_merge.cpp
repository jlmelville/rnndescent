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

// NOLINTBEGIN(modernize-use-trailing-return-type)

#include <Rcpp.h>

#include "RcppPerpendicular.h"
#include "rnn_heaptor.h"
#include "rnn_rtoheap.h"
#include "rnn_util.h"

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

std::pair<IntegerMatrix, NumericMatrix>
extract_from_list(const List &nn_graph) {
  NumericMatrix nn_dist = nn_graph["dist"];
  IntegerMatrix nn_idx = nn_graph["idx"];
  return {nn_idx, nn_dist};
}

template <typename NeighborHeap>
void add_graph(NeighborHeap &heap, const IntegerMatrix &nn_idx,
               const NumericMatrix &nn_dist, bool is_query,
               std::size_t n_threads) {

  auto nn_idxc = clone(nn_idx);

  const constexpr std::size_t block_size = 4096;
  const constexpr std::size_t grain_size = 1;
  if (is_query) {
    r_add_to_query_heap(heap, nn_idxc, nn_dist, n_threads, grain_size,
                        block_size);
  } else {
    r_add_to_knn_heap(heap, nn_idxc, nn_dist, n_threads, grain_size,
                      block_size);
  }
}

auto merge_nn_impl(const IntegerMatrix &nn_idx1, const NumericMatrix &nn_dist1,
                   const IntegerMatrix &nn_idx2, const NumericMatrix &nn_dist2,
                   bool is_query, std::size_t n_threads, bool verbose) -> List {
  tdoann::NNHeap<float> nn_merged(nn_idx1.nrow(), nn_idx1.ncol());

  if (verbose) {
    ts("Merging graphs");
  }
  add_graph(nn_merged, nn_idx1, nn_dist1, is_query, n_threads);
  add_graph(nn_merged, nn_idx2, nn_dist2, is_query, n_threads);

  return heap_to_r(nn_merged, n_threads);
}

auto merge_nn_all_impl(const List &nn_graphs, bool is_query,
                       std::size_t n_threads, bool verbose = false) -> List {
  const auto n_graphs = nn_graphs.size();

  RPProgress progress(static_cast<std::size_t>(n_graphs), verbose);

  auto [nn_idx, nn_dist] = extract_from_list(nn_graphs[0]);
  tdoann::NNHeap<float> nn_merged(nn_idx.nrow(), nn_idx.ncol());

  add_graph(nn_merged, nn_idx, nn_dist, is_query, n_threads);

  progress.iter_finished();

  // iterate over other graphs
  for (auto i = 1; i < n_graphs; i++) {
    auto [nn_idxi, nn_disti] = extract_from_list(nn_graphs[i]);

    add_graph(nn_merged, nn_idxi, nn_disti, is_query, n_threads);

    if (progress.check_interrupt()) {
      break;
    }
    progress.iter_finished();
  }

  return heap_to_r(nn_merged, n_threads);
}

// [[Rcpp::export]]
List merge_nn(const IntegerMatrix &nn_idx1, const NumericMatrix &nn_dist1,
              const IntegerMatrix &nn_idx2, const NumericMatrix &nn_dist2,
              bool is_query, std::size_t n_threads, bool verbose) {
  return merge_nn_impl(nn_idx1, nn_dist1, nn_idx2, nn_dist2, is_query,
                       n_threads, verbose);
}

// [[Rcpp::export]]
List merge_nn_all(const List &nn_graphs, bool is_query, std::size_t n_threads,
                  bool verbose) {
  return merge_nn_all_impl(nn_graphs, is_query, n_threads, verbose);
}

// NOLINTEND(modernize-use-trailing-return-type)

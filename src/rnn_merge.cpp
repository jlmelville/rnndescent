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
#include "rnn_heaptor.hpp"
#include "rnn_rtoheap.hpp"
#include "rnn_util.hpp"

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

template <typename NeighborHeap> struct SerialHeapImpl {
  template <typename HeapAdd>
  void init(NeighborHeap &heap, IntegerMatrix nn_idx, NumericMatrix nn_dist) {
    const std::size_t block_size = 4096;
    r_to_heap<HeapAdd>(heap, nn_idx, nn_dist, block_size, RNND_MAX_IDX, true);
  }
  auto to_r(NeighborHeap &heap) -> List { return heap_to_r(heap); }
};

template <typename NeighborHeap> struct ParallelHeapImpl {
public:
  explicit ParallelHeapImpl(std::size_t n_threads) : n_threads(n_threads) {}

  template <typename HeapAdd>
  void init(NeighborHeap &heap, IntegerMatrix nn_idx, NumericMatrix nn_dist) {
    const std::size_t block_size = 4096;
    const std::size_t grain_size = 1;
    r_to_heap<HeapAdd>(heap, nn_idx, nn_dist, n_threads, grain_size, block_size,
                       RNND_MAX_IDX, true);
  }
  auto to_r(NeighborHeap &heap) -> List { return heap_to_r(heap, n_threads); }

private:
  std::size_t n_threads;
};

template <typename NeighborHeap, typename MergeImpl, typename HeapAdd>
auto merge_nn_impl(const IntegerMatrix &nn_idx1, const NumericMatrix &nn_dist1,
                   const IntegerMatrix &nn_idx2, const NumericMatrix &nn_dist2,
                   MergeImpl &merge_impl, bool verbose) -> List {
  NeighborHeap nn_merged(nn_idx1.nrow(), nn_idx1.ncol());

  auto nn_idx1c = clone(nn_idx1);
  auto nn_idx2c = clone(nn_idx2);

  if (verbose) {
    ts("Merging graphs");
  }
  merge_impl.template init<HeapAdd>(nn_merged, nn_idx1c, nn_dist1);
  merge_impl.template init<HeapAdd>(nn_merged, nn_idx2c, nn_dist2);

  return merge_impl.to_r(nn_merged);
}

template <typename NeighborHeap, typename MergeImpl, typename HeapAdd>
auto merge_nn_all_impl(List nn_graphs, MergeImpl &merge_impl,
                       bool verbose = false) -> List {
  auto n_graphs = nn_graphs.size();

  List nn_graph = nn_graphs[0];
  NumericMatrix nn_dist = nn_graph["dist"];
  IntegerMatrix nn_idx = nn_graph["idx"];
  auto nn_idxc = clone(nn_idx);

  RPProgress progress(static_cast<std::size_t>(n_graphs), verbose);
  NeighborHeap nn_merged(nn_idxc.nrow(), nn_idxc.ncol());
  merge_impl.template init<HeapAdd>(nn_merged, nn_idxc, nn_dist);
  progress.iter_finished();

  // iterate over other graphs
  for (auto i = 1; i < n_graphs; i++) {
    List nn_graphi = nn_graphs[i];
    NumericMatrix nn_disti = nn_graphi["dist"];
    IntegerMatrix nn_idxi = nn_graphi["idx"];
    auto nn_idxic = clone(nn_idxi);
    merge_impl.template init<HeapAdd>(nn_merged, nn_idxic, nn_disti);
    TDOANN_ITERFINISHED()
  }

  return merge_impl.to_r(nn_merged);
}

#define CONFIGURE_MERGE(NEXT_MACRO)                                            \
  if (n_threads > 0) {                                                         \
    using MergeImpl = ParallelHeapImpl<tdoann::NNHeap<float>>;                 \
    MergeImpl merge_impl(n_threads);                                           \
    if (is_query) {                                                            \
      using HeapAdd = tdoann::HeapAddQuery;                                    \
      NEXT_MACRO();                                                            \
    }                                                                          \
    using HeapAdd = tdoann::LockingHeapAddSymmetric;                           \
    NEXT_MACRO();                                                              \
  }                                                                            \
  using MergeImpl = SerialHeapImpl<tdoann::NNHeap<float>>;                     \
  MergeImpl merge_impl;                                                        \
  if (is_query) {                                                              \
    using HeapAdd = tdoann::HeapAddQuery;                                      \
    NEXT_MACRO();                                                              \
  }                                                                            \
  using HeapAdd = tdoann::HeapAddSymmetric;                                    \
  NEXT_MACRO();

#define MERGE_NN()                                                             \
  return merge_nn_impl<tdoann::NNHeap<float>, MergeImpl, HeapAdd>(             \
      nn_idx1, nn_dist1, nn_idx2, nn_dist2, merge_impl, verbose);

#define MERGE_NN_ALL()                                                         \
  return merge_nn_all_impl<tdoann::NNHeap<float>, MergeImpl, HeapAdd>(         \
      nn_graphs, merge_impl, verbose);

// [[Rcpp::export]]
List merge_nn(const IntegerMatrix &nn_idx1, const NumericMatrix &nn_dist1,
              const IntegerMatrix &nn_idx2, const NumericMatrix &nn_dist2,
              bool is_query, std::size_t n_threads, bool verbose) {
  CONFIGURE_MERGE(MERGE_NN);
}

// [[Rcpp::export]]
List merge_nn_all(const List &nn_graphs, bool is_query, std::size_t n_threads,
                  bool verbose) {
  CONFIGURE_MERGE(MERGE_NN_ALL);
}

// NOLINTEND(modernize-use-trailing-return-type)

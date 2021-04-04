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

#include <Rcpp.h>

#include "RcppPerpendicular.h"
#include "rnn_heaptor.h"
#include "rnn_rtoheap.h"
#include "rnn_util.h"

using namespace Rcpp;

template <typename NeighborHeap> struct SerialHeapImpl {
  std::size_t block_size;

  SerialHeapImpl(std::size_t block_size) : block_size(block_size) {}

  template <typename HeapAdd>
  void init(NeighborHeap &heap, IntegerMatrix nn_idx, NumericMatrix nn_dist) {
    r_to_heap_serial<HeapAdd>(heap, nn_idx, nn_dist, block_size, RNND_MAX_IDX,
                              true);
  }
  void sort_heap(NeighborHeap &heap) { heap.deheap_sort(); }
};

template <typename NeighborHeap> struct ParallelHeapImpl {
  std::size_t block_size;
  std::size_t n_threads;
  std::size_t grain_size;

  ParallelHeapImpl(std::size_t block_size, std::size_t n_threads,
                   std::size_t grain_size)
      : block_size(block_size), n_threads(n_threads), grain_size(grain_size) {}

  template <typename HeapAdd>
  void init(NeighborHeap &heap, IntegerMatrix nn_idx, NumericMatrix nn_dist) {
    r_to_heap_parallel<HeapAdd>(heap, nn_idx, nn_dist, n_threads, grain_size,
                                block_size, RNND_MAX_IDX, true);
  }
  void sort_heap(NeighborHeap &heap) {
    sort_heap_parallel(heap, n_threads, block_size, grain_size);
  }
};

template <typename NeighborHeap, typename MergeImpl, typename HeapAdd>
auto merge_nn_impl(IntegerMatrix nn_idx1, NumericMatrix nn_dist1,
                   IntegerMatrix nn_idx2, NumericMatrix nn_dist2,
                   MergeImpl &merge_impl, bool verbose = false) -> List {
  NeighborHeap nn_merged(nn_idx1.nrow(), nn_idx1.ncol());

  auto nn_idx1c = clone(nn_idx1);
  auto nn_idx2c = clone(nn_idx2);

  if (verbose) {
    ts("Merging graphs");
  }
  merge_impl.template init<HeapAdd>(nn_merged, nn_idx1c, nn_dist1);
  merge_impl.template init<HeapAdd>(nn_merged, nn_idx2c, nn_dist2);

  merge_impl.sort_heap(nn_merged);
  return heap_to_r(nn_merged);
}

template <typename NeighborHeap, typename MergeImpl, typename HeapAdd>
auto merge_nn_all_impl(List nn_graphs, MergeImpl &merge_impl,
                       bool verbose = false) -> List {
  auto n_graphs = static_cast<std::size_t>(nn_graphs.size());

  List nn_graph = nn_graphs[0];
  NumericMatrix nn_dist = nn_graph["dist"];
  IntegerMatrix nn_idx = nn_graph["idx"];
  auto nn_idxc = clone(nn_idx);

  RPProgress progress(n_graphs, verbose);
  NeighborHeap nn_merged(nn_idxc.nrow(), nn_idxc.ncol());
  merge_impl.template init<HeapAdd>(nn_merged, nn_idxc, nn_dist);
  progress.iter_finished();

  // iterate over other graphs
  for (std::size_t i = 1; i < n_graphs; i++) {
    List nn_graphi = nn_graphs[i];
    NumericMatrix nn_disti = nn_graphi["dist"];
    IntegerMatrix nn_idxi = nn_graphi["idx"];
    auto nn_idxic = clone(nn_idxi);
    merge_impl.template init<HeapAdd>(nn_merged, nn_idxic, nn_disti);
    TDOANN_ITERFINISHED()
  }

  merge_impl.sort_heap(nn_merged);
  return heap_to_r(nn_merged);
}

#define CONFIGURE_MERGE(NEXT_MACRO)                                            \
  if (n_threads > 0) {                                                         \
    using MergeImpl = ParallelHeapImpl<tdoann::NNHeap<float>>;                 \
    MergeImpl merge_impl(block_size, n_threads, grain_size);                   \
    if (is_query) {                                                            \
      using HeapAdd = tdoann::HeapAddQuery;                                    \
      NEXT_MACRO();                                                            \
    } else {                                                                   \
      using HeapAdd = tdoann::LockingHeapAddSymmetric;                         \
      NEXT_MACRO();                                                            \
    }                                                                          \
  } else {                                                                     \
    using MergeImpl = SerialHeapImpl<tdoann::NNHeap<float>>;                   \
    MergeImpl merge_impl(block_size);                                          \
    if (is_query) {                                                            \
      using HeapAdd = tdoann::HeapAddQuery;                                    \
      NEXT_MACRO();                                                            \
    } else {                                                                   \
      using HeapAdd = tdoann::HeapAddSymmetric;                                \
      NEXT_MACRO();                                                            \
    }                                                                          \
  }

#define MERGE_NN()                                                             \
  return merge_nn_impl<tdoann::NNHeap<float>, MergeImpl, HeapAdd>(             \
      nn_idx1, nn_dist1, nn_idx2, nn_dist2, merge_impl, verbose);

#define MERGE_NN_ALL()                                                         \
  return merge_nn_all_impl<tdoann::NNHeap<float>, MergeImpl, HeapAdd>(         \
      nn_graphs, merge_impl, verbose);

// [[Rcpp::export]]
List merge_nn(IntegerMatrix nn_idx1, NumericMatrix nn_dist1,
              IntegerMatrix nn_idx2, NumericMatrix nn_dist2, bool is_query,
              std::size_t block_size, std::size_t n_threads,
              std::size_t grain_size = 1, bool verbose = false) {
  CONFIGURE_MERGE(MERGE_NN);
}

// [[Rcpp::export]]
List merge_nn_all(List nn_graphs, bool is_query, std::size_t block_size,
                  std::size_t n_threads, std::size_t grain_size = 1,
                  bool verbose = false) {
  CONFIGURE_MERGE(MERGE_NN_ALL);
}

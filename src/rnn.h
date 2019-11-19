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

#ifndef RNND_RNN_H
#define RNND_RNN_H

#include <limits>

#include <Rcpp.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include "tdoann/distance.h"
#include "tdoann/graphupdate.h"
#include "tdoann/heap.h"

#define DISPATCH_ON_DISTANCES(NEXT_MACRO)                                      \
  if (metric == "euclidean") {                                                 \
    using Distance = tdoann::Euclidean<float, float>;                          \
    NEXT_MACRO()                                                               \
  } else if (metric == "l2sqr") {                                              \
    using Distance = tdoann::L2Sqr<float, float>;                              \
    NEXT_MACRO()                                                               \
  } else if (metric == "cosine") {                                             \
    using Distance = tdoann::Cosine<float, float>;                             \
    NEXT_MACRO()                                                               \
  } else if (metric == "manhattan") {                                          \
    using Distance = tdoann::Manhattan<float, float>;                          \
    NEXT_MACRO()                                                               \
  } else if (metric == "hamming") {                                            \
    using Distance = tdoann::Hamming<uint8_t, std::size_t>;                    \
    NEXT_MACRO()                                                               \
  } else {                                                                     \
    Rcpp::stop("Bad metric");                                                  \
  }

void print_time(bool print_date = false);
void ts(const std::string &msg);

// Sums the distances in a neighbor heap as a way of measuring progress.
// Useful for diagnostic purposes
struct HeapSumProgress {
  tdoann::NeighborHeap &neighbor_heap;
  const std::size_t n_iters;
  std::size_t iter;
  bool verbose;

  bool is_aborted;

  HeapSumProgress(tdoann::NeighborHeap &neighbor_heap, std::size_t n_iters,
                  bool verbose = false);
  void block_finished();
  void iter_finished();
  void stopping_early();
  bool check_interrupt();
  void converged(std::size_t n_updates, double tol);
  double dist_sum() const;
};

struct RPProgress {
  const std::size_t scale;
  Progress progress;
  const std::size_t n_iters;
  const double n_blocks;
  bool verbose;

  std::size_t iter;
  std::size_t block;

  bool is_aborted;

  RPProgress(std::size_t n_iters, std::size_t n_blocks, bool verbose);
  RPProgress(std::size_t n_iters, bool verbose);
  void block_finished();
  void iter_finished();
  void stopping_early();
  bool check_interrupt();
  void converged(std::size_t n_updates, double tol);
  // convert float between 0...n_iters to int from 0...scale
  int scaled(double d);
};

struct HeapAddSymmetric {
  template <typename NbrHeap>
  static void push(NbrHeap &current_graph, std::size_t ref, std::size_t query,
                   double d) {
    current_graph.checked_push_pair(ref, d, query);
  }
};

struct HeapAddQuery {
  template <typename NbrHeap>
  static void push(NbrHeap &current_graph, std::size_t ref, std::size_t query,
                   double d) {
    current_graph.checked_push(ref, d, query);
  }
};

// input idx R matrix is 1-indexed and transposed
// output heap index is 0-indexed
template <typename HeapAdd, typename NbrHeap,
          typename IdxMatrix = Rcpp::IntegerMatrix,
          typename DistMatrix = Rcpp::NumericMatrix>
void r_to_heap(NbrHeap &current_graph, IdxMatrix nn_idx, DistMatrix nn_dist,
               const std::size_t begin, const std::size_t end,
               const int max_idx = std::numeric_limits<int>::max()) {
  const std::size_t n_nbrs = nn_idx.ncol();

  for (std::size_t i = begin; i < end; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      const int k = nn_idx(i, j) - 1;
      if (k < 0 || k > max_idx) {
        Rcpp::stop("Bad indexes in input");
      }
      double d = nn_dist(i, j);
      HeapAdd::push(current_graph, i, k, d);
    }
  }
}

template <typename HeapAdd, typename NbrHeap>
void r_to_heap(NbrHeap &current_graph, Rcpp::IntegerMatrix nn_idx,
               Rcpp::NumericMatrix nn_dist,
               const int max_idx = std::numeric_limits<int>::max()) {
  const std::size_t n_points = nn_idx.nrow();
  r_to_heap<HeapAdd, NbrHeap>(current_graph, nn_idx, nn_dist, 0, n_points,
                              max_idx);
}

// input heap index is 0-indexed
// output idx R matrix is 1-indexed and untransposed
template <typename NbrHeap>
void heap_to_r(const NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
               Rcpp::NumericMatrix nn_dist) {
  const std::size_t n_points = heap.n_points;
  const std::size_t n_nbrs = heap.n_nbrs;

  for (std::size_t i = 0; i < n_points; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      nn_idx(i, j) = heap.index(i, j) + 1;
      nn_dist(i, j) = heap.distance(i, j);
    }
  }
}

// input heap index is 0-indexed
// output idx R matrix is 1-indexed and transposed
template <typename NbrHeap> Rcpp::List heap_to_r(const NbrHeap &heap) {
  const std::size_t n_points = heap.n_points;
  const std::size_t n_nbrs = heap.n_nbrs;

  Rcpp::IntegerMatrix nn_idx(n_points, n_nbrs);
  Rcpp::NumericMatrix nn_dist(n_points, n_nbrs);

  heap_to_r(heap, nn_idx, nn_dist);

  return Rcpp::List::create(Rcpp::Named("idx") = nn_idx,
                            Rcpp::Named("dist") = nn_dist);
}

template <typename HeapAdd, typename NbrHeap = tdoann::SimpleNeighborHeap>
void sort_knn_graph(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist) {
  const std::size_t n_points = nn_idx.nrow();
  const std::size_t n_nbrs = nn_idx.ncol();

  NbrHeap heap(n_points, n_nbrs);
  r_to_heap<HeapAdd>(heap, nn_idx, nn_dist);
  heap.deheap_sort();
  heap_to_r(heap, nn_idx, nn_dist);
}

#endif // RNND_RNN_H

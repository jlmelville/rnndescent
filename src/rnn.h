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

#ifndef RNN_H
#define RNN_H

#include <limits>

#include <Rcpp.h>

#include "tdoann/distance.h"
#include "tdoann/heap.h"
#include "tdoann/typedefs.h"

#include "rnn_progress.h"

#define DISPATCH_ON_DISTANCES(NEXT_MACRO)                                      \
  if (metric == "euclidean") {                                                 \
    using Distance = tdoann::Euclidean<float, float>;                          \
    NEXT_MACRO()                                                               \
  } else if (metric == "l2sqr") {                                              \
    using Distance = tdoann::L2Sqr<float, float>;                              \
    NEXT_MACRO()                                                               \
  } else if (metric == "cosine") {                                             \
    using Distance = tdoann::CosineSelf<float, float>;                         \
    NEXT_MACRO()                                                               \
  } else if (metric == "manhattan") {                                          \
    using Distance = tdoann::Manhattan<float, float>;                          \
    NEXT_MACRO()                                                               \
  } else if (metric == "hamming") {                                            \
    using Distance = tdoann::HammingSelf<uint8_t, std::size_t>;                \
    NEXT_MACRO()                                                               \
  } else {                                                                     \
    Rcpp::stop("Bad metric");                                                  \
  }

#define DISPATCH_ON_QUERY_DISTANCES(NEXT_MACRO)                                \
  if (metric == "euclidean") {                                                 \
    using Distance = tdoann::Euclidean<float, float>;                          \
    NEXT_MACRO()                                                               \
  } else if (metric == "l2sqr") {                                              \
    using Distance = tdoann::L2Sqr<float, float>;                              \
    NEXT_MACRO()                                                               \
  } else if (metric == "cosine") {                                             \
    using Distance = tdoann::CosineQuery<float, float>;                        \
    NEXT_MACRO()                                                               \
  } else if (metric == "manhattan") {                                          \
    using Distance = tdoann::Manhattan<float, float>;                          \
    NEXT_MACRO()                                                               \
  } else if (metric == "hamming") {                                            \
    using Distance = tdoann::HammingQuery<uint8_t, std::size_t>;               \
    NEXT_MACRO()                                                               \
  } else {                                                                     \
    Rcpp::stop("Bad metric");                                                  \
  }

/* Structs */

struct Empty {};

template <typename Distance> struct KnnQueryFactory {
  using DataVec = std::vector<typename Distance::Input>;

  DataVec reference_vec;
  DataVec query_vec;
  int nrow;
  int ndim;

  KnnQueryFactory(Rcpp::NumericMatrix reference, Rcpp::NumericMatrix query)
      : reference_vec(Rcpp::as<DataVec>(Rcpp::transpose(reference))),
        query_vec(Rcpp::as<DataVec>(Rcpp::transpose(query))),
        nrow(query.nrow()), ndim(query.ncol()) {}

  Distance create_distance() const {
    return Distance(reference_vec, query_vec, ndim);
  }

  Rcpp::NumericMatrix create_distance_matrix(int k) const {
    return Rcpp::NumericMatrix(k, nrow);
  }

  Rcpp::IntegerMatrix create_index_matrix(int k) const {
    return Rcpp::IntegerMatrix(k, nrow);
  }
};

template <typename Distance> struct KnnBuildFactory {
  using DataVec = std::vector<typename Distance::Input>;

  DataVec data_vec;
  int nrow;
  int ndim;

  KnnBuildFactory(Rcpp::NumericMatrix data)
      : data_vec(Rcpp::as<DataVec>(Rcpp::transpose(data))), nrow(data.nrow()),
        ndim(data.ncol()) {}

  Distance create_distance() const { return Distance(data_vec, ndim); }

  Rcpp::NumericMatrix create_distance_matrix(int k) const {
    return Rcpp::NumericMatrix(k, nrow);
  }

  Rcpp::IntegerMatrix create_index_matrix(int k) const {
    return Rcpp::IntegerMatrix(k, nrow);
  }
};

void print_time(bool print_date = false);
void ts(const std::string &msg);

struct HeapAddSymmetric {
  template <typename NbrHeap>
  void push(NbrHeap &heap, std::size_t ref, std::size_t query, double d) {
    heap.checked_push_pair(ref, d, query);
  }
};

struct HeapAddQuery {
  template <typename NbrHeap>
  void push(NbrHeap &heap, std::size_t ref, std::size_t query, double d) {
    heap.checked_push(ref, d, query);
  }
};

// input idx R matrix is 1-indexed and transposed
// output heap index is 0-indexed
template <typename HeapAdd, typename NbrHeap,
          typename IdxMatrix = Rcpp::IntegerMatrix,
          typename DistMatrix = Rcpp::NumericMatrix>
void r_to_heap(NbrHeap &current_graph, IdxMatrix nn_idx, DistMatrix nn_dist,
               const std::size_t begin, const std::size_t end,
               HeapAdd &heap_add,
               const int max_idx = (std::numeric_limits<int>::max)()) {
  const std::size_t n_nbrs = nn_idx.ncol();

  for (std::size_t i = begin; i < end; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      const int k = nn_idx(i, j) - 1;
      if (k < 0 || k > max_idx) {
        Rcpp::stop("Bad indexes in input: " + std::to_string(k));
      }
      double d = nn_dist(i, j);
      heap_add.push(current_graph, i, k, d);
    }
  }
}

template <typename HeapAdd, typename NbrHeap>
void r_to_heap(NbrHeap &current_graph, Rcpp::IntegerMatrix nn_idx,
               Rcpp::NumericMatrix nn_dist,
               const int max_idx = std::numeric_limits<int>::max()) {
  const std::size_t n_points = nn_idx.nrow();
  HeapAdd heap_add;
  r_to_heap<HeapAdd>(current_graph, nn_idx, nn_dist, 0, n_points, heap_add,
                     max_idx);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap,
          typename IdxMatrix = Rcpp::IntegerMatrix,
          typename DistMatrix = Rcpp::NumericMatrix, typename Base = Empty>
struct RToHeapWorker : public Base {
  NbrHeap &heap;
  IdxMatrix nn_idx;
  DistMatrix nn_dist;
  int max_idx;
  HeapAdd heap_add;

  RToHeapWorker(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                Rcpp::NumericMatrix nn_dist,
                int max_idx = (std::numeric_limits<int>::max)())
      : heap(heap), nn_idx(nn_idx), nn_dist(nn_dist), max_idx(max_idx),
        heap_add() {}

  void operator()(std::size_t begin, std::size_t end) {
    r_to_heap<HeapAdd, NbrHeap, IdxMatrix, DistMatrix>(
        heap, nn_idx, nn_dist, begin, end, heap_add, max_idx);
  }
};

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void r_to_heap_serial(NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
                      Rcpp::NumericMatrix nn_dist, std::size_t block_size,
                      int max_idx = (std::numeric_limits<int>::max)()) {
  RToHeapWorker<HeapAdd, NbrHeap, Rcpp::IntegerMatrix, Rcpp::NumericMatrix,
                Empty>
      worker(heap, nn_idx, nn_dist, max_idx);
  InterruptableProgress progress;
  const std::size_t n_points = nn_idx.nrow();
  batch_serial_for(worker, progress, n_points, block_size);
}

// input heap index is 0-indexed
// output idx R matrix is 1-indexed and untransposed
template <typename NbrHeap>
void heap_to_r(const NbrHeap &heap, Rcpp::IntegerMatrix nn_idx,
               Rcpp::NumericMatrix nn_dist) {
  const std::size_t n_points = heap.n_points;
  const std::size_t n_nbrs = heap.n_nbrs;

  for (std::size_t i = 0; i < n_points; i++) {
    const std::size_t innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      const std::size_t ij = innbrs + j;
      nn_idx(i, j) = heap.idx[ij] + 1;
      nn_dist(i, j) = heap.dist[ij];
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

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void sort_knn_graph(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist) {
  const std::size_t n_points = nn_idx.nrow();
  const std::size_t n_nbrs = nn_idx.ncol();

  NbrHeap heap(n_points, n_nbrs);
  r_to_heap_serial<HeapAdd>(heap, nn_idx, nn_dist, 1000);
  heap.deheap_sort();
  heap_to_r(heap, nn_idx, nn_dist);
}

#endif // RNN_H

#include <iostream>
#include <utility>

#include <Rcpp.h>

#include "tdoann/heap.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

using namespace Rcpp;

template <typename T> std::pair<T, T> pair_dmax() {
  return std::make_pair((std::numeric_limits<T>::max)(),
                        (std::numeric_limits<T>::max)());
}

// [[Rcpp::export]]
List local_scaled_nbrs(IntegerMatrix idx, NumericMatrix dist,
                       NumericMatrix sdist, std::size_t n_nbrs,
                       std::size_t n_threads = 0) {
  if (n_nbrs > static_cast<std::size_t>(idx.ncol())) {
    stop("Can't return more neighbors than is in the original graph");
  }

  using Idx = typename DummyDistance::Index;
  using Out = typename DummyDistance::Output;

  auto idx_vec = r_to_idxt<Idx>(idx);
  auto dist_vec = r_to_vect<Out>(dist);
  auto sdist_vec = r_to_vect<Out>(sdist);

  // Pair up the scaled and unscaled distances
  using DPair = std::pair<Out, Out>;
  std::vector<DPair> dpairs;
  dpairs.reserve(dist_vec.size());
  for (std::size_t i = 0; i < dist_vec.size(); i++) {
    dpairs.emplace_back(sdist_vec[i], dist_vec[i]);
  }

  // Create an unsorted top-k neighbor heap of size n_nbrs using the paired distances as values
  using PairNbrHeap = tdoann::NNHeap<DPair, Idx, pair_dmax>;
  tdoann::HeapAddQuery heap_add;
  const std::size_t n_points = idx.nrow();
  const std::size_t block_size = 100;
  const bool transpose = false;
  PairNbrHeap pair_heap(n_points, n_nbrs);
  tdoann::vec_to_heap<tdoann::HeapAddQuery, RInterruptableProgress>(
      pair_heap, idx_vec, n_points, dpairs, block_size, transpose);

  tdoann::NNHeap<Out, Idx> nn_heap(n_points, n_nbrs);
  for (std::size_t i = 0; i < n_points; i++) {
    for (std::size_t j = 0; j < n_nbrs; j++) {
      heap_add.push(nn_heap, i, pair_heap.index(i, j), pair_heap.distance(i, j).second);
    }
  }

  return heap_to_r(nn_heap);
}

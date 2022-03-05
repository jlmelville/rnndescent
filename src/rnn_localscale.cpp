#include <Rcpp.h>

#include "tdoann/heap.h"
#include "tdoann/hubness.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

using namespace Rcpp;

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

  std::size_t n_points = idx.nrow();
  tdoann::NNHeap<Out, Idx> nn_heap(n_points, n_nbrs);
  tdoann::local_scale<RInterruptableProgress>(idx_vec, dist_vec, sdist_vec,
                                              nn_heap);

  return heap_to_r(nn_heap);
}

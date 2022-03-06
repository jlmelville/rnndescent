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

// [[Rcpp::export]]
NumericVector get_local_scales_cpp(NumericMatrix dist, std::size_t k_begin,
                                   std::size_t k_end) {
  if (k_begin < 1) {
    stop("k_begin must be >= 1");
  }
  if (k_end < k_begin) {
    stop("k_end must be >= k_begin");
  }
  std::size_t n_nbrs = dist.ncol();
  if (k_begin > n_nbrs) {
    stop("k_begin must be <= number of neighbors");
  }
  if (k_end > n_nbrs) {
    stop("k_end must be <= number of neighbors");
  }
  auto dist_vec = r_to_vect<double>(dist);

  // for C++ code k_begin/end should be converted to zero index (subtract 1)
  // and end should be one past the final element (those last two cancel out)
  auto k_begin0 = k_begin - 1;
  auto k_end0 = k_end;

  std::vector<double> local_scales =
      tdoann::get_local_scales(dist_vec, n_nbrs, k_begin0, k_end0);
  NumericVector res(local_scales.begin(), local_scales.end());
  return res;
}

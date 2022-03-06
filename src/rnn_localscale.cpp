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
                       std::size_t n_scaled_nbrs, std::size_t k_begin,
                       std::size_t k_end, std::size_t n_threads = 0) {
  std::size_t n_orig_nbrs = idx.ncol();
  if (n_scaled_nbrs > n_orig_nbrs) {
    stop("Can't return more neighbors than is in the original graph");
  }

  using Idx = typename DummyDistance::Index;
  using Out = typename DummyDistance::Output;

  auto idx_vec = r_to_idxt<Idx>(idx);
  auto dist_vec = r_to_vect<Out>(dist);
  std::size_t n_points = idx.nrow();

  // for C++ code k_begin/end should be converted to zero index (subtract 1)
  // and end should be one past the final element (those last two cancel out)
  auto k_begin0 = k_begin - 1;
  auto k_end0 = k_end;
  Out min_scale = 1e-10;
  auto local_scales = tdoann::get_local_scales(dist_vec, n_orig_nbrs, k_begin0,
                                               k_end0, min_scale);

  auto sdist_vec = tdoann::local_scaled_distances(idx_vec, dist_vec,
                                                  n_orig_nbrs, local_scales);

  tdoann::NNHeap<Out, Idx> nn_heap(n_points, n_scaled_nbrs);
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
  auto min_scale = 1e-10;

  std::vector<double> local_scales =
      tdoann::get_local_scales(dist_vec, n_nbrs, k_begin0, k_end0, min_scale);
  NumericVector res(local_scales.begin(), local_scales.end());
  return res;
}

// [[Rcpp::export]]
NumericVector local_scale_distances_cpp(IntegerMatrix idx, NumericMatrix dist,
                                        NumericVector local_scales) {
  using Idx = typename DummyDistance::Index;
  using Out = typename DummyDistance::Output;

  auto idx_vec = r_to_idxt<Idx>(idx);
  auto dist_vec = r_to_vect<Out>(dist);
  auto local_scales_vec = r_to_vec<Out>(local_scales);

  std::size_t n_nbrs = idx.ncol();
  auto sdist_vec = tdoann::local_scaled_distances(idx_vec, dist_vec, n_nbrs,
                                                  local_scales_vec);

  NumericMatrix res(n_nbrs, idx.nrow(), sdist_vec.begin());
  return transpose(res);
}

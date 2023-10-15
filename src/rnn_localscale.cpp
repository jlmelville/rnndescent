#include <Rcpp.h>

#include "tdoann/heap.h"
#include "tdoann/hubness.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::stop;

// NOLINTBEGIN(modernize-use-trailing-return-type)

// [[Rcpp::export]]
List local_scaled_nbrs(const IntegerMatrix &idx, const NumericMatrix &dist,
                       std::size_t n_scaled_nbrs, std::size_t k_begin,
                       std::size_t k_end, bool ret_scales = false,
                       std::size_t n_threads = 0) {
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
  constexpr Out MIN_SCALE = 1e-10;

  RInterruptableProgress progress;
  RParallelExecutor executor;

  auto local_scales =
      tdoann::get_local_scales(dist_vec, n_orig_nbrs, k_begin0, k_end0,
                               MIN_SCALE, n_threads, progress, executor);

  auto sdist_vec = tdoann::local_scaled_distances(
      idx_vec, dist_vec, n_orig_nbrs, local_scales, n_threads, progress,
      executor);

  tdoann::NNHeap<Out, Idx> nn_heap(n_points, n_scaled_nbrs);
  tdoann::local_scale(idx_vec, dist_vec, sdist_vec, nn_heap, n_threads,
                      progress, executor);
  auto res = heap_to_r(nn_heap, n_threads, progress, executor);
  if (ret_scales) {
    NumericVector scalesr(local_scales.begin(), local_scales.end());
    res["scales"] = scalesr;
  }
  return res;
}

// NOLINTEND(modernize-use-trailing-return-type)

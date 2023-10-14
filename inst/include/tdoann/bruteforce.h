// BSD 2-Clause License
//
// Copyright 2019 James Melville
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// OF SUCH DAMAGE.

#ifndef TDOANN_BRUTE_FORCE_H
#define TDOANN_BRUTE_FORCE_H

#include <cmath>
#include <vector>

#include "heap.h"
#include "nngraph.h"
#include "parallel.h"
// #include "progress.h"

namespace tdoann {

template <typename Distance>
void nnbf_query(
    NNHeap<typename Distance::Output, typename Distance::Index> &neighbor_heap,
    Distance &distance, std::size_t begin, std::size_t end) {

  std::size_t n_ref_points = distance.nx;
  for (std::size_t ref = 0; ref < n_ref_points; ref++) {
    for (std::size_t query = begin; query < end; query++) {
      typename Distance::Output dist_rq = distance(ref, query);
      if (neighbor_heap.accepts(query, dist_rq)) {
        neighbor_heap.unchecked_push(query, dist_rq, ref);
      }
    }
  }
}

template <typename Distance, typename Parallel>
auto nnbf_query(Distance &distance, typename Distance::Index n_nbrs,
                std::size_t n_threads, ProgressBase &progress)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  NNHeap<typename Distance::Output, typename Distance::Index> neighbor_heap(
      distance.ny, n_nbrs);
  auto worker = [&](std::size_t begin, std::size_t end) {
    nnbf_query(neighbor_heap, distance, begin, end);
  };
  progress.set_n_iters(1);
  ExecutionParams exec_params{64};
  batch_parallel_for<Parallel>(worker, neighbor_heap.n_points, n_threads,
                               exec_params, progress);
  sort_heap<Parallel>(neighbor_heap, n_threads, progress);
  return heap_to_graph(neighbor_heap);
}

// convert from 1D index k of upper triangular matrix of size n to 2D index i,j
// https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
// e.g. for n = 5:
// k = 0  -> i = 0, j = 0
// k = 1  -> i = 0, j = 1
// k = 4  -> i = 0, j = 4
// k = 5  -> i = 1, j = 1
// k = 14 -> i = 4, j = 4
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-identifier-length,readability-magic-numbers)
inline void upper_tri_2d(std::size_t k, std::size_t n, std::size_t &i,
                         std::size_t &j) {
  i = n - 1 -
      static_cast<int>(
          sqrt(static_cast<double>(-8 * k + 4 * n * (n + 1) - 7)) / 2 - 0.5);
  j = k - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-identifier-length,readability-magic-numbers)

template <typename Distance>
void nnbf_impl(
    Distance &distance,
    NNHeap<typename Distance::Output, typename Distance::Index> &neighbor_heap,
    std::size_t begin, std::size_t end) {
  const std::size_t n_points = neighbor_heap.n_points;

  std::size_t idx_i{0};
  std::size_t idx_j{0};
  upper_tri_2d(begin, n_points, idx_i, idx_j);

  for (std::size_t k = begin; k < end; k++) {
    typename Distance::Output dist_ij = distance(idx_i, idx_j);
    if (neighbor_heap.accepts(idx_i, dist_ij)) {
      neighbor_heap.unchecked_push(idx_i, dist_ij, idx_j);
    }
    if (idx_i != idx_j && neighbor_heap.accepts(idx_j, dist_ij)) {
      neighbor_heap.unchecked_push(idx_j, dist_ij, idx_i);
    }
    ++idx_j;
    if (idx_j == n_points) {
      ++idx_i;
      idx_j = idx_i;
    }
  }
}

template <typename Distance, typename Parallel>
auto brute_force_build(Distance &distance, typename Distance::Index n_nbrs,
                       std::size_t n_threads, ProgressBase &progress)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  if (n_threads > 0) {
    return nnbf_query<Distance, Parallel>(distance, n_nbrs, n_threads,
                                          progress);
  }
  NNHeap<typename Distance::Output, typename Distance::Index> neighbor_heap(
      distance.ny, n_nbrs);
  auto worker = [&](std::size_t begin, std::size_t end) {
    nnbf_impl(distance, neighbor_heap, begin, end);
  };
  progress.set_n_iters(1);
  const std::size_t n_points = neighbor_heap.n_points;
  // in single-threaded case work is divided up across all unique pairs
  // (including the self-distance)
  const std::size_t n_pairs = (n_points * (n_points + 1)) / 2;
  ExecutionParams exec_params{2048};
  batch_parallel_for<Parallel>(worker, n_pairs, n_threads, exec_params,
                               progress);
  sort_heap(neighbor_heap);
  return heap_to_graph(neighbor_heap);
}

template <typename Distance, typename Parallel>
auto brute_force_query(Distance &distance, typename Distance::Index n_nbrs,
                       std::size_t n_threads, ProgressBase &progress)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  return nnbf_query<Distance, Parallel>(distance, n_nbrs, n_threads, progress);
}

} // namespace tdoann
#endif // TDOANN_BRUTE_FORCE_H

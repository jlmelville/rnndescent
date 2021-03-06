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

#include <vector>

#include "heap.h"
#include "nngraph.h"
#include "parallel.h"
#include "progress.h"

namespace tdoann {

template <typename Distance>
void nnbf_query(
    NNHeap<typename Distance::Output, typename Distance::Index> &neighbor_heap,
    Distance &distance, std::size_t begin, std::size_t end) {

  std::size_t n_ref_points = distance.nx;
  for (std::size_t ref = 0; ref < n_ref_points; ref++) {
    for (std::size_t query = begin; query < end; query++) {
      typename Distance::Output d = distance(ref, query);
      if (neighbor_heap.accepts(query, d)) {
        neighbor_heap.unchecked_push(query, d, ref);
      }
    }
  }
}

template <typename Distance, typename Progress, typename Parallel>
auto nnbf_query(Distance &distance, typename Distance::Index n_nbrs,
                std::size_t n_threads = 0, bool verbose = false)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  NNHeap<typename Distance::Output, typename Distance::Index> neighbor_heap(
      distance.ny, n_nbrs);
  auto worker = [&](std::size_t begin, std::size_t end) {
    nnbf_query(neighbor_heap, distance, begin, end);
  };
  Progress progress(1, verbose);
  const std::size_t block_size = 64;
  const std::size_t grain_size = 1;
  batch_parallel_for<Parallel>(worker, progress, neighbor_heap.n_points,
                               block_size, n_threads, grain_size);
  sort_heap(neighbor_heap, block_size, n_threads, grain_size);
  return heap_to_graph(neighbor_heap);
}

template <typename Distance, typename Progress>
auto nnbf_query(Distance &distance, typename Distance::Index n_nbrs,
                bool verbose)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  NNHeap<typename Distance::Output, typename Distance::Index> neighbor_heap(
      distance.ny, n_nbrs);
  auto worker = [&](std::size_t begin, std::size_t end) {
    nnbf_query(neighbor_heap, distance, begin, end);
  };
  Progress progress(1, verbose);
  const std::size_t block_size = 64;
  batch_serial_for(worker, progress, neighbor_heap.n_points, block_size);
  sort_heap(neighbor_heap);
  return heap_to_graph(neighbor_heap);
}

template <typename Distance>
void nnbf_impl(
    Distance &distance, typename Distance::Index n_nbrs, bool verbose,
    NNHeap<typename Distance::Output, typename Distance::Index> &neighbor_heap,
    std::size_t begin, std::size_t end) {
  const std::size_t n = neighbor_heap.n_points;

  // Convert from the upper triangular index k back to i, j (including diagonal)
  // e.g. for N = 5:
  // k = 0  -> i = 0, j = 0
  // k = 1  -> i = 0, j = 1
  // k = 4  -> i = 0, j = 4
  // k = 5  -> i = 1, j = 1
  // k = 14 -> i = 4, j = 4
  std::size_t i = n - 1 - int(sqrt(-8 * begin + 4 * n * (n + 1) - 7) / 2 - 0.5);
  std::size_t j = begin - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
  for (std::size_t k = begin; k < end; k++) {
    typename Distance::Output d = distance(i, j);
    if (neighbor_heap.accepts(i, d)) {
      neighbor_heap.unchecked_push(i, d, j);
    }
    if (i != j && neighbor_heap.accepts(j, d)) {
      neighbor_heap.unchecked_push(j, d, i);
    }
    ++j;
    if (j == n) {
      ++i;
      j = i;
    }
  }
}

template <typename Distance, typename Progress, typename Parallel>
auto brute_force_build(const std::vector<typename Distance::Input> &data,
                       std::size_t ndim, typename Distance::Index n_nbrs,
                       std::size_t n_threads = 0, bool verbose = false)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  Distance distance(data, ndim);

  if (n_threads > 0) {
    return nnbf_query<Distance, Progress, Parallel>(distance, n_nbrs, n_threads,
                                                    verbose);
  } else {
    NNHeap<typename Distance::Output, typename Distance::Index> neighbor_heap(
        distance.ny, n_nbrs);
    auto worker = [&](std::size_t begin, std::size_t end) {
      nnbf_impl(distance, n_nbrs, verbose, neighbor_heap, begin, end);
    };
    Progress progress(1, verbose);
    const std::size_t n = neighbor_heap.n_points;
    // in single-threaded case work is divided up across all unique pairs
    // (including the self-distance)
    const std::size_t n_pairs = (n * (n + 1)) / 2;
    const std::size_t block_size = 2048;
    batch_serial_for(worker, progress, n_pairs, block_size);

    sort_heap(neighbor_heap);
    return heap_to_graph(neighbor_heap);
  }
}

template <typename Distance, typename Progress, typename Parallel>
auto brute_force_query(const std::vector<typename Distance::Input> &reference,
                       std::size_t ndim,
                       const std::vector<typename Distance::Input> &query,
                       typename Distance::Index n_nbrs,
                       std::size_t n_threads = 0, bool verbose = false)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  Distance distance(reference, query, ndim);

  if (n_threads > 0) {
    return nnbf_query<Distance, Progress, Parallel>(distance, n_nbrs, n_threads,
                                                    verbose);
  } else {
    return nnbf_query<Distance, Progress>(distance, n_nbrs, verbose);
  }
}

} // namespace tdoann
#endif // TDOANN_BRUTE_FORCE_H

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

template <typename Distance, typename Progress>
void nnbf_query_window(
    NNHeap<typename Distance::Output, typename Distance::Index> &neighbor_heap,
    Distance &distance, Progress &progress, std::size_t begin,
    std::size_t end) {

  std::size_t n_ref_points = distance.nx;
  for (std::size_t ref = 0; ref < n_ref_points; ref++) {
    for (std::size_t query = begin; query < end; query++) {
      typename Distance::Output d = distance(ref, query);
      if (neighbor_heap.accepts(query, d)) {
        neighbor_heap.unchecked_push(query, d, ref);
      }
    }
    TDOANN_ITERFINISHED();
  }
}

template <typename Distance>
struct BruteForceWorker : public BatchParallelWorker {

  NNHeap<typename Distance::Output, typename Distance::Index> &neighbor_heap;
  Distance &distance;
  NullProgress progress;

  BruteForceWorker(NNHeap<typename Distance::Output, typename Distance::Index>
                       &neighbor_heap,
                   Distance &distance)
      : neighbor_heap(neighbor_heap), distance(distance), progress() {}

  void operator()(std::size_t begin, std::size_t end) {
    nnbf_query_window(neighbor_heap, distance, progress, begin, end);
  }
};

template <typename Distance, typename Progress, typename Parallel>
auto nnbf_query(Distance &distance, typename Distance::Index n_nbrs,
                std::size_t block_size = 64, std::size_t n_threads = 0,
                std::size_t grain_size = 1, bool verbose = false)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  NNHeap<typename Distance::Output, typename Distance::Index> neighbor_heap(
      distance.ny, n_nbrs);
  Progress progress(1, verbose);

  BruteForceWorker<Distance> worker(neighbor_heap, distance);
  batch_parallel_for<Parallel>(worker, progress, neighbor_heap.n_points,
                               block_size, n_threads, grain_size);

  sort_heap(neighbor_heap, block_size, n_threads, grain_size);

  return heap_to_graph(neighbor_heap);
}

template <typename Distance, typename Progress, typename Parallel>
auto nnbf(Distance &distance, typename Distance::Index n_nbrs,
          std::size_t block_size = 64, std::size_t n_threads = 0,
          std::size_t grain_size = 1, bool verbose = false)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  return nnbf_query<Distance, Progress, Parallel>(
      distance, n_nbrs, block_size, n_threads, grain_size, verbose);
}

template <typename Distance, typename Progress>
auto nnbf_query(Distance &distance, typename Distance::Index n_nbrs,
                bool verbose)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  NNHeap<typename Distance::Output, typename Distance::Index> neighbor_heap(
      distance.ny, n_nbrs);
  Progress progress(distance.nx, verbose);

  nnbf_query_window(neighbor_heap, distance, progress, 0,
                    neighbor_heap.n_points);
  neighbor_heap.deheap_sort();

  return heap_to_graph(neighbor_heap);
}

template <typename Distance, typename Progress>
auto nnbf(Distance &distance, typename Distance::Index n_nbrs, bool verbose)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  // distance.nx == distance.ny but this pattern is consistent with the
  // query usage
  NNHeap<typename Distance::Output, typename Distance::Index> neighbor_heap(
      distance.ny, n_nbrs);
  Progress progress(distance.nx, verbose);

  std::size_t n_points = neighbor_heap.n_points;
  for (std::size_t i = 0; i < n_points; i++) {
    for (std::size_t j = i; j < n_points; j++) {
      typename Distance::Output d = distance(i, j);
      if (neighbor_heap.accepts(i, d)) {
        neighbor_heap.unchecked_push(i, d, j);
      }
      if (i != j && neighbor_heap.accepts(j, d)) {
        neighbor_heap.unchecked_push(j, d, i);
      }
    }
    TDOANN_ITERFINISHED();
  }

  neighbor_heap.deheap_sort();

  return heap_to_graph(neighbor_heap);
}

template <typename Distance, typename Progress, typename Parallel>
auto brute_force_build(const std::vector<typename Distance::Input> &data,
                       std::size_t ndim, typename Distance::Index n_nbrs,
                       std::size_t block_size = 64, std::size_t n_threads = 0,
                       std::size_t grain_size = 1, bool verbose = false)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  Distance distance(data, ndim);

  if (n_threads > 0) {
    return nnbf<Distance, Progress, Parallel>(distance, n_nbrs, block_size,
                                              n_threads, grain_size, verbose);
  } else {
    return nnbf<Distance, Progress>(distance, n_nbrs, verbose);
  }
}

template <typename Distance, typename Progress, typename Parallel>
auto brute_force_query(const std::vector<typename Distance::Input> &reference,
                       std::size_t ndim,
                       const std::vector<typename Distance::Input> &query,
                       typename Distance::Index n_nbrs,
                       std::size_t block_size = 64, std::size_t n_threads = 0,
                       std::size_t grain_size = 1, bool verbose = false)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  Distance distance(reference, query, ndim);

  if (n_threads > 0) {
    return nnbf_query<Distance, Progress, Parallel>(
        distance, n_nbrs, block_size, n_threads, grain_size, verbose);
  } else {
    return nnbf_query<Distance, Progress>(distance, n_nbrs, verbose);
  }
}

} // namespace tdoann
#endif // TDOANN_BRUTE_FORCE_H

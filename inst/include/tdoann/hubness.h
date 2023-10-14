// BSD 2-Clause License
//
// Copyright 2022 James Melville
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

#ifndef TDOANN_HUBNESS_H
#define TDOANN_HUBNESS_H

#include <limits>
#include <utility>
#include <vector>

#include "nngraph.h"
#include "parallel.h"
#include "progressbase.h"

namespace tdoann {

template <typename T> auto pair_dmax() -> std::pair<T, T> {
  return std::make_pair((std::numeric_limits<T>::max)(),
                        (std::numeric_limits<T>::max)());
}

template <typename Parallel, typename NbrHeap>
void local_scale(const std::vector<typename NbrHeap::Index> &idx_vec,
                 const std::vector<typename NbrHeap::DistanceOut> &dist_vec,
                 const std::vector<typename NbrHeap::DistanceOut> &sdist_vec,
                 NbrHeap &nn_heap, std::size_t n_threads,
                 ProgressBase &progress) {
  using Idx = typename NbrHeap::Index;
  using Out = typename NbrHeap::DistanceOut;

  // Pair up the scaled and unscaled distances
  using DPair = std::pair<Out, Out>;
  std::vector<DPair> dpairs;
  dpairs.reserve(dist_vec.size());

  auto sdist_worker = [&](std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; i++) {
      dpairs.emplace_back(sdist_vec[i], dist_vec[i]);
    }
  };
  batch_parallel_for<Parallel>(sdist_worker, dist_vec.size(), n_threads,
                               progress);

  // Create an unsorted top-k neighbor heap of size n_nbrs using the paired
  // distances as values
  using PairNbrHeap = tdoann::NNHeap<DPair, Idx, pair_dmax>;
  constexpr std::size_t batch_size = 100;
  auto n_points = nn_heap.n_points;
  auto n_nbrs = nn_heap.n_nbrs;
  bool transpose = false;
  PairNbrHeap pair_heap(n_points, n_nbrs);
  tdoann::vec_to_query_heap<Parallel>(pair_heap, idx_vec, n_points, dpairs,
                                      batch_size, n_threads, transpose,
                                      progress);

  auto heap_worker = [&](std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; i++) {
      for (decltype(n_nbrs) j = 0; j < n_nbrs; j++) {
        nn_heap.checked_push(i, pair_heap.distance(i, j).second,
                             pair_heap.index(i, j));
      }
    }
  };
  batch_parallel_for<Parallel>(heap_worker, n_points, n_threads, progress);
}

template <typename Dist, typename Idx>
void local_scaled_distances(std::size_t begin, std::size_t end,
                            const std::vector<Idx> &idx,
                            const std::vector<Dist> &dist, std::size_t n_nbrs,
                            const std::vector<Dist> &local_scales,
                            std::vector<Dist> &sdist) {
  for (auto i = begin; i < end; i++) {
    std::size_t innbrs = i * n_nbrs;
    auto scalei = local_scales[i];
    for (std::size_t j = 0; j < n_nbrs; j++) {
      auto idx_ij = innbrs + j;
      auto dist_ij = dist[idx_ij];
      sdist[idx_ij] =
          (dist_ij * dist_ij) / (scalei * local_scales[idx[idx_ij]]);
    }
  }
}

template <typename Parallel, typename Dist, typename Idx>
auto local_scaled_distances(const std::vector<Idx> &idx,
                            const std::vector<Dist> &dist, std::size_t n_nbrs,
                            const std::vector<Dist> &local_scales,
                            std::size_t n_threads, ProgressBase &progress)
    -> std::vector<Dist> {

  std::size_t n_points = local_scales.size();
  std::vector<Dist> sdist(dist.size());

  auto worker = [&](std::size_t begin, std::size_t end) {
    local_scaled_distances(begin, end, idx, dist, n_nbrs, local_scales, sdist);
  };

  batch_parallel_for<Parallel>(worker, n_points, n_threads, progress);

  return sdist;
}

// Welford-style
template <typename T>
auto mean_average(const std::vector<T> &vec, std::size_t begin, std::size_t end)
    -> double {
  long double mean = 0.0;
  auto onemb = 1 - begin;
  for (auto i = begin; i < end; ++i) {
    mean += (vec[i] - mean) / (i + onemb);
  }
  return static_cast<T>(mean);
}

template <typename T>
void get_local_scales(std::size_t begin, std::size_t end,
                      const std::vector<T> &dist_vec, std::size_t n_nbrs,
                      std::size_t k_begin, std::size_t k_end, T min_scale,
                      std::vector<T> &local_scales) {
  for (auto i = begin; i < end; i++) {
    auto innbrs = i * n_nbrs;
    local_scales[i] = std::max(
        min_scale, mean_average(dist_vec, innbrs + k_begin, innbrs + k_end));
  }
}

template <typename Parallel, typename T>
auto get_local_scales(const std::vector<T> &dist_vec, std::size_t n_nbrs,
                      std::size_t k_begin, std::size_t k_end, T min_scale,
                      std::size_t n_threads, ProgressBase &progress)
    -> std::vector<T> {
  std::size_t n_points = dist_vec.size() / n_nbrs;
  std::vector<T> local_scales(n_points);

  auto worker = [&](std::size_t begin, std::size_t end) {
    get_local_scales(begin, end, dist_vec, n_nbrs, k_begin, k_end, min_scale,
                     local_scales);
  };
  batch_parallel_for<Parallel>(worker, n_points, n_threads, progress);

  return local_scales;
}

} // namespace tdoann

#endif // TDOANN_HUBNESS_H

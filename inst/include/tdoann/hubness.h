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
#include "progressbase.h"

namespace tdoann {

template <typename T> std::pair<T, T> pair_dmax() {
  return std::make_pair((std::numeric_limits<T>::max)(),
                        (std::numeric_limits<T>::max)());
}

template <typename Progress = NullProgress, typename NbrHeap>
void local_scale(const std::vector<typename NbrHeap::Index> &idx_vec,
                 const std::vector<typename NbrHeap::DistanceOut> &dist_vec,
                 const std::vector<typename NbrHeap::DistanceOut> &sdist_vec,
                 NbrHeap &nn_heap) {
  using Idx = typename NbrHeap::Index;
  using Out = typename NbrHeap::DistanceOut;

  // Pair up the scaled and unscaled distances
  using DPair = std::pair<Out, Out>;
  std::vector<DPair> dpairs;
  dpairs.reserve(dist_vec.size());
  for (std::size_t i = 0; i < dist_vec.size(); i++) {
    dpairs.emplace_back(sdist_vec[i], dist_vec[i]);
  }

  // Create an unsorted top-k neighbor heap of size n_nbrs using the paired
  // distances as values
  using PairNbrHeap = tdoann::NNHeap<DPair, Idx, pair_dmax>;
  tdoann::HeapAddQuery heap_add;
  std::size_t block_size = 100;
  auto n_points = nn_heap.n_points;
  auto n_nbrs = nn_heap.n_nbrs;
  bool transpose = false;
  PairNbrHeap pair_heap(nn_heap.n_points, nn_heap.n_nbrs);
  tdoann::vec_to_heap<tdoann::HeapAddQuery, Progress>(
      pair_heap, idx_vec, n_points, dpairs, block_size, transpose);

  for (decltype(n_points) i = 0; i < n_points; i++) {
    for (decltype(n_nbrs) j = 0; j < n_nbrs; j++) {
      heap_add.push(nn_heap, i, pair_heap.index(i, j),
                    pair_heap.distance(i, j).second);
    }
  }
}

// Welford-style
template <typename T>
auto mean_average(const std::vector<T> &v, std::size_t begin, std::size_t end)
    -> double {
  long double mean = 0.0;
  auto b1 = 1 - begin;
  for (auto i = begin; i < end; ++i) {
    mean += (v[i] - mean) / (i + b1);
  }
  return static_cast<T>(mean);
}

template <typename T>
auto get_local_scales(const std::vector<T> &dist_vec, std::size_t n_nbrs,
                      std::size_t k_begin, std::size_t k_end)
    -> std::vector<T> {
  std::size_t n_points = dist_vec.size() / n_nbrs;
  std::vector<T> local_scales(n_points);
  for (std::size_t i = 0; i < n_points; i++) {
    std::size_t innbrs = i * n_nbrs;
    local_scales[i] = mean_average(dist_vec, innbrs + k_begin, innbrs + k_end);
  }

  return local_scales;
}

} // namespace tdoann

#endif // TDOANN_HUBNESS_H

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

#ifndef TDOANN_RANDNBRS_H
#define TDOANN_RANDNBRS_H

#include <vector>

#include "intsampler.h"
#include "nngraph.h"
#include "parallel.h"

namespace tdoann {

template <typename Distance, typename Parallel, typename HeapAdd>
// typename Sampler,
auto get_nn(Distance &distance, typename Distance::Index n_nbrs,
            BaseIntSampler<typename Distance::Index> &sampler, bool sort,
            std::size_t n_threads, ProgressBase &progress)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {

  using Out = typename Distance::Output;
  using Idx = typename Distance::Index;

  const std::size_t n_points = distance.ny;
  const std::size_t n_refs = distance.nx;

  // needs to happen outside of threads
  const uint64_t seed = sampler.initialize_seed();

  std::vector<Idx> nn_idx(n_points * n_nbrs);
  std::vector<Out> nn_dist(n_points * n_nbrs);

  auto worker = [&, seed](std::size_t begin, std::size_t end) {
    auto thread_sampler = sampler.clone(seed, end);

    for (auto qi = begin, kqi = n_nbrs * begin; qi < end; ++qi, kqi += n_nbrs) {
      const auto idxi = thread_sampler->sample(n_refs, n_nbrs);
      for (std::size_t j = 0, idx_offset = kqi; j < n_nbrs; ++j, ++idx_offset) {
        const auto &rand_nbri = idxi[j];
        nn_idx[idx_offset] = rand_nbri;
        // distance calcs are 0-indexed
        nn_dist[idx_offset] = distance(rand_nbri, qi);
      }
    }
  };

  progress.set_n_iters(1);
  const std::size_t block_size = 128;
  const std::size_t grain_size = 1;
  batch_parallel_for<Parallel>(worker, n_points, block_size, n_threads,
                               grain_size, progress);

  NNGraph<Out, Idx> nn_graph(nn_idx, nn_dist, n_points);
  if (sort) {
    sort_knn_graph<HeapAdd, Parallel>(nn_graph, block_size, n_threads,
                                      grain_size, progress);
  }
  return nn_graph;
}

template <typename Distance, typename Parallel>
auto random_build(Distance &distance, typename Distance::Index n_nbrs,
                  BaseIntSampler<typename Distance::Index> &sampler, bool sort,
                  std::size_t n_threads, ProgressBase &progress)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  if (n_threads == 0) {
    return get_nn<Distance, Parallel, tdoann::HeapAddSymmetric>(
        distance, n_nbrs, sampler, sort, n_threads, progress);
  }
  return get_nn<Distance, Parallel, tdoann::LockingHeapAddSymmetric>(
      distance, n_nbrs, sampler, sort, n_threads, progress);
}

template <typename Distance, typename Parallel>
auto random_query(Distance &distance, typename Distance::Index n_nbrs,
                  BaseIntSampler<typename Distance::Index> &sampler, bool sort,
                  std::size_t n_threads, ProgressBase &progress)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  return get_nn<Distance, Parallel, tdoann::HeapAddQuery>(
      distance, n_nbrs, sampler, sort, n_threads, progress);
}

} // namespace tdoann

#endif // TDOANN_RANDNBRS_H

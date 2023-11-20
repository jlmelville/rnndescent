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

#include "distancebase.h"
#include "heap.h"
#include "nngraph.h"
#include "parallel.h"
#include "random.h"

namespace tdoann {
template <typename NbrHeap>
auto fill_random(NbrHeap &current_graph,
                 const BaseDistance<typename NbrHeap::DistanceOut,
                                    typename NbrHeap::Index> &distance,
                 RandomIntGenerator<typename NbrHeap::Index> &rng,
                 typename NbrHeap::Index query,
                 typename NbrHeap::Index n_ref_points) {
  for (std::size_t j = 0; j < n_ref_points; j++) {
    if (current_graph.is_full(query)) {
      return;
    }
    typename NbrHeap::Index ref = rng.rand_int(n_ref_points);
    const auto dist_rq = distance.calculate(ref, query);
    current_graph.checked_push(query, dist_rq, ref);
  }
}

// fill any "missing" data with random neighbors
template <typename NbrHeap>
auto fill_random(
    NbrHeap &current_graph,
    const BaseDistance<typename NbrHeap::DistanceOut, typename NbrHeap::Index>
        &distance,
    ParallelRandomIntProvider<typename NbrHeap::Index> &rng_provider,
    std::size_t n_threads, ProgressBase &progress, const Executor &executor) {
  using Idx = typename NbrHeap::Index;

  Idx n_ref_points = distance.get_nx();
  rng_provider.initialize();
  auto worker = [&](std::size_t begin, std::size_t end) {
    auto rng_ptr = rng_provider.get_parallel_instance(end);
    for (auto query = begin; query < end; ++query) {
      fill_random(current_graph, distance, *rng_ptr, static_cast<Idx>(query),
                  n_ref_points);
    }
  };

  progress.set_n_iters(1);
  ExecutionParams exec_params{};
  dispatch_work(worker, current_graph.n_points, n_threads, exec_params,
                progress, executor);
}

template <typename Out, typename Idx>
auto sample_neighbors(const BaseDistance<Out, Idx> &distance, Idx n_nbrs,
                      RandomIntGenerator<Idx> &sampler,
                      std::vector<Idx> &nn_idx, std::vector<Out> &nn_dist,
                      std::size_t begin, std::size_t end) {
  const std::size_t n_refs = distance.get_nx();

  for (auto qi = begin, kqi = n_nbrs * begin; qi < end; ++qi, kqi += n_nbrs) {
    const auto idxi = sampler.sample(n_refs, n_nbrs);
    for (std::size_t j = 0, idx_offset = kqi; j < n_nbrs; ++j, ++idx_offset) {
      const auto &rand_nbri = idxi[j];
      nn_idx[idx_offset] = rand_nbri;
      // distance calcs are 0-indexed
      nn_dist[idx_offset] = distance.calculate(rand_nbri, qi);
    }
  }
}

template <typename Out, typename Idx>
auto sample_neighbors(const BaseDistance<Out, Idx> &distance, Idx n_nbrs,
                      ParallelRandomIntProvider<Idx> &sampler,
                      std::size_t n_threads, ProgressBase &progress,
                      const Executor &executor) -> NNGraph<Out, Idx> {
  const std::size_t n_points = distance.get_ny();

  std::vector<Idx> nn_idx(n_points * n_nbrs);
  std::vector<Out> nn_dist(n_points * n_nbrs);

  // needs to happen outside of any threads
  sampler.initialize();

  auto worker = [&](std::size_t begin, std::size_t end) {
    auto thread_sampler = sampler.get_parallel_instance(end);
    sample_neighbors(distance, n_nbrs, *thread_sampler, nn_idx, nn_dist, begin,
                     end);
  };

  progress.set_n_iters(1);
  ExecutionParams exec_params{128};
  dispatch_work(worker, n_points, n_threads, exec_params, progress, executor);
  return NNGraph<Out, Idx>(nn_idx, nn_dist, n_points);
}

template <typename Out, typename Idx>
auto sample_neighbors(const BaseDistance<Out, Idx> &distance, Idx n_nbrs,
                      RandomIntGenerator<Idx> &sampler, ProgressBase &progress)
    -> NNGraph<Out, Idx> {
  const std::size_t n_points = distance.get_ny();

  std::vector<Idx> nn_idx(n_points * n_nbrs);
  std::vector<Out> nn_dist(n_points * n_nbrs);

  auto worker = [&](std::size_t begin, std::size_t end) {
    sample_neighbors(distance, n_nbrs, sampler, nn_idx, nn_dist, begin, end);
  };

  progress.set_n_iters(1);
  ExecutionParams exec_params{128};
  dispatch_work(worker, n_points, exec_params, progress);
  return NNGraph<Out, Idx>(nn_idx, nn_dist, n_points);
}

template <typename Out, typename Idx>
auto random_build(const BaseDistance<Out, Idx> &distance, Idx n_nbrs,
                  ParallelRandomIntProvider<Idx> &sampler, bool sort,
                  std::size_t n_threads, ProgressBase &progress,
                  const Executor &executor) -> NNGraph<Out, Idx> {

  auto nn_graph = sample_neighbors(distance, n_nbrs, sampler, n_threads,
                                   progress, executor);
  if (sort) {
    sort_knn_graph(nn_graph, n_threads, progress, executor);
  }
  return nn_graph;
}

template <typename Out, typename Idx>
auto random_build(const BaseDistance<Out, Idx> &distance, Idx n_nbrs,
                  RandomIntGenerator<Idx> &sampler, bool sort,
                  ProgressBase &progress) -> NNGraph<Out, Idx> {

  auto nn_graph = sample_neighbors(distance, n_nbrs, sampler, progress);
  if (sort) {
    sort_knn_graph(nn_graph, progress);
  }
  return nn_graph;
}

template <typename Out, typename Idx>
auto random_query(const BaseDistance<Out, Idx> &distance, Idx n_nbrs,
                  ParallelRandomIntProvider<Idx> &sampler, bool sort,
                  std::size_t n_threads, ProgressBase &progress,
                  const Executor &executor) -> NNGraph<Out, Idx> {

  auto nn_graph = sample_neighbors(distance, n_nbrs, sampler, n_threads,
                                   progress, executor);
  if (sort) {
    sort_query_graph(nn_graph, n_threads, progress, executor);
  }
  return nn_graph;
}

template <typename Out, typename Idx>
auto random_query(const BaseDistance<Out, Idx> &distance, Idx n_nbrs,
                  RandomIntGenerator<Idx> &sampler, bool sort,
                  ProgressBase &progress) -> NNGraph<Out, Idx> {

  auto nn_graph = sample_neighbors(distance, n_nbrs, sampler, progress);
  if (sort) {
    sort_query_graph(nn_graph, progress);
  }
  return nn_graph;
}

} // namespace tdoann

#endif // TDOANN_RANDNBRS_H

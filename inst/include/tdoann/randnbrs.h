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

#include "nngraph.h"
#include "parallel.h"

namespace tdoann {

template <typename Distance, typename Sampler> struct RandomNbrQueryWorker {
  using Idx = typename Distance::Index;
  using Out = typename Distance::Output;
  Distance &distance;

  std::size_t n_points;
  Idx n_nbrs;
  std::vector<Idx> nn_idx;
  std::vector<Out> nn_dist;

  int nrefs;

  uint64_t seed;

  RandomNbrQueryWorker(Distance &distance, Idx n_nbrs)
      : distance(distance), n_points(distance.ny), n_nbrs(n_nbrs),
        nn_idx(n_points * n_nbrs), nn_dist(n_points * n_nbrs, 0.0),
        nrefs(distance.nx), seed(Sampler::get_seed()) {}

  void operator()(std::size_t begin, std::size_t end) {
    Sampler int_sampler(seed, end);

    for (std::size_t qi = begin; qi < end; qi++) {
      auto idxi = int_sampler.sample(nrefs, n_nbrs);
      std::size_t kqi = n_nbrs * qi;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        auto &ri = idxi[j];
        nn_idx[j + kqi] = ri;
        nn_dist[j + kqi] = distance(ri, qi); // distance calcs are 0-indexed
      }
    }
  }
};

template <typename Distance, typename Progress, typename Parallel,
          typename Worker, typename HeapAdd>
auto get_nn(Distance &distance, typename Distance::Index n_nbrs, bool sort,
            std::size_t n_threads = 0, bool verbose = false)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  std::size_t n_points = distance.ny;
  Progress progress(1, verbose);

  const std::size_t block_size = 128;
  const std::size_t grain_size = 1;
  Worker worker(distance, n_nbrs);
  if (n_threads > 0) {
    batch_parallel_for<Parallel>(worker, progress, n_points, block_size,
                                 n_threads, grain_size);
  } else {
    batch_serial_for(worker, progress, n_points, block_size);
  }

  NNGraph<typename Distance::Output, typename Distance::Index> nn_graph(
      worker.nn_idx, worker.nn_dist, n_points);

  if (sort) {
    if (n_threads > 0) {
      sort_knn_graph<HeapAdd, NullProgress, Parallel>(nn_graph, block_size,
                                                      n_threads, grain_size);
    } else {
      sort_knn_graph<HeapAdd, NullProgress>(nn_graph);
    }
  }

  return nn_graph;
}

template <typename Distance, typename Sampler, typename Progress,
          typename Parallel>
auto random_build(const std::vector<typename Distance::Input> &data,
                  std::size_t ndim, typename Distance::Index n_nbrs, bool sort,
                  std::size_t n_threads = 0, bool verbose = false)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  Distance distance(data, ndim);

  using Worker = tdoann::RandomNbrQueryWorker<Distance, Sampler>;
  if (n_threads > 0) {
    using HeapAdd = tdoann::LockingHeapAddSymmetric;
    return get_nn<Distance, Progress, Parallel, Worker, HeapAdd>(
        distance, n_nbrs, sort, n_threads, verbose);

  } else {
    using HeapAdd = tdoann::HeapAddSymmetric;
    return get_nn<Distance, Progress, Parallel, Worker, HeapAdd>(
        distance, n_nbrs, sort, n_threads, verbose);
  }
}

template <typename Distance, typename Sampler, typename Progress,
          typename Parallel>
auto random_query(const std::vector<typename Distance::Input> &reference,
                  std::size_t ndim,
                  const std::vector<typename Distance::Input> &query,
                  typename Distance::Index n_nbrs, bool sort,
                  std::size_t n_threads = 0, bool verbose = false)
    -> NNGraph<typename Distance::Output, typename Distance::Index> {
  Distance distance(reference, query, ndim);

  using Worker = tdoann::RandomNbrQueryWorker<Distance, Sampler>;
  using HeapAdd = tdoann::HeapAddQuery;
  return get_nn<Distance, Progress, Parallel, Worker, HeapAdd>(
      distance, n_nbrs, sort, n_threads, verbose);
}

} // namespace tdoann

#endif // TDOANN_RANDNBRS_H

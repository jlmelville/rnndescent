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

#include "nngraph.h"
#include "parallel.h"

namespace tdoann {

template <typename Distance, typename Sampler>
struct RandomNbrQueryWorker : public BatchParallelWorker {

  Distance &distance;

  std::size_t n_points;
  std::size_t k;
  std::vector<int> nn_idx;
  std::vector<double> nn_dist;

  int nrefs;

  uint64_t seed;

  RandomNbrQueryWorker(Distance &distance, std::size_t k, uint64_t seed)
      : distance(distance), n_points(distance.ny), k(k), nn_idx(n_points * k),
        nn_dist(n_points * k, 0.0), nrefs(distance.nx), seed(seed) {}

  void operator()(std::size_t begin, std::size_t end) {
    Sampler int_sampler(seed, end);

    for (int qi = static_cast<int>(begin); qi < static_cast<int>(end); qi++) {
      auto idxi = int_sampler.template sample<uint32_t>(nrefs, k);
      std::size_t kqi = k * qi;
      for (std::size_t j = 0; j < k; j++) {
        auto &ri = idxi[j];
        nn_idx[j + kqi] = ri;
        nn_dist[j + kqi] = distance(ri, qi); // distance calcs are 0-indexed
      }
    }
  }
};

template <typename Distance, typename Sampler>
struct RandomNbrBuildWorker : public BatchParallelWorker {

  Distance &distance;

  std::size_t n_points;
  std::size_t k;
  std::vector<int> nn_idx;
  std::vector<double> nn_dist;

  int n_points_minus_1;
  int k_minus_1;
  uint64_t seed;

  RandomNbrBuildWorker(Distance &distance, std::size_t k, uint64_t seed)
      : distance(distance), n_points(distance.ny), k(k), nn_idx(n_points * k),
        nn_dist(n_points * k), n_points_minus_1(n_points - 1), k_minus_1(k - 1),
        seed(seed) {}

  void operator()(std::size_t begin, std::size_t end) {
    Sampler int_sampler(seed, end);

    for (auto qi = static_cast<int>(begin); qi < static_cast<int>(end); qi++) {
      std::size_t kqi = k * qi;
      std::size_t kqi1 = kqi + 1;
      nn_idx[0 + kqi] = qi;
      auto ris =
          int_sampler.template sample<uint32_t>(n_points_minus_1, k_minus_1);

      for (auto j = 0; j < k_minus_1; j++) {
        int ri = ris[j];
        if (ri >= qi) {
          ri += 1;
        }
        nn_idx[j + kqi1] = ri;
        nn_dist[j + kqi1] = distance(ri, qi); // distance calcs are 0-indexed
      }
    }
  }
};

template <typename Distance, template <typename, typename> class Worker,
          typename HeapAdd, typename Sampler, typename Progress>
struct SerialRandomNbrsImpl {
  static NNGraph get_nn(Distance &distance, std::size_t k, uint64_t seed,
                        bool sort, std::size_t block_size, bool verbose,
                        std::size_t, std::size_t) {
    std::size_t n_points = distance.ny;

    Progress progress(1, verbose);

    Worker<Distance, Sampler> worker(distance, k, seed);
    batch_serial_for(worker, progress, n_points, block_size);

    NNGraph nn_graph(worker.nn_idx, worker.nn_dist, n_points);

    if (sort) {
      sort_knn_graph<HeapAdd, NullProgress>(nn_graph);
    }

    return nn_graph;
  }
};

template <typename Distance, template <typename, typename> class Worker,
          typename HeapAdd, typename Sampler, typename Progress,
          typename Parallel>
struct ParallelRandomNbrsImpl {
  static NNGraph get_nn(Distance &distance, std::size_t k, uint64_t seed,
                        bool sort, std::size_t block_size = 4096,
                        bool verbose = false, std::size_t n_threads = 0,
                        std::size_t grain_size = 1) {
    std::size_t n_points = distance.ny;
    Progress progress(1, verbose);

    Worker<Distance, Sampler> worker(distance, k, seed);
    batch_parallel_for<Progress, decltype(worker), Parallel>(
        worker, progress, n_points, n_threads, block_size, grain_size);

    NNGraph nn_graph(worker.nn_idx, worker.nn_dist, n_points);

    if (sort) {
      sort_knn_graph_parallel<HeapAdd, NullProgress, SimpleNeighborHeap,
                              Parallel>(nn_graph, n_threads, block_size,
                                        grain_size);
    }

    return nn_graph;
  }
};

} // namespace tdoann

#endif // TDOANN_RANDNBRS_H

//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2019 James Melville
//
//  This file is part of rnndescent
//
//  rnndescent is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  rnndescent is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with rnndescent.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RNN_RANDNBRSPARALLEL_H
#define RNN_RANDNBRSPARALLEL_H

#include "rnn_parallel.h"
#include "rnn_randnbrs.h"
#include <Rcpp.h>
// [[Rcpp::depends(dqrng)]]
#include <dqrng.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>

struct LockingIndexSampler {
  tthread::mutex mutex;

  Rcpp::IntegerVector sample(int max_val, int num_to_sample) {
    tthread::lock_guard<tthread::mutex> guard(mutex);
    return Rcpp::IntegerVector(dqrng::dqsample_int(max_val, num_to_sample));
  }
};

template <typename Distance>
using ParallelRandomNbrQueryWorker =
    RandomNbrQueryWorker<Distance, LockingIndexSampler,
                         RcppParallel::RMatrix<int>,
                         RcppParallel::RMatrix<double>, BatchParallelWorker>;

template <typename Distance>
using ParallelRandomNbrBuildWorker =
    RandomNbrBuildWorker<Distance, LockingIndexSampler,
                         RcppParallel::RMatrix<int>,
                         RcppParallel::RMatrix<double>, BatchParallelWorker>;

struct ParallelRandomKnnBuild {
  // Can't use symmetric heap addition with parallel approach
  using HeapAdd = HeapAddQuery;
  template <typename D> using Worker = ParallelRandomNbrBuildWorker<D>;
};

struct ParallelRandomKnnQuery {
  using HeapAdd = HeapAddQuery;
  template <typename D> using Worker = ParallelRandomNbrQueryWorker<D>;
};

template <template <typename> class Worker, typename Progress,
          typename Distance>
void rknn_parallel(Progress &progress, Distance &distance,
                   Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix dist,
                   const std::size_t block_size = 4096,
                   const std::size_t grain_size = 1) {
  Worker<Distance> worker(distance, indices, dist);
  batch_parallel_for(worker, progress, indices.ncol(), block_size, grain_size);
}

template <typename ParallelRandomKnn> struct ParallelRandomNbrsImpl {
  std::size_t block_size;
  std::size_t grain_size;

  ParallelRandomNbrsImpl(std::size_t block_size = 4096,
                         std::size_t grain_size = 1)
      : block_size(block_size), grain_size(grain_size) {}

  template <typename Distance>
  void build_knn(Distance &distance, Rcpp::IntegerMatrix indices,
                 Rcpp::NumericMatrix dist, bool verbose) {
    const auto nr = indices.ncol();
    const auto n_blocks = (nr / block_size) + 1;
    RPProgress progress(1, n_blocks, verbose);
    rknn_parallel<Worker>(progress, distance, indices, dist, block_size,
                          grain_size);
  }
  void sort_knn(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix dist) {
    sort_knn_graph_parallel<HeapAdd>(indices, dist, block_size, grain_size);
  }

  template <typename D>
  using Worker = typename ParallelRandomKnn::template Worker<D>;
  using HeapAdd = typename ParallelRandomKnn::HeapAdd;
};

#endif // RNN_RANDNBRSPARALLEL_H

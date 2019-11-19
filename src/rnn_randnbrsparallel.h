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
#include "rnn_rng.h"
#include <Rcpp.h>
// [[Rcpp::depends(dqrng)]]
#include <dqrng.h>

struct LockingIndexSampler {
  tthread::mutex mutex;

  Rcpp::IntegerVector sample(int max_val, int num_to_sample) {
    tthread::lock_guard<tthread::mutex> guard(mutex);
    return Rcpp::IntegerVector(dqrng::dqsample_int(max_val, num_to_sample));
  }
};

struct Empty {};

template <typename Distance, typename IndexSampler, typename IdxMatrix,
          typename DistMatrix, typename Base>
struct RandomNbrQueryWorker : public Base {

  Distance &distance;

  IdxMatrix indices;
  DistMatrix dist;

  const int nrefs;
  const int k;
  IndexSampler index_sampler;

  RandomNbrQueryWorker(Distance &distance, Rcpp::IntegerMatrix output_indices,
                       Rcpp::NumericMatrix output_dist)
      : distance(distance), indices(output_indices), dist(output_dist),
        nrefs(distance.nx), k(output_indices.nrow()), index_sampler() {}

  void operator()(std::size_t begin, std::size_t end) {
    for (int qi = static_cast<int>(begin); qi < static_cast<int>(end); qi++) {
      auto idxi = dqrng::dqsample_int(nrefs, k); // 0-indexed
      for (auto j = 0; j < k; j++) {
        auto &ri = idxi[j];
        indices(j, qi) = ri + 1;        // store val as 1-index
        dist(j, qi) = distance(ri, qi); // distance calcs are 0-indexed
      }
    }
  }
};

template <typename Distance, typename IndexSampler, typename IdxMatrix,
          typename DistMatrix, typename Base>
struct RandomNbrBuildWorker : public Base {

  Distance &distance;

  IdxMatrix indices;
  DistMatrix dist;

  const int nr1;
  const int n_to_sample;
  IndexSampler index_sampler;

  RandomNbrBuildWorker(Distance &distance, Rcpp::IntegerMatrix output_indices,
                       Rcpp::NumericMatrix output_dist)
      : distance(distance), indices(output_indices), dist(output_dist),
        nr1(indices.ncol() - 1), n_to_sample(indices.nrow() - 1),
        index_sampler() {}

  void operator()(std::size_t begin, std::size_t end) {
    for (int i = static_cast<int>(begin); i < static_cast<int>(end); i++) {
      indices(0, i) = i + 1;
      auto idxi = index_sampler.sample(nr1, n_to_sample);
      for (auto j = 0; j < n_to_sample; j++) {
        auto val = idxi[j];
        val = val >= i ? val + 1 : val;    // ensure i isn't in the sample
        indices(j + 1, i) = val + 1;       // store val as 1-index
        dist(j + 1, i) = distance(i, val); // distance calcs are 0-indexed
      }
    }
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

template <template <typename> class Worker, typename Progress,
          typename Distance>
void rknn_parallel(Progress &progress, Distance &distance,
                   Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix dist,
                   const std::size_t block_size = 4096,
                   const std::size_t grain_size = 1) {
  Worker<Distance> worker(distance, indices, dist);
  batch_parallel_for(worker, progress, indices.ncol(), block_size, grain_size);
}

#endif // RNN_RANDNBRSPARALLEL_H

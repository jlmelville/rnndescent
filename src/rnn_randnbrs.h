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

#ifndef RNN_RANDNBRS_H
#define RNN_RANDNBRS_H

// [[Rcpp::depends(dqrng)]]
#include <Rcpp.h>
#include <dqrng.h>

#include "rnn.h"

struct Empty {};

template <typename Distance, typename IndexSampler, typename IdxMatrix,
          typename DistMatrix, typename Base>
struct RandomNbrQueryWorker : public Base {

  Distance &distance;

  IdxMatrix nn_idx;
  DistMatrix nn_dist;

  const int nrefs;
  const int k;
  IndexSampler index_sampler;

  RandomNbrQueryWorker(Distance &distance, Rcpp::IntegerMatrix nn_idx,
                       Rcpp::NumericMatrix nn_dist)
      : distance(distance), nn_idx(nn_idx), nn_dist(nn_dist),
        nrefs(distance.nx), k(nn_idx.nrow()), index_sampler() {}

  void operator()(std::size_t begin, std::size_t end) {
    for (int qi = static_cast<int>(begin); qi < static_cast<int>(end); qi++) {
      auto idxi = index_sampler.sample(nrefs, k);
      for (auto j = 0; j < k; j++) {
        auto &ri = idxi[j];
        nn_idx(j, qi) = ri + 1;            // store val as 1-index
        nn_dist(j, qi) = distance(ri, qi); // distance calcs are 0-indexed
      }
    }
  }
};

template <typename Distance, typename IndexSampler, typename IdxMatrix,
          typename DistMatrix, typename Base>
struct RandomNbrBuildWorker : public Base {

  Distance &distance;

  IdxMatrix nn_idx;
  DistMatrix nn_dist;

  const int nr1;
  const int n_to_sample;
  IndexSampler index_sampler;

  RandomNbrBuildWorker(Distance &distance, Rcpp::IntegerMatrix nn_idx,
                       Rcpp::NumericMatrix nn_dist)
      : distance(distance), nn_idx(nn_idx), nn_dist(nn_dist),
        nr1(nn_idx.ncol() - 1), n_to_sample(nn_idx.nrow() - 1),
        index_sampler() {}

  void operator()(std::size_t begin, std::size_t end) {
    for (int i = static_cast<int>(begin); i < static_cast<int>(end); i++) {
      nn_idx(0, i) = i + 1;
      auto idxi = index_sampler.sample(nr1, n_to_sample);
      for (auto j = 0; j < n_to_sample; j++) {
        auto val = idxi[j];
        val = val >= i ? val + 1 : val;       // ensure i isn't in the sample
        nn_idx(j + 1, i) = val + 1;           // store val as 1-index
        nn_dist(j + 1, i) = distance(i, val); // distance calcs are 0-indexed
      }
    }
  }
};

struct NonLockingIndexSampler {
  Rcpp::IntegerVector sample(int max_val, int num_to_sample) {
    return Rcpp::IntegerVector(dqrng::dqsample_int(max_val, num_to_sample));
  }
};

struct SerialRandomKnnQuery {
  using HeapAdd = HeapAddQuery;
  template <typename Distance>
  using SerialRandomNbrQueryWorker =
      RandomNbrQueryWorker<Distance, NonLockingIndexSampler,
                           Rcpp::IntegerMatrix, Rcpp::NumericMatrix, Empty>;
  template <typename D> using Worker = SerialRandomNbrQueryWorker<D>;
};

struct SerialRandomKnnBuild {
  using HeapAdd = HeapAddSymmetric;
  template <typename Distance>
  using SerialRandomNbrBuildWorker =
      RandomNbrBuildWorker<Distance, NonLockingIndexSampler,
                           Rcpp::IntegerMatrix, Rcpp::NumericMatrix, Empty>;
  template <typename D> using Worker = SerialRandomNbrBuildWorker<D>;
};

template <template <typename> class RandomKnnWorker, typename Progress,
          typename Distance>
void rknn_serial(Progress &progress, Distance &distance,
                 Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
                 const std::size_t block_size = 4096) {
  RandomKnnWorker<Distance> worker(distance, nn_idx, nn_dist);
  batch_serial_for(worker, progress, nn_idx.ncol(), block_size);
}

template <typename SerialRandomKnn> struct SerialRandomNbrsImpl {
  std::size_t block_size;
  SerialRandomNbrsImpl(std::size_t block_size) : block_size(block_size) {}

  template <typename Distance>
  void build_knn(Distance &distance, Rcpp::IntegerMatrix nn_idx,
                 Rcpp::NumericMatrix nn_dist, bool verbose) {
    const auto nr = nn_idx.ncol();
    const auto n_blocks = (nr / block_size) + 1;
    RPProgress progress(1, n_blocks, verbose);
    rknn_serial<Worker>(progress, distance, nn_idx, nn_dist, block_size);
  }
  void sort_knn(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist) {
    sort_knn_graph<HeapAdd>(nn_idx, nn_dist);
  }

  template <typename D>
  using Worker = typename SerialRandomKnn::template Worker<D>;
  using HeapAdd = typename SerialRandomKnn::HeapAdd;
};

#endif // RNN_RANDNBRS_H

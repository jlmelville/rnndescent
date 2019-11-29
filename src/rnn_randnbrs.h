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

#include <Rcpp.h>

// [[Rcpp::depends(dqrng)]]
#include "convert_seed.h"
#include "dqrng_generator.h"
#include <dqrng.h>

#include "rnn.h"
#include "rnn_progress.h"
#include "rnn_rng.h"
#include "rnn_sample.h"

template <typename Distance, typename IdxMatrix, typename DistMatrix,
          typename Base>
struct RandomNbrQueryWorker : public Base {

  Distance &distance;

  IdxMatrix nn_idx;
  DistMatrix nn_dist;

  const int nrefs;
  const int k;

  uint64_t seed;

  RandomNbrQueryWorker(Distance &distance, Rcpp::IntegerMatrix nn_idx,
                       Rcpp::NumericMatrix nn_dist)
      : distance(distance), nn_idx(nn_idx), nn_dist(nn_dist),
        nrefs(distance.nx), k(nn_idx.nrow()), seed(pseed()) {}

  void operator()(std::size_t begin, std::size_t end) {
    dqrng::rng64_t rng = std::make_shared<dqrng::random_64bit_wrapper<pcg64>>();
    rng->seed(seed, end);
    for (int qi = static_cast<int>(begin); qi < static_cast<int>(end); qi++) {
      auto idxi = sample<uint32_t>(rng, nrefs, k);
      for (auto j = 0; j < k; j++) {
        auto &ri = idxi[j];
        nn_idx(j, qi) = ri + 1;            // store val as 1-index
        nn_dist(j, qi) = distance(ri, qi); // distance calcs are 0-indexed
      }
    }
  }
};

template <typename Distance, typename IdxMatrix, typename DistMatrix,
          typename Base>
struct RandomNbrBuildWorker : public Base {

  Distance &distance;

  IdxMatrix nn_idx;
  DistMatrix nn_dist;

  const int nr1;
  const int k_minus_1;
  uint64_t seed;

  RandomNbrBuildWorker(Distance &distance, Rcpp::IntegerMatrix nn_idx,
                       Rcpp::NumericMatrix nn_dist)
      : distance(distance), nn_idx(nn_idx), nn_dist(nn_dist),
        nr1(nn_idx.ncol() - 1), k_minus_1(nn_idx.nrow() - 1), seed(pseed()) {}

  void operator()(std::size_t begin, std::size_t end) {
    dqrng::rng64_t rng = std::make_shared<dqrng::random_64bit_wrapper<pcg64>>();
    rng->seed(seed, end);
    for (int qi = static_cast<int>(begin); qi < static_cast<int>(end); qi++) {
      nn_idx(0, qi) = qi + 1;
      auto ris = sample<uint32_t>(rng, nr1, k_minus_1);
      for (auto j = 0; j < k_minus_1; j++) {
        int ri = ris[j];
        if (ri >= qi) {
          ri += 1;
        }
        nn_idx(j + 1, qi) = ri + 1;            // store val as 1-index
        nn_dist(j + 1, qi) = distance(ri, qi); // distance calcs are 0-indexed
      }
    }
  }
};

struct SerialRandomKnnQuery {
  using HeapAdd = HeapAddQuery;
  template <typename Distance>
  using SerialRandomNbrQueryWorker =
      RandomNbrQueryWorker<Distance, Rcpp::IntegerMatrix, Rcpp::NumericMatrix,
                           Empty>;
  template <typename D> using Worker = SerialRandomNbrQueryWorker<D>;
};

struct SerialRandomKnnBuild {
  using HeapAdd = HeapAddSymmetric;
  template <typename Distance>
  using SerialRandomNbrBuildWorker =
      RandomNbrBuildWorker<Distance, Rcpp::IntegerMatrix, Rcpp::NumericMatrix,
                           Empty>;
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
    RPProgress progress(1, verbose);
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

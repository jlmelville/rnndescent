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

#include <Rcpp.h>
// [[Rcpp::depends(dqrng)]]
#include <dqrng.h>

#include "RcppPerpendicular.h"

#include "rnn_knnsort.h"
#include "rnn_parallel.h"
#include "rnn_randnbrs.h"
#include "rnn_rtoheap.h"

template <typename Distance>
using ParallelRandomNbrQueryWorker =
    RandomNbrQueryWorker<Distance, RcppPerpendicular::RMatrix<int>,
                         RcppPerpendicular::RMatrix<double>,
                         BatchParallelWorker>;

template <typename Distance>
using ParallelRandomNbrBuildWorker =
    RandomNbrBuildWorker<Distance, RcppPerpendicular::RMatrix<int>,
                         RcppPerpendicular::RMatrix<double>,
                         BatchParallelWorker>;

struct ParallelRandomKnnBuild {
  using HeapAdd = LockingHeapAddSymmetric;
  template <typename D> using Worker = ParallelRandomNbrBuildWorker<D>;
};

struct ParallelRandomKnnQuery {
  using HeapAdd = HeapAddQuery;
  template <typename D> using Worker = ParallelRandomNbrQueryWorker<D>;
};

template <typename ParallelRandomKnn> struct ParallelRandomNbrsImpl {
  std::size_t block_size;
  std::size_t grain_size;

  ParallelRandomNbrsImpl(std::size_t block_size = 4096,
                         std::size_t grain_size = 1)
      : block_size(block_size), grain_size(grain_size) {}

  template <typename Distance>
  void build_knn(Distance &distance, Rcpp::IntegerMatrix nn_idx,
                 Rcpp::NumericMatrix nn_dist, bool verbose) {
    RPProgress progress(1, verbose);

    Worker<Distance> worker(distance, nn_idx, nn_dist);
    batch_parallel_for(worker, progress, nn_idx.ncol(), block_size, grain_size);
  }
  void sort_knn(Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist) {
    sort_knn_graph_parallel<HeapAdd>(nn_idx, nn_dist, block_size, grain_size);
  }

  template <typename D>
  using Worker = typename ParallelRandomKnn::template Worker<D>;
  using HeapAdd = typename ParallelRandomKnn::HeapAdd;
};

#endif // RNN_RANDNBRSPARALLEL_H

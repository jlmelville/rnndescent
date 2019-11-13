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
// [[Rcpp::depends(RcppParallel)]]
#include "rnn_parallel.h"
#include "rnn_rng.h"
#include <RcppParallel.h>
// [[Rcpp::depends(dqrng)]]
#include <dqrng.h>

template <typename Distance>
struct RandomNbrWorker : public RcppParallel::Worker {

  const std::vector<typename Distance::in_type> data_vec;
  Distance distance;
  const int nr1;
  const int n_to_sample;

  RcppParallel::RMatrix<int> indices;
  RcppParallel::RMatrix<double> dist;

  tthread::mutex mutex;

  RandomNbrWorker(Rcpp::NumericMatrix data, int k,
                  Rcpp::IntegerMatrix output_indices,
                  Rcpp::NumericMatrix output_dist)
      : data_vec(Rcpp::as<std::vector<typename Distance::in_type>>(
            Rcpp::transpose(data))),
        distance(data_vec, data.ncol()), nr1(data.nrow() - 1),
        n_to_sample(k - 1), indices(output_indices), dist(output_dist) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (int i = static_cast<int>(begin); i < static_cast<int>(end); i++) {
      indices(0, i) = i + 1;
      std::unique_ptr<Rcpp::IntegerVector> idxi(nullptr);
      {
        tthread::lock_guard<tthread::mutex> guard(mutex);
        idxi.reset(
            new Rcpp::IntegerVector(dqrng::dqsample_int(nr1, n_to_sample)));
      }
      for (auto j = 0; j < n_to_sample; j++) {
        auto &val = (*idxi)[j];
        val = val >= i ? val + 1 : val;    // ensure i isn't in the sample
        indices(j + 1, i) = val + 1;       // store val as 1-index
        dist(j + 1, i) = distance(i, val); // distance calcs are 0-indexed
      }
    }
  }
};

template <typename Distance>
Rcpp::List random_knn_parallel(Rcpp::NumericMatrix data, int k,
                               const std::size_t block_size = 4096,
                               const std::size_t grain_size = 1,
                               bool verbose = false) {
  set_seed();

  const auto nr = data.nrow();
  Rcpp::IntegerMatrix indices(k, nr);
  Rcpp::NumericMatrix dist(k, nr);

  RandomNbrWorker<Distance> worker(data, k, indices, dist);

  RPProgress progress(nr, verbose);
  batch_parallel_for(worker, progress, nr, block_size, grain_size);
  return Rcpp::List::create(Rcpp::Named("idx") = Rcpp::transpose(indices),
                            Rcpp::Named("dist") = Rcpp::transpose(dist));
}

template <typename Distance>
struct RandomNbrQueryWorker : public RcppParallel::Worker {

  const std::vector<typename Distance::in_type> reference_vec;
  const std::vector<typename Distance::in_type> query_vec;

  Distance distance;
  const int nrefs;
  const int k;

  RcppParallel::RMatrix<int> indices;
  RcppParallel::RMatrix<double> dist;

  tthread::mutex mutex;

  RandomNbrQueryWorker(Rcpp::NumericMatrix reference, Rcpp::NumericMatrix query,
                       int k, Rcpp::IntegerMatrix output_indices,
                       Rcpp::NumericMatrix output_dist)
      : reference_vec(Rcpp::as<std::vector<typename Distance::in_type>>(
            Rcpp::transpose(reference))),
        query_vec(Rcpp::as<std::vector<typename Distance::in_type>>(
            Rcpp::transpose(query))),
        distance(reference_vec, query_vec, query.ncol()),
        nrefs(reference.nrow()), k(k), indices(output_indices),
        dist(output_dist) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (int query = static_cast<int>(begin); query < static_cast<int>(end); query++) {
      std::unique_ptr<Rcpp::IntegerVector> idxi(nullptr);
      {
        tthread::lock_guard<tthread::mutex> guard(mutex);
        idxi.reset(new Rcpp::IntegerVector(dqrng::dqsample_int(nrefs, k)));
      }
      for (auto j = 0; j < k; j++) {
        auto &ref = (*idxi)[j];
        indices(j, query) = ref + 1;       // store val as 1-index
        dist(j, query) = distance(ref, query); // distance calcs are 0-indexed
      }
    }
  }
};

template <typename Distance>
Rcpp::List random_knn_query_parallel(Rcpp::NumericMatrix reference,
                                     Rcpp::NumericMatrix query, int k,
                                     std::size_t block_size = 4096,
                                     std::size_t grain_size = 1,
                                     bool verbose = false) {
  set_seed();

  const auto nr = query.nrow();
  Rcpp::IntegerMatrix indices(k, nr);
  Rcpp::NumericMatrix dist(k, nr);

  RandomNbrQueryWorker<Distance> worker(reference, query, k, indices, dist);

  RPProgress progress(nr, verbose);
  batch_parallel_for(worker, progress, nr, block_size, grain_size);
  return Rcpp::List::create(Rcpp::Named("idx") = Rcpp::transpose(indices),
                            Rcpp::Named("dist") = Rcpp::transpose(dist));
}

#endif // RNN_RANDNBRSPARALLEL_H

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

#ifndef RNND_RRAND_INIT_H
#define RNND_RRAND_INIT_H

#include <Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
// [[Rcpp::depends(dqrng)]]
#include <dqrng.h>
#include "distance.h"
#include "rnn_parallel.h"

void set_seed() {
  dqrng::dqRNGkind("Xoroshiro128+");
  auto seed = Rcpp::IntegerVector::create(R::runif(0, 1) * std::numeric_limits<int>::max());
  dqrng::dqset_seed(seed);
}

template<typename Distance>
struct RandomNbrWorker : public RcppParallel::Worker {

  const std::vector<typename Distance::in_type> data_vec;
  Distance distance;
  const int nr1;
  const int n_to_sample;

  RcppParallel::RMatrix<int> indices;
  RcppParallel::RMatrix<double> dist;

  tthread::mutex mutex;

  RandomNbrWorker(
    Rcpp::NumericMatrix data,
    int k,
    Rcpp::IntegerMatrix output_indices,
    Rcpp::NumericMatrix output_dist
  ) :
    data_vec(Rcpp::as<std::vector<typename Distance::in_type>>(Rcpp::transpose(data))),
    distance(data_vec, data.ncol()),
    nr1(data.nrow() - 1),
    n_to_sample(k - 1),
    indices(output_indices),
    dist(output_dist)
  {}

  void operator()(std::size_t begin, std::size_t end) {
    for (int i = static_cast<int>(begin); i < static_cast<int>(end); i++) {
      indices(0, i) = i + 1;
      std::unique_ptr<Rcpp::IntegerVector> idxi(nullptr);
      {
        tthread::lock_guard<tthread::mutex> guard(mutex);
        idxi.reset(new Rcpp::IntegerVector(dqrng::dqsample_int(nr1, n_to_sample)));
      }
      for (auto j = 0; j < n_to_sample; j++) {
        auto& val = (*idxi)[j];
        val = val >= i ? val + 1 : val; // ensure i isn't in the sample
        indices(j + 1, i) = val + 1; // store val as 1-index
        dist(j + 1, i) = distance(i, val); // distance calcs are 0-indexed
      }
    }
  }
};

template<typename Distance,
         typename Progress>
Rcpp::List random_nbrs_parallel(
    Rcpp::NumericMatrix data,
    int k,
    std::size_t grain_size)
{
  set_seed();

  const auto nr = data.nrow();
  Rcpp::IntegerMatrix indices(k, nr);
  Rcpp::NumericMatrix dist(k, nr);

  RandomNbrWorker<Distance> worker(data, k, indices, dist);

  Progress progress;
  const constexpr std::size_t min_batch = 4096;
  batch_parallel_for(worker, progress, nr, min_batch, grain_size);
  return Rcpp::List::create(
    Rcpp::Named("idx") = Rcpp::transpose(indices),
    Rcpp::Named("dist") = Rcpp::transpose(dist)
  );
}

#endif // RNND_RRAND_INIT_H

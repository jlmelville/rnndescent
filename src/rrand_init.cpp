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
#include <Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include "dqrng.h"
#include "distance.h"

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
        // shift to 1-index and ensure i isn't in the sample
        indices(j + 1, i) = val >= i ? val + 2 : val + 1;
        dist(j + 1, i) = distance(i, val); // distance calcs are 0-indexed
      }
    }
  }
};


void set_seed() {
  dqrng::dqRNGkind("Xoroshiro128+");
  auto seed = Rcpp::IntegerVector::create(R::runif(0, 1) * std::numeric_limits<int>::max());
  dqrng::dqset_seed(seed);
}

template<typename Distance>
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
  RcppParallel::parallelFor(0, nr, worker, grain_size);

  return Rcpp::List::create(
    Rcpp::Named("idx") = Rcpp::transpose(indices),
    Rcpp::Named("dist") = Rcpp::transpose(dist)
  );
}

template<typename Distance>
Rcpp::List random_nbrs_impl(
    Rcpp::NumericMatrix data,
    int k)
{
  set_seed();

  const auto nr = data.nrow();
  const auto ndim = data.ncol();

  Rcpp::IntegerMatrix indices(k, nr);
  Rcpp::NumericMatrix dist(k, nr);

  auto data_vec = Rcpp::as<std::vector<typename Distance::in_type>>(Rcpp::transpose(data));
  Distance distance(data_vec, ndim);

  const auto nr1 = nr - 1;
  const auto n_to_sample = k - 1;
  for (auto i = 0; i < nr; i++) {
    indices(0, i) = i + 1;
    auto idxi = dqrng::dqsample_int(nr1, n_to_sample); // 0-indexed
    for (auto j = 0; j < n_to_sample; j++) {
      auto& val = idxi[j];
      // shift to 1-index and ensure i isn't in the sample
      indices(j + 1, i) = val >= i ? val + 2 : val + 1;
      dist(j + 1, i) = distance(i, val); // distance calcs are 0-indexed
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("idx") = Rcpp::transpose(indices),
    Rcpp::Named("dist") = Rcpp::transpose(dist)
  );
}

// [[Rcpp::export]]
Rcpp::List random_nbrs_cpp(
    Rcpp::NumericMatrix data,
    int k,
    const std::string& metric = "euclidean",
    bool parallelize = false,
    std::size_t grain_size = 1
)
{
  if (metric == "euclidean") {
    using dist_type = Euclidean<float, float>;
    if (parallelize) {
      return random_nbrs_parallel<dist_type>(data, k, grain_size);
    }
    else {
      return random_nbrs_impl<dist_type>(data, k);
    }
  }
  else if (metric == "l2") {
    using dist_type = L2<float, float>;
    if (parallelize) {
      return random_nbrs_parallel<dist_type>(data, k, grain_size);
    }
    else {
      return random_nbrs_impl<dist_type>(data, k);
    }
  }
  else if (metric == "cosine") {
    using dist_type = Cosine<float, float>;
    if (parallelize) {
      return random_nbrs_parallel<dist_type>(data, k, grain_size);
    }
    else {
      return random_nbrs_impl<dist_type>(data, k);
    }
  }
  else if (metric == "manhattan") {
    using dist_type = Manhattan<float, float>;
    if (parallelize) {
      return random_nbrs_parallel<dist_type>(data, k, grain_size);
    }
    else {
      return random_nbrs_impl<dist_type>(data, k);
    }
  }
  else if (metric == "hamming") {
    using dist_type = Hamming<uint8_t, std::size_t>;
    if (parallelize) {
      return random_nbrs_parallel<dist_type>(data, k, grain_size);
    }
    else {
      return random_nbrs_impl<dist_type>(data, k);
    }
  }
  else {
    Rcpp::stop("Bad metric");
  }
}

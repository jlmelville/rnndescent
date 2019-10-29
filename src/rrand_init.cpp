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
#include "distance.h"
#include "rnn.h"
#include "rrand_init_parallel.h"

#define RandomNbrs(DistType)                                                 \
if (parallelize) {                                                           \
  return random_nbrs_parallel<DistType, RProgress>(data, k, grain_size);     \
}                                                                            \
else {                                                                       \
  return random_nbrs_impl<DistType, RProgress>(data, k);                     \
}

template<typename Distance,
         typename Progress>
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
  Progress progress;

  const auto nr1 = nr - 1;
  const auto n_to_sample = k - 1;
  for (auto i = 0; i < nr; i++) {
    indices(0, i) = i + 1;
    auto idxi = dqrng::dqsample_int(nr1, n_to_sample); // 0-indexed
    for (auto j = 0; j < n_to_sample; j++) {
      auto& val = idxi[j];
      val = val >= i ? val + 1 : val; // ensure i isn't in the sample
      indices(j + 1, i) = val + 1; // store val as 1-index
      dist(j + 1, i) = distance(i, val); // distance calcs are 0-indexed
    }
    progress.check_interrupt();
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
    using DistType = Euclidean<float, float>;
    RandomNbrs(DistType)
  }
  else if (metric == "l2") {
    using DistType = L2<float, float>;
    RandomNbrs(DistType)
  }
  else if (metric == "cosine") {
    using DistType = Cosine<float, float>;
    RandomNbrs(DistType)
  }
  else if (metric == "manhattan") {
    using DistType = Manhattan<float, float>;
    RandomNbrs(DistType)
  }
  else if (metric == "hamming") {
    using DistType = Hamming<uint8_t, std::size_t>;
    RandomNbrs(DistType)
  }
  else {
    Rcpp::stop("Bad metric");
  }
}

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

#include "distance.h"
#include "rnn.h"
#include "rnn_rand_parallel.h"
#include <Rcpp.h>

#define RandomNbrs(Distance)                                                   \
  if (parallelize) {                                                           \
    return random_knn_parallel<Distance>(data, k, grain_size, verbose);        \
  } else {                                                                     \
    return random_knn_impl<Distance>(data, k, verbose);                        \
  }

template <typename Distance>
Rcpp::List random_knn_impl(Rcpp::NumericMatrix data, int k, bool verbose) {
  set_seed();

  const auto nr = data.nrow();
  const auto ndim = data.ncol();

  Rcpp::IntegerMatrix indices(k, nr);
  Rcpp::NumericMatrix dist(k, nr);

  auto data_vec =
      Rcpp::as<std::vector<typename Distance::in_type>>(Rcpp::transpose(data));
  Distance distance(data_vec, ndim);
  RPProgress progress(nr, verbose);

  const auto nr1 = nr - 1;
  const auto n_to_sample = k - 1;
  for (auto i = 0; i < nr; i++) {
    indices(0, i) = i + 1;
    auto idxi = dqrng::dqsample_int(nr1, n_to_sample); // 0-indexed
    for (auto j = 0; j < n_to_sample; j++) {
      auto &val = idxi[j];
      val = val >= i ? val + 1 : val;    // ensure i isn't in the sample
      indices(j + 1, i) = val + 1;       // store val as 1-index
      dist(j + 1, i) = distance(i, val); // distance calcs are 0-indexed
    }
    progress.increment();
    if (progress.check_interrupt()) {
      break;
    };
  }

  return Rcpp::List::create(Rcpp::Named("idx") = Rcpp::transpose(indices),
                            Rcpp::Named("dist") = Rcpp::transpose(dist));
}

// [[Rcpp::export]]
Rcpp::List random_knn_cpp(Rcpp::NumericMatrix data, int k,
                          const std::string &metric = "euclidean",
                          bool parallelize = false, std::size_t grain_size = 1,
                          bool verbose = false) {
  if (metric == "euclidean") {
    using Distance = Euclidean<float, float>;
    RandomNbrs(Distance)
  } else if (metric == "l2") {
    using Distance = L2<float, float>;
    RandomNbrs(Distance)
  } else if (metric == "cosine") {
    using Distance = Cosine<float, float>;
    RandomNbrs(Distance)
  } else if (metric == "manhattan") {
    using Distance = Manhattan<float, float>;
    RandomNbrs(Distance)
  } else if (metric == "hamming") {
    using Distance = Hamming<uint8_t, std::size_t>;
    RandomNbrs(Distance)
  } else {
    Rcpp::stop("Bad metric");
  }
}

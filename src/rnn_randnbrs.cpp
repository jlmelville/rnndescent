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

#include "tdoann/randnbrs.h"

#include "rnn_knnfactory.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_rng.h"
#include "rnn_rtoheap.h"
#include "rnn_sample.h"

using namespace Rcpp;

/* Macros */

#define RANDOM_NBRS_BUILD()                                                    \
  using KnnFactory = KnnBuildFactory<Distance>;                                \
  KnnFactory knn_factory(data);                                                \
  if (n_threads > 0) {                                                         \
    using RandomNbrsImpl = tdoann::ParallelRandomNbrsImpl<                     \
        tdoann::ParallelRandomKnnBuild<RPProgress, DQIntSampler, RParallel>>;  \
    RandomNbrsImpl impl(n_threads, block_size, grain_size);                    \
    RANDOM_NBRS_IMPL()                                                         \
  } else {                                                                     \
    using RandomNbrsImpl = tdoann::SerialRandomNbrsImpl<                       \
        tdoann::SerialRandomKnnBuild<RPProgress, DQIntSampler>>;               \
    RandomNbrsImpl impl(block_size);                                           \
    RANDOM_NBRS_IMPL()                                                         \
  }

#define RANDOM_NBRS_QUERY()                                                    \
  using KnnFactory = KnnQueryFactory<Distance>;                                \
  KnnFactory knn_factory(reference, query);                                    \
  if (n_threads > 0) {                                                         \
    using RandomNbrsImpl = tdoann::ParallelRandomNbrsImpl<                     \
        tdoann::ParallelRandomKnnQuery<RPProgress, DQIntSampler, RParallel>>;  \
    RandomNbrsImpl impl(n_threads, block_size, grain_size);                    \
    RANDOM_NBRS_IMPL()                                                         \
  } else {                                                                     \
    using RandomNbrsImpl = tdoann::SerialRandomNbrsImpl<                       \
        tdoann::SerialRandomKnnQuery<RPProgress, DQIntSampler>>;               \
    RandomNbrsImpl impl(block_size);                                           \
    RANDOM_NBRS_IMPL()                                                         \
  }

#define RANDOM_NBRS_IMPL()                                                     \
  return random_knn_impl<KnnFactory, RandomNbrsImpl, Distance>(                \
      k, order_by_distance, knn_factory, impl, verbose);

/* Functions */

template <typename KnnFactory, typename RandomNbrsImpl, typename Distance>
auto random_knn_impl(std::size_t k, bool order_by_distance,
                     KnnFactory &knn_factory, RandomNbrsImpl &impl,
                     bool verbose = false) -> List {
  uint64_t seed = pseed();

  auto distance = knn_factory.create_distance();
  std::size_t n_points = knn_factory.n_points;
  auto nn_graph = impl.build_knn(distance, n_points, k, seed, verbose);

  if (order_by_distance) {
    impl.sort_knn(nn_graph);
  }

  IntegerMatrix indices(k, n_points, nn_graph.idx.begin());
  NumericMatrix dist(k, n_points, nn_graph.dist.begin());

  return List::create(_("idx") = transpose(indices),
                      _("dist") = transpose(dist));
}

/* Exports */

// [[Rcpp::export]]
List random_knn_cpp(Rcpp::NumericMatrix data, int k,
                    const std::string &metric = "euclidean",
                    bool order_by_distance = true, std::size_t n_threads = 0,
                    std::size_t block_size = 4096, std::size_t grain_size = 1,
                    bool verbose = false){
    DISPATCH_ON_DISTANCES(RANDOM_NBRS_BUILD)}

// [[Rcpp::export]]
List random_knn_query_cpp(NumericMatrix reference, NumericMatrix query, int k,
                          const std::string &metric = "euclidean",
                          bool order_by_distance = true,
                          std::size_t n_threads = 0,
                          std::size_t block_size = 4096,
                          std::size_t grain_size = 1, bool verbose = false) {
  DISPATCH_ON_QUERY_DISTANCES(RANDOM_NBRS_QUERY)
}

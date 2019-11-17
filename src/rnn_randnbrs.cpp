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

#include "rnn.h"
#include "rnn_parallel.h"
#include "rnn_randnbrsparallel.h"
#include "rnn_rng.h"
#include "tdoann/distance.h"
#include "tdoann/progress.h"
#include <Rcpp.h>

using namespace tdoann;

/* Macros */

#define Distances(RandomKnn)                                                   \
  if (metric == "euclidean") {                                                 \
    using Distance = Euclidean<float, float>;                                  \
    RandomKnn(Distance)                                                        \
  } else if (metric == "l2sqr") {                                              \
    using Distance = L2Sqr<float, float>;                                      \
    RandomKnn(Distance)                                                        \
  } else if (metric == "cosine") {                                             \
    using Distance = Cosine<float, float>;                                     \
    RandomKnn(Distance)                                                        \
  } else if (metric == "manhattan") {                                          \
    using Distance = Manhattan<float, float>;                                  \
    RandomKnn(Distance)                                                        \
  } else if (metric == "hamming") {                                            \
    using Distance = Hamming<uint8_t, std::size_t>;                            \
    RandomKnn(Distance)                                                        \
  } else {                                                                     \
    Rcpp::stop("Bad metric");                                                  \
  }

#define RandomNbrs(Distance)                                                   \
  using KnnFactory = KnnBuildFactory<Distance>;                                \
  KnnFactory knn_factory(data);                                                \
  if (parallelize) {                                                           \
    using RandomNbrsImpl = ParallelRandomNbrsImpl<ParallelRandomKnnBuild>;     \
    RandomNbrsImpl impl(block_size, grain_size);                               \
    RandomNbrsKnn(KnnFactory, RandomNbrsImpl, Distance)                        \
  } else {                                                                     \
    using RandomNbrsImpl = SerialRandomNbrsImpl<SerialRandomKnnBuild>;         \
    RandomNbrsImpl impl;                                                       \
    RandomNbrsKnn(KnnFactory, RandomNbrsImpl, Distance)                        \
  }

#define RandomNbrsQuery(Distance)                                              \
  using KnnFactory = KnnQueryFactory<Distance>;                                \
  KnnFactory knn_factory(reference, query);                                    \
  if (parallelize) {                                                           \
    using RandomNbrsImpl = ParallelRandomNbrsImpl<ParallelRandomKnnQuery>;     \
    RandomNbrsImpl impl(block_size, grain_size);                               \
    RandomNbrsKnn(KnnFactory, RandomNbrsImpl, Distance)                        \
  } else {                                                                     \
    using RandomNbrsImpl = SerialRandomNbrsImpl<SerialRandomKnnQuery>;         \
    RandomNbrsImpl impl;                                                       \
    RandomNbrsKnn(KnnFactory, RandomNbrsImpl, Distance)                        \
  }

#define RandomNbrsKnn(KnnFactory, RandomNbrsImpl, Distance)                    \
  return random_knn_impl<KnnFactory, RandomNbrsImpl, Distance>(                \
      k, order_by_distance, knn_factory, impl, verbose);

/* Structs */

template <typename Distance> struct KnnQueryFactory {
  using DataVec = std::vector<typename Distance::in_type>;

  DataVec reference_vec;
  DataVec query_vec;
  int nrow;
  int ndim;

  KnnQueryFactory(Rcpp::NumericMatrix reference, Rcpp::NumericMatrix query)
      : reference_vec(Rcpp::as<DataVec>(Rcpp::transpose(reference))),
        query_vec(Rcpp::as<DataVec>(Rcpp::transpose(query))),
        nrow(query.nrow()), ndim(query.ncol()) {}

  Distance create_distance() const {
    return Distance(reference_vec, query_vec, ndim);
  }

  Rcpp::NumericMatrix create_distance_matrix(int k) const {
    return Rcpp::NumericMatrix(k, nrow);
  }

  Rcpp::IntegerMatrix create_index_matrix(int k) const {
    return Rcpp::IntegerMatrix(k, nrow);
  }
};

template <typename Distance> struct KnnBuildFactory {
  using DataVec = std::vector<typename Distance::in_type>;

  DataVec data_vec;
  int nrow;
  int ndim;

  KnnBuildFactory(Rcpp::NumericMatrix data)
      : data_vec(Rcpp::as<DataVec>(Rcpp::transpose(data))), nrow(data.nrow()),
        ndim(data.ncol()) {}

  Distance create_distance() const { return Distance(data_vec, ndim); }

  Rcpp::NumericMatrix create_distance_matrix(int k) const {
    return Rcpp::NumericMatrix(k, nrow);
  }

  Rcpp::IntegerMatrix create_index_matrix(int k) const {
    return Rcpp::IntegerMatrix(k, nrow);
  }
};

struct SerialRandomKnnQuery {
  template <typename Progress, typename Distance>
  static void build_knn(Progress &progress, Distance &distance,
                        Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix dist) {
    const auto nr = indices.ncol();
    const auto k = indices.nrow();
    const auto nrefs = distance.nx;

    for (auto i = 0; i < nr; i++) {
      auto idxi = dqrng::dqsample_int(nrefs, k); // 0-indexed
      for (auto j = 0; j < k; j++) {
        auto &ref_idx = idxi[j];
        indices(j, i) = ref_idx + 1;       // store val as 1-index
        dist(j, i) = distance(ref_idx, i); // distance calcs are 0-indexed
      }
      progress.iter_finished();
      TDOANN_BREAKIFINTERRUPTED();
    }
  }
  using HeapAdd = HeapAddQuery;
};

struct SerialRandomKnnBuild {
  template <typename Progress, typename Distance>
  static void build_knn(Progress &progress, Distance &distance,
                        Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix dist) {
    const auto nr = indices.ncol();
    const auto k = indices.nrow();
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
      progress.iter_finished();
      TDOANN_BREAKIFINTERRUPTED();
    }
  }
  using HeapAdd = HeapAddSymmetric;
};

template <typename SerialRandomKnn> struct SerialRandomNbrsImpl {

  template <typename Distance>
  void build_knn(Distance &distance, Rcpp::IntegerMatrix indices,
                 Rcpp::NumericMatrix dist, bool verbose) {
    const auto nr = indices.ncol();
    RPProgress progress(nr, verbose);
    SerialRandomKnn::build_knn(progress, distance, indices, dist);
  }
  void sort_knn(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix dist) {
    sort_knn_graph<HeapAdd>(indices, dist);
  }

  using HeapAdd = typename SerialRandomKnn::HeapAdd;
};

struct ParallelRandomKnnBuild {
  template <typename Progress, typename Distance>
  static void build_knn(Progress &progress, Distance &distance,
                        Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix dist,
                        const std::size_t block_size = 4096,
                        const std::size_t grain_size = 1) {
    rknn_parallel<RandomNbrWorker>(progress, distance, indices, dist,
                                   block_size, grain_size);
  }
  // Can't use symmetric heap addition with parallel approach
  using HeapAdd = HeapAddQuery;
};

struct ParallelRandomKnnQuery {
  template <typename Progress, typename Distance>
  static void build_knn(Progress &progress, Distance &distance,
                        Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix dist,
                        const std::size_t block_size = 4096,
                        const std::size_t grain_size = 1) {
    rknn_parallel<RandomNbrQueryWorker>(progress, distance, indices, dist,
                                        block_size, grain_size);
  }
  using HeapAdd = HeapAddQuery;
};

template <typename ParallelRandomKnn> struct ParallelRandomNbrsImpl {
  std::size_t block_size;
  std::size_t grain_size;

  ParallelRandomNbrsImpl(std::size_t block_size = 4096,
                         std::size_t grain_size = 1)
      : block_size(block_size), grain_size(grain_size) {}

  template <typename Distance>
  void build_knn(Distance &distance, Rcpp::IntegerMatrix indices,
                 Rcpp::NumericMatrix dist, bool verbose) {
    const auto nr = indices.ncol();
    const auto n_blocks = (nr / block_size) + 1;
    RPProgress progress(1, n_blocks, verbose);
    ParallelRandomKnn::build_knn(progress, distance, indices, dist, block_size,
                                 grain_size);
  }
  void sort_knn(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix dist) {
    sort_knn_graph_parallel<HeapAdd>(indices, dist, block_size, grain_size);
  }

  using HeapAdd = typename ParallelRandomKnn::HeapAdd;
};

/* Functions */

template <typename KnnFactory, typename RandomNbrsImpl, typename Distance>
Rcpp::List random_knn_impl(int k, bool order_by_distance,
                           KnnFactory &knn_factory, RandomNbrsImpl &impl,
                           bool verbose = false) {
  set_seed();

  auto distance = knn_factory.create_distance();
  auto indices = knn_factory.create_index_matrix(k);
  auto dist = knn_factory.create_distance_matrix(k);

  impl.build_knn(distance, indices, dist, verbose);

  indices = Rcpp::transpose(indices);
  dist = Rcpp::transpose(dist);

  if (order_by_distance) {
    impl.sort_knn(indices, dist);
  }

  return Rcpp::List::create(Rcpp::Named("idx") = indices,
                            Rcpp::Named("dist") = dist);
}

/* Exports */

// [[Rcpp::export]]
Rcpp::List
random_knn_cpp(Rcpp::NumericMatrix data, int k,
               const std::string &metric = "euclidean",
               bool order_by_distance = true, bool parallelize = false,
               std::size_t block_size = 4096, std::size_t grain_size = 1,
               bool verbose = false){Distances(RandomNbrs)}

// [[Rcpp::export]]
Rcpp::List
    random_knn_query_cpp(Rcpp::NumericMatrix reference,
                         Rcpp::NumericMatrix query, int k,
                         const std::string &metric = "euclidean",
                         bool order_by_distance = true,
                         bool parallelize = false,
                         std::size_t block_size = 4096,
                         std::size_t grain_size = 1, bool verbose = false) {
  Distances(RandomNbrsQuery)
}

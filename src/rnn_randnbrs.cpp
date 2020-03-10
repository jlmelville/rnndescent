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
// [[Rcpp::depends(dqrng)]]
#include "convert_seed.h"
#include "dqrng_generator.h"
#include <dqrng.h>

#include "tdoann/progress.h"

#include "RcppPerpendicular.h"

#include "rnn_knnfactory.h"
#include "rnn_knnsort.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_rng.h"
#include "rnn_rtoheap.h"
#include "rnn_sample.h"

using namespace Rcpp;
using namespace tdoann;

template <typename Distance, typename IdxMatrix, typename DistMatrix,
          typename Base>
struct RandomNbrQueryWorker : public Base {

  Distance &distance;

  IdxMatrix nn_idx;
  DistMatrix nn_dist;

  int nrefs;
  int k;

  uint64_t seed;

  RandomNbrQueryWorker(Distance &distance, IntegerMatrix nn_idx,
                       NumericMatrix nn_dist)
      : distance(distance), nn_idx(nn_idx), nn_dist(nn_dist),
        nrefs(distance.nx), k(nn_idx.nrow()), seed(pseed()) {}

  void operator()(std::size_t begin, std::size_t end) {
    dqrng::rng64_t rng = parallel_rng(seed);
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

  int nr1;
  int k_minus_1;
  uint64_t seed;

  RandomNbrBuildWorker(Distance &distance, IntegerMatrix nn_idx,
                       NumericMatrix nn_dist)
      : distance(distance), nn_idx(nn_idx), nn_dist(nn_dist),
        nr1(nn_idx.ncol() - 1), k_minus_1(nn_idx.nrow() - 1), seed(pseed()) {}

  void operator()(std::size_t begin, std::size_t end) {
    dqrng::rng64_t rng = parallel_rng(seed);
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
      RandomNbrQueryWorker<Distance, IntegerMatrix, NumericMatrix, Empty>;
  template <typename D> using Worker = SerialRandomNbrQueryWorker<D>;
};

struct SerialRandomKnnBuild {
  using HeapAdd = HeapAddSymmetric;
  template <typename Distance>
  using SerialRandomNbrBuildWorker =
      RandomNbrBuildWorker<Distance, IntegerMatrix, NumericMatrix, Empty>;
  template <typename D> using Worker = SerialRandomNbrBuildWorker<D>;
};

template <template <typename> class RandomKnnWorker, typename Progress,
          typename Distance>
void rknn_serial(Progress &progress, Distance &distance, IntegerMatrix nn_idx,
                 NumericMatrix nn_dist, std::size_t block_size = 4096) {
  RandomKnnWorker<Distance> worker(distance, nn_idx, nn_dist);
  batch_serial_for(worker, progress, nn_idx.ncol(), block_size);
}

template <typename SerialRandomKnn> struct SerialRandomNbrsImpl {
  std::size_t block_size;
  SerialRandomNbrsImpl(std::size_t block_size) : block_size(block_size) {}

  template <typename Distance>
  void build_knn(Distance &distance, IntegerMatrix nn_idx,
                 NumericMatrix nn_dist, bool verbose) {
    RPProgress progress(1, verbose);
    rknn_serial<Worker>(progress, distance, nn_idx, nn_dist, block_size);
  }
  void sort_knn(IntegerMatrix nn_idx, NumericMatrix nn_dist) {
    sort_knn_graph<HeapAdd>(nn_idx, nn_dist);
  }

  template <typename D>
  using Worker = typename SerialRandomKnn::template Worker<D>;
  using HeapAdd = typename SerialRandomKnn::HeapAdd;
};

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
  void build_knn(Distance &distance, IntegerMatrix nn_idx,
                 NumericMatrix nn_dist, bool verbose) {
    RPProgress progress(1, verbose);

    Worker<Distance> worker(distance, nn_idx, nn_dist);
    batch_parallel_for(worker, progress, nn_idx.ncol(), block_size, grain_size);
  }
  void sort_knn(IntegerMatrix nn_idx, NumericMatrix nn_dist) {
    sort_knn_graph_parallel<HeapAdd>(nn_idx, nn_dist, block_size, grain_size);
  }

  template <typename D>
  using Worker = typename ParallelRandomKnn::template Worker<D>;
  using HeapAdd = typename ParallelRandomKnn::HeapAdd;
};

/* Macros */

#define RANDOM_NBRS_BUILD()                                                    \
  using KnnFactory = KnnBuildFactory<Distance>;                                \
  KnnFactory knn_factory(data);                                                \
  if (parallelize) {                                                           \
    using RandomNbrsImpl = ParallelRandomNbrsImpl<ParallelRandomKnnBuild>;     \
    RandomNbrsImpl impl(block_size, grain_size);                               \
    RANDOM_NBRS_IMPL()                                                         \
  } else {                                                                     \
    using RandomNbrsImpl = SerialRandomNbrsImpl<SerialRandomKnnBuild>;         \
    RandomNbrsImpl impl(block_size);                                           \
    RANDOM_NBRS_IMPL()                                                         \
  }

#define RANDOM_NBRS_QUERY()                                                    \
  using KnnFactory = KnnQueryFactory<Distance>;                                \
  KnnFactory knn_factory(reference, query);                                    \
  if (parallelize) {                                                           \
    using RandomNbrsImpl = ParallelRandomNbrsImpl<ParallelRandomKnnQuery>;     \
    RandomNbrsImpl impl(block_size, grain_size);                               \
    RANDOM_NBRS_IMPL()                                                         \
  } else {                                                                     \
    using RandomNbrsImpl = SerialRandomNbrsImpl<SerialRandomKnnQuery>;         \
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
  auto distance = knn_factory.create_distance();
  auto indices = knn_factory.create_index_matrix(k);
  auto dist = knn_factory.create_distance_matrix(k);

  impl.build_knn(distance, indices, dist, verbose);

  indices = transpose(indices);
  dist = transpose(dist);

  if (order_by_distance) {
    impl.sort_knn(indices, dist);
  }

  return List::create(_("idx") = indices, _("dist") = dist);
}

/* Exports */

// [[Rcpp::export]]
List random_knn_cpp(Rcpp::NumericMatrix data, int k,
                    const std::string &metric = "euclidean",
                    bool order_by_distance = true, bool parallelize = false,
                    std::size_t block_size = 4096, std::size_t grain_size = 1,
                    bool verbose = false){
    DISPATCH_ON_DISTANCES(RANDOM_NBRS_BUILD)}

// [[Rcpp::export]]
List random_knn_query_cpp(NumericMatrix reference, NumericMatrix query, int k,
                          const std::string &metric = "euclidean",
                          bool order_by_distance = true,
                          bool parallelize = false,
                          std::size_t block_size = 4096,
                          std::size_t grain_size = 1, bool verbose = false) {
  DISPATCH_ON_QUERY_DISTANCES(RANDOM_NBRS_QUERY)
}

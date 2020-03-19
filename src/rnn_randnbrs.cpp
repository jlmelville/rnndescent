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

#include "tdoann/nngraph.h"
#include "tdoann/parallel.h"
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

template <typename Distance>
struct RandomNbrQueryWorker : public BatchParallelWorker {

  Distance &distance;

  std::size_t n_points;
  std::size_t k;
  std::vector<int> nn_idx;
  std::vector<double> nn_dist;

  int nrefs;

  uint64_t seed;

  RandomNbrQueryWorker(Distance &distance, std::size_t n_points, std::size_t k)
      : distance(distance), n_points(n_points), k(k), nn_idx(n_points * k),
        nn_dist(n_points * k, 0.0), nrefs(distance.nx), seed(pseed()) {}

  void operator()(std::size_t begin, std::size_t end) {
    dqrng::rng64_t rng = parallel_rng(seed);
    rng->seed(seed, end);
    for (int qi = static_cast<int>(begin); qi < static_cast<int>(end); qi++) {
      auto idxi = sample<uint32_t>(rng, nrefs, k);
      std::size_t kqi = k * qi;
      for (std::size_t j = 0; j < k; j++) {
        auto &ri = idxi[j];
        nn_idx[j + kqi] = ri;
        nn_dist[j + kqi] = distance(ri, qi); // distance calcs are 0-indexed
      }
    }
  }
};

template <typename Distance>
struct RandomNbrBuildWorker : public BatchParallelWorker {

  Distance &distance;

  std::size_t n_points;
  std::size_t k;
  std::vector<int> nn_idx;
  std::vector<double> nn_dist;

  int n_points_minus_1;
  int k_minus_1;
  uint64_t seed;

  RandomNbrBuildWorker(Distance &distance, std::size_t n_points, std::size_t k)
      : distance(distance), n_points(n_points), k(k), nn_idx(n_points * k),
        nn_dist(n_points * k), n_points_minus_1(n_points - 1), k_minus_1(k - 1),
        seed(pseed()) {}

  void operator()(std::size_t begin, std::size_t end) {
    dqrng::rng64_t rng = parallel_rng(seed);
    rng->seed(seed, end);
    for (auto qi = static_cast<int>(begin); qi < static_cast<int>(end); qi++) {
      std::size_t kqi = k * qi;
      std::size_t kqi1 = kqi + 1;
      nn_idx[0 + kqi] = qi;
      auto ris = sample<uint32_t>(rng, n_points_minus_1, k_minus_1);
      for (auto j = 0; j < k_minus_1; j++) {
        int ri = ris[j];
        if (ri >= qi) {
          ri += 1;
        }
        nn_idx[j + kqi1] = ri;
        nn_dist[j + kqi1] = distance(ri, qi); // distance calcs are 0-indexed
      }
    }
  }
};

struct SerialRandomKnnQuery {
  using HeapAdd = tdoann::HeapAddQuery;
  template <typename Distance>
  using SerialRandomNbrQueryWorker = RandomNbrQueryWorker<Distance>;
  template <typename D> using Worker = SerialRandomNbrQueryWorker<D>;
};

struct SerialRandomKnnBuild {
  using HeapAdd = tdoann::HeapAddSymmetric;
  template <typename Distance>
  using SerialRandomNbrBuildWorker = RandomNbrBuildWorker<Distance>;
  template <typename D> using Worker = SerialRandomNbrBuildWorker<D>;
};

template <typename SerialRandomKnn> struct SerialRandomNbrsImpl {
  std::size_t block_size;
  SerialRandomNbrsImpl(std::size_t block_size) : block_size(block_size) {}

  template <typename Distance>
  NNGraph build_knn(Distance &distance, std::size_t n_points, std::size_t k,
                    bool verbose) {
    RPProgress progress(1, verbose);

    Worker<Distance> worker(distance, n_points, k);
    tdoann::batch_serial_for(worker, progress, n_points, block_size);
    return NNGraph(worker.nn_idx, worker.nn_dist, n_points);
  }
  void sort_knn(NNGraph &nn_graph) {
    sort_knn_graph<HeapAdd, RInterruptableProgress>(nn_graph);
  }

  template <typename D>
  using Worker = typename SerialRandomKnn::template Worker<D>;
  using HeapAdd = typename SerialRandomKnn::HeapAdd;
};

template <typename Distance>
using ParallelRandomNbrQueryWorker = RandomNbrQueryWorker<Distance>;

template <typename Distance>
using ParallelRandomNbrBuildWorker = RandomNbrBuildWorker<Distance>;

struct ParallelRandomKnnBuild {
  using HeapAdd = tdoann::LockingHeapAddSymmetric;
  template <typename D> using Worker = ParallelRandomNbrBuildWorker<D>;
};

struct ParallelRandomKnnQuery {
  using HeapAdd = tdoann::HeapAddQuery;
  template <typename D> using Worker = ParallelRandomNbrQueryWorker<D>;
};

template <typename ParallelRandomKnn> struct ParallelRandomNbrsImpl {
  std::size_t n_threads;
  std::size_t block_size;
  std::size_t grain_size;

  ParallelRandomNbrsImpl(std::size_t n_threads = 0,
                         std::size_t block_size = 4096,
                         std::size_t grain_size = 1)
      : n_threads(n_threads), block_size(block_size), grain_size(grain_size) {}

  template <typename Distance>
  NNGraph build_knn(Distance &distance, std::size_t n_points, std::size_t k,
                    bool verbose) {
    RPProgress progress(1, verbose);

    Worker<Distance> worker(distance, n_points, k);
    tdoann::batch_parallel_for<decltype(progress), decltype(worker), RParallel>(
        worker, progress, n_points, n_threads, block_size, grain_size);

    return NNGraph(worker.nn_idx, worker.nn_dist, n_points);
  }
  void sort_knn(NNGraph &nn_graph) {
    // use Null Progress for parallel case
    sort_knn_graph_parallel<HeapAdd, tdoann::NullProgress, SimpleNeighborHeap,
                            RParallel>(nn_graph, n_threads, block_size,
                                       grain_size);
  }

  template <typename D>
  using Worker = typename ParallelRandomKnn::template Worker<D>;
  using HeapAdd = typename ParallelRandomKnn::HeapAdd;
};

/* Macros */

#define RANDOM_NBRS_BUILD()                                                    \
  using KnnFactory = KnnBuildFactory<Distance>;                                \
  KnnFactory knn_factory(data);                                                \
  if (n_threads > 0) {                                                         \
    using RandomNbrsImpl = ParallelRandomNbrsImpl<ParallelRandomKnnBuild>;     \
    RandomNbrsImpl impl(n_threads, block_size, grain_size);                    \
    RANDOM_NBRS_IMPL()                                                         \
  } else {                                                                     \
    using RandomNbrsImpl = SerialRandomNbrsImpl<SerialRandomKnnBuild>;         \
    RandomNbrsImpl impl(block_size);                                           \
    RANDOM_NBRS_IMPL()                                                         \
  }

#define RANDOM_NBRS_QUERY()                                                    \
  using KnnFactory = KnnQueryFactory<Distance>;                                \
  KnnFactory knn_factory(reference, query);                                    \
  if (n_threads > 0) {                                                         \
    using RandomNbrsImpl = ParallelRandomNbrsImpl<ParallelRandomKnnQuery>;     \
    RandomNbrsImpl impl(n_threads, block_size, grain_size);                    \
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
  std::size_t n_points = knn_factory.n_points;
  auto nn_graph = impl.build_knn(distance, n_points, k, verbose);

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

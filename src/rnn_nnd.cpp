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

#include "tdoann/distance.h"
#include "tdoann/graphupdate.h"
#include "tdoann/heap.h"
#include "tdoann/nndescent.h"
#include "tdoann/nndparallel.h"

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_rng.h"
#include "rnn_rtoheap.h"

using namespace Rcpp;

#define NND_IMPL()                                                             \
  return nnd_impl.get_nn<GraphUpdate, Distance, Progress, NNDProgress>(        \
      nn_idx, nn_dist, max_candidates, n_iters, delta, verbose);

#define NND_QUERY_IMPL()                                                       \
  return nnd_impl.get_nn<GraphUpdate, Distance, Progress, NNDProgress>(        \
      nn_idx, nn_dist, max_candidates, epsilon, n_iters, verbose);

#define NND_PROGRESS()                                                         \
  if (progress == "bar") {                                                     \
    using Progress = RPProgress;                                               \
    using NNDProgress = tdoann::NNDProgress<Progress>;                         \
    NND_IMPL()                                                                 \
  } else {                                                                     \
    using Progress = RIterProgress;                                            \
    using NNDProgress = tdoann::HeapSumProgress<Progress>;                     \
    NND_IMPL()                                                                 \
  }

#define NND_QUERY_PROGRESS()                                                   \
  using Progress = RPProgress;                                                 \
  using NNDProgress = tdoann::NNDProgress<Progress>;                           \
  NND_QUERY_IMPL()

#define NND_BUILD_UPDATER()                                                    \
  if (n_threads > 0) {                                                         \
    using NNDImpl = NNDBuildParallel;                                          \
    NNDImpl nnd_impl(data, n_threads, block_size, grain_size);                 \
    if (low_memory) {                                                          \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::Batch>;            \
      NND_PROGRESS()                                                           \
    } else {                                                                   \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::BatchHiMem>;       \
      NND_PROGRESS()                                                           \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDBuildSerial;                                            \
    NNDImpl nnd_impl(data);                                                    \
    if (low_memory) {                                                          \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::Serial>;           \
      NND_PROGRESS()                                                           \
    } else {                                                                   \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::SerialHiMem>;      \
      NND_PROGRESS()                                                           \
    }                                                                          \
  }

#define NND_QUERY_UPDATER()                                                    \
  if (n_threads > 0) {                                                         \
    using NNDImpl = NNDQueryParallel;                                          \
    NNDImpl nnd_impl(reference, query, reference_idx, reference_dist,          \
                     n_threads, grain_size);                                   \
    if (low_memory) {                                                          \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::QueryBatch>;       \
      NND_QUERY_PROGRESS()                                                     \
    } else {                                                                   \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::QueryBatchHiMem>;  \
      NND_QUERY_PROGRESS()                                                     \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDQuerySerial;                                            \
    NNDImpl nnd_impl(reference, query, reference_idx, reference_dist);         \
    if (low_memory) {                                                          \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::QuerySerial>;      \
      NND_QUERY_PROGRESS()                                                     \
    } else {                                                                   \
      using GraphUpdate = tdoann::upd::Factory<tdoann::upd::QuerySerialHiMem>; \
      NND_QUERY_PROGRESS()                                                     \
    }                                                                          \
  }

struct NNDBuildSerial {
  NumericMatrix data;

  NNDBuildSerial(NumericMatrix data) : data(data) {}

  template <typename GraphUpdate, typename Distance, typename Progress,
            typename NNDProgress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false) -> List {
    auto data_vec = r2dvt<Distance>(data);
    Distance distance(data_vec, data.ncol());

    using Out = typename Distance::Output;
    using Index = typename Distance::Index;
    using Graph = tdoann::NNGraph<Out, Index>;

    auto nn_idx_copy = Rcpp::clone(nn_idx);
    Graph nn_graph =
        r_to_graph<Out, Index>(nn_idx_copy, nn_dist, data.nrow() - 1);

    const std::size_t g2h_block_size = 1000;
    const bool is_transposed = true;
    auto nnd_heap =
        tdoann::graph_to_heap_serial<tdoann::HeapAddSymmetric, tdoann::NNDHeap>(
            nn_graph, g2h_block_size, is_transposed);

    auto graph_updater = GraphUpdate::create(nnd_heap, distance);

    Progress progress(n_iters, verbose);
    NNDProgress nnd_progress(progress);
    RRand rand;
    tdoann::nnd_build(graph_updater, max_candidates, n_iters, delta, rand,
                      nnd_progress);
    nnd_heap.deheap_sort();
    Graph result = tdoann::heap_to_graph(nnd_heap);

    return graph_to_r(result);
  }
};

struct NNDBuildParallel {
  NumericMatrix data;

  std::size_t n_threads;
  std::size_t block_size;
  std::size_t grain_size;

  NNDBuildParallel(NumericMatrix data, std::size_t n_threads,
                   std::size_t block_size, std::size_t grain_size)
      : data(data), n_threads(n_threads), block_size(block_size),
        grain_size(grain_size) {}

  template <typename GraphUpdate, typename Distance, typename Progress,
            typename NNDProgress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false) -> List {
    auto data_vec = r2dvt<Distance>(data);

    Distance distance(data_vec, data.ncol());

    using Out = typename Distance::Output;
    using Index = typename Distance::Index;
    using Graph = tdoann::NNGraph<Out, Index>;

    auto nn_idx_copy = Rcpp::clone(nn_idx);
    Graph nn_graph =
        r_to_graph<Out, Index>(nn_idx_copy, nn_dist, data.nrow() - 1);
    const std::size_t g2h_block_size = 1000;
    const bool is_transposed = true;
    auto nnd_heap =
        tdoann::graph_to_heap_parallel<tdoann::LockingHeapAddSymmetric,
                                       tdoann::NNDHeap>(
            nn_graph, n_threads, g2h_block_size, grain_size, is_transposed);
    auto graph_updater = GraphUpdate::create(nnd_heap, distance);

    Progress progress(n_iters, verbose);
    NNDProgress nnd_progress(progress);
    ParallelRand parallel_rand;

    tdoann::nnd_build_parallel<RParallel>(
        graph_updater, max_candidates, n_iters, delta, nnd_progress,
        parallel_rand, n_threads, block_size, grain_size);

    tdoann::sort_heap_parallel(nnd_heap, n_threads, block_size, grain_size);
    Graph result = tdoann::heap_to_graph(nnd_heap);
    return graph_to_r(result);
  }
};

struct NNDQuerySerial {
  NumericMatrix reference;
  NumericMatrix query;

  IntegerMatrix ref_idx;
  NumericMatrix ref_dist;

  NNDQuerySerial(NumericMatrix reference, NumericMatrix query,
                 IntegerMatrix ref_idx, NumericMatrix ref_dist)
      : reference(reference), query(query), ref_idx(ref_idx),
        ref_dist(ref_dist) {}

  template <typename GraphUpdate, typename Distance, typename Progress,
            typename NNDProgress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              std::size_t max_candidates = 50, double epsilon = 0.1,
              std::size_t n_iters = 10, bool verbose = false) -> List {

    using Out = typename Distance::Output;
    using Index = typename Distance::Index;
    using Graph = tdoann::NNGraph<Out, Index>;

    auto ref_vec = r2dvt<Distance>(reference);
    auto query_vec = r2dvt<Distance>(query);

    auto nn_idx_copy = Rcpp::clone(nn_idx);
    auto nn_graph =
        r_to_graph<Out, Index>(nn_idx_copy, nn_dist, reference.nrow() - 1);

    const std::size_t g2h_block_size = 1000;
    const bool is_transposed = true;
    auto nnd_heap =
        tdoann::graph_to_heap_serial<tdoann::HeapAddQuery, tdoann::NNDHeap>(
            nn_graph, g2h_block_size, is_transposed);

    Distance distance(ref_vec, query_vec, reference.ncol());

    auto graph_updater = GraphUpdate::create(nnd_heap, distance);

    auto ref_idx_vec = r_to_idx<Index>(ref_idx);
    auto ref_dist_vec = Rcpp::as<std::vector<Out>>(ref_dist);

    Progress progress(query.nrow(), verbose);
    NNDProgress nnd_progress(progress);

    tdoann::nnd_query(ref_idx_vec, ref_dist_vec, graph_updater, max_candidates,
                      epsilon, n_iters, nnd_progress);
    nnd_heap.deheap_sort();
    Graph result = heap_to_graph(nnd_heap);
    return graph_to_r(result);
  }
};

struct NNDQueryParallel {
  NumericMatrix reference;
  NumericMatrix query;

  IntegerMatrix ref_idx;
  NumericMatrix ref_dist;

  std::size_t n_threads;
  std::size_t grain_size;

  NNDQueryParallel(NumericMatrix reference, NumericMatrix query,
                   IntegerMatrix ref_idx, NumericMatrix ref_dist,
                   std::size_t n_threads, std::size_t grain_size)
      : reference(reference), query(query), ref_idx(ref_idx),
        ref_dist(ref_dist), n_threads(n_threads), grain_size(grain_size) {}

  template <typename GraphUpdate, typename Distance, typename Progress,
            typename NNDProgress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              std::size_t max_candidates = 50, double epsilon = 0.1,
              std::size_t n_iters = 10, bool verbose = false) -> List {
    using Out = typename Distance::Output;
    using Index = typename Distance::Index;
    using Graph = tdoann::NNGraph<Out, Index>;

    auto ref_vec = r2dvt<Distance>(reference);
    auto query_vec = r2dvt<Distance>(query);
    auto nn_idx_copy = Rcpp::clone(nn_idx);
    auto nn_graph =
        r_to_graph<Out, Index>(nn_idx_copy, nn_dist, reference.nrow() - 1);

    const std::size_t g2h_block_size = 1000;
    const bool is_transposed = true;
    auto nnd_heap =
        tdoann::graph_to_heap_serial<tdoann::HeapAddQuery, tdoann::NNDHeap>(
            nn_graph, g2h_block_size, is_transposed);

    auto ref_idx_vec = r_to_idx<Index>(ref_idx);
    auto ref_dist_vec = Rcpp::as<std::vector<Out>>(ref_dist);

    Distance distance(ref_vec, query_vec, reference.ncol());
    auto graph_updater = GraphUpdate::create(nnd_heap, distance);
    Progress progress(n_iters, verbose);
    NNDProgress nnd_progress(progress);

    tdoann::nnd_query_parallel<RParallel>(
        ref_idx_vec, ref_dist_vec, graph_updater, max_candidates, epsilon,
        n_iters, nnd_progress, n_threads, grain_size);

    nnd_heap.deheap_sort();
    Graph result = heap_to_graph(nnd_heap);
    return graph_to_r(result);
  }
};

// [[Rcpp::export]]
List nn_descent(NumericMatrix data, IntegerMatrix nn_idx, NumericMatrix nn_dist,
                const std::string &metric = "euclidean",
                std::size_t max_candidates = 50, std::size_t n_iters = 10,
                double delta = 0.001, bool low_memory = true,
                std::size_t n_threads = 0, std::size_t block_size = 16384,
                std::size_t grain_size = 1, bool verbose = false,
                const std::string &progress = "bar") {
  DISPATCH_ON_DISTANCES(NND_BUILD_UPDATER);
}

// [[Rcpp::export]]
List nn_descent_query(NumericMatrix reference, IntegerMatrix reference_idx,
                      NumericMatrix reference_dist, NumericMatrix query,
                      IntegerMatrix nn_idx, NumericMatrix nn_dist,
                      const std::string &metric = "euclidean",
                      std::size_t max_candidates = 50, double epsilon = 0.1,
                      std::size_t n_iters = 10, bool low_memory = true,
                      std::size_t n_threads = 0, std::size_t grain_size = 1,
                      bool verbose = false,
                      const std::string &progress = "bar") {
  DISPATCH_ON_QUERY_DISTANCES(NND_QUERY_UPDATER)
}

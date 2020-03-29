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

#include "rnn_candidatepriority.h"
#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_rng.h"
#include "rnn_rtoheap.h"

using namespace Rcpp;

#define NND_IMPL()                                                             \
  return nnd_impl.get_nn<GUFactoryT, Distance, CandidatePriority, Progress>(   \
      nn_idx, nn_dist, cp, max_candidates, n_iters, delta, verbose);

#define NND_PROGRESS()                                                         \
  if (progress == "bar") {                                                     \
    using Progress = RPProgress;                                               \
    NND_IMPL()                                                                 \
  } else {                                                                     \
    using Progress = HeapSumProgress;                                          \
    NND_IMPL()                                                                 \
  }

#define NND_CANDIDATE_PRIORITY_SERIAL()                                        \
  if (candidate_priority == "random") {                                        \
    using CandidatePriority = tdoann::cp::Factory<rnnd::cp::RandomSerial>;     \
    CandidatePriority cp;                                                      \
    NND_PROGRESS()                                                             \
  } else if (candidate_priority == "distance") {                               \
    using CandidatePriority = tdoann::cp::Factory<tdoann::cp::LowDistance>;    \
    CandidatePriority cp;                                                      \
    NND_PROGRESS()                                                             \
  } else if (candidate_priority == "highdistance") {                           \
    using CandidatePriority = tdoann::cp::Factory<tdoann::cp::HighDistance>;   \
    CandidatePriority cp;                                                      \
    NND_PROGRESS()                                                             \
  } else {                                                                     \
    stop("Unknown candidate priority '%s'", candidate_priority);               \
  }

#define NND_CANDIDATE_PRIORITY_PARALLEL()                                      \
  if (candidate_priority == "random") {                                        \
    using CandidatePriority = tdoann::cp::Factory<rnnd::cp::RandomParallel>;   \
    CandidatePriority cp(pseed());                                             \
    NND_PROGRESS()                                                             \
  } else if (candidate_priority == "distance") {                               \
    using CandidatePriority = tdoann::cp::Factory<tdoann::cp::LowDistance>;    \
    CandidatePriority cp;                                                      \
    NND_PROGRESS()                                                             \
  } else if (candidate_priority == "highdistance") {                           \
    using CandidatePriority = tdoann::cp::Factory<tdoann::cp::HighDistance>;   \
    CandidatePriority cp;                                                      \
    NND_PROGRESS()                                                             \
  } else {                                                                     \
    stop("Unknown candidate priority '%s'", candidate_priority);               \
  }

#define NND_BUILD_UPDATER()                                                    \
  if (n_threads > 0) {                                                         \
    using NNDImpl = NNDBuildParallel;                                          \
    NNDImpl nnd_impl(data, n_threads, block_size, grain_size);                 \
    if (low_memory) {                                                          \
      using GUFactoryT = tdoann::GUFactory<tdoann::BatchGraphUpdater>;         \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    } else {                                                                   \
      using GUFactoryT = tdoann::GUFactory<tdoann::BatchGraphUpdaterHiMem>;    \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDBuildSerial;                                            \
    NNDImpl nnd_impl(data);                                                    \
    if (low_memory) {                                                          \
      using GUFactoryT = tdoann::GUFactory<tdoann::SerialGraphUpdater>;        \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    } else {                                                                   \
      using GUFactoryT = tdoann::GUFactory<tdoann::SerialGraphUpdaterHiMem>;   \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    }                                                                          \
  }

#define NND_QUERY_UPDATER()                                                    \
  if (n_threads > 0) {                                                         \
    using NNDImpl = NNDQueryParallel;                                          \
    NNDImpl nnd_impl(reference, query, reference_idx, n_threads, block_size,   \
                     grain_size);                                              \
    if (low_memory) {                                                          \
      using GUFactoryT = tdoann::GUFactory<tdoann::QueryBatchGraphUpdater>;    \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    } else {                                                                   \
      using GUFactoryT =                                                       \
          tdoann::GUFactory<tdoann::QueryBatchGraphUpdaterHiMem>;              \
      NND_CANDIDATE_PRIORITY_PARALLEL()                                        \
    }                                                                          \
  } else {                                                                     \
    using NNDImpl = NNDQuerySerial;                                            \
    NNDImpl nnd_impl(reference, query, reference_idx);                         \
    if (low_memory) {                                                          \
      using GUFactoryT = tdoann::GUFactory<tdoann::QuerySerialGraphUpdater>;   \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    } else {                                                                   \
      using GUFactoryT =                                                       \
          tdoann::GUFactory<tdoann::QuerySerialGraphUpdaterHiMem>;             \
      NND_CANDIDATE_PRIORITY_SERIAL()                                          \
    }                                                                          \
  }

struct NNDBuildSerial {
  NumericMatrix data;

  NNDBuildSerial(NumericMatrix data) : data(data) {}

  template <typename GUFactoryT, typename Distance, typename CandidatePriority,
            typename Progress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              CandidatePriority &candidate_priority,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false) -> List {
    auto data_vec = r2dvt<Distance>(data);
    auto init_nn = r_to_graph(nn_idx, nn_dist, data.nrow() - 1);

    auto result = tdoann::nnd_build<Distance, GUFactoryT, Progress>(
        data_vec, data.ncol(), init_nn, max_candidates, n_iters,
        candidate_priority, delta, verbose);

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

  template <typename GUFactoryT, typename Distance, typename CandidatePriority,
            typename Progress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              CandidatePriority &candidate_priority,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false) -> List {
    auto data_vec = r2dvt<Distance>(data);
    auto init_nn = r_to_graph(nn_idx, nn_dist, data.nrow() - 1);

    auto result =
        tdoann::nnd_build_parallel<Distance, GUFactoryT, Progress, RParallel>(
            data_vec, data.ncol(), init_nn, max_candidates, n_iters,
            candidate_priority, delta, n_threads, block_size, grain_size,
            verbose);

    return graph_to_r(result);
  }
};

struct NNDQuerySerial {
  NumericMatrix reference;
  NumericMatrix query;

  IntegerMatrix ref_idx;

  NNDQuerySerial(NumericMatrix reference, NumericMatrix query,
                 IntegerMatrix ref_idx)
      : reference(reference), query(query), ref_idx(ref_idx) {}

  template <typename GUFactoryT, typename Distance, typename CandidatePriority,
            typename Progress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              CandidatePriority &candidate_priority,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false) -> List {

    auto ref_vec = r2dvt<Distance>(reference);
    auto query_vec = r2dvt<Distance>(query);
    auto nn_init = r_to_graph(nn_idx, nn_dist, reference.nrow() - 1);
    auto ref_idx_vec = r_to_idx<std::size_t>(ref_idx);

    auto result = tdoann::nnd_query<Distance, GUFactoryT, Progress>(
        ref_vec, reference.ncol(), query_vec, nn_init, ref_idx_vec,
        max_candidates, n_iters, candidate_priority, delta, verbose);

    return graph_to_r(result);
  }
};

struct NNDQueryParallel {
  NumericMatrix reference;
  NumericMatrix query;
  IntegerMatrix ref_idx;

  std::size_t n_threads;
  std::size_t block_size;
  std::size_t grain_size;

  NNDQueryParallel(NumericMatrix reference, NumericMatrix query,
                   IntegerMatrix ref_idx, std::size_t n_threads,
                   std::size_t block_size, std::size_t grain_size)
      : reference(reference), query(query), ref_idx(ref_idx),
        n_threads(n_threads), block_size(block_size), grain_size(grain_size) {}

  template <typename GUFactoryT, typename Distance, typename CandidatePriority,
            typename Progress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              CandidatePriority &candidate_priority,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false) -> List {
    auto ref_vec = r2dvt<Distance>(reference);
    auto query_vec = r2dvt<Distance>(query);
    auto nn_init = r_to_graph(nn_idx, nn_dist, reference.nrow() - 1);
    auto ref_idx_vec = r_to_idx<std::size_t>(ref_idx);

    auto result =
        tdoann::nnd_query_parallel<Distance, GUFactoryT, Progress, RParallel>(
            ref_vec, reference.ncol(), query_vec, nn_init, ref_idx_vec,
            max_candidates, n_iters, candidate_priority, delta, n_threads,
            block_size, grain_size, verbose);

    return graph_to_r(result);
  }
};

// [[Rcpp::export]]
List nn_descent(NumericMatrix data, IntegerMatrix nn_idx, NumericMatrix nn_dist,
                const std::string &metric = "euclidean",
                std::size_t max_candidates = 50, std::size_t n_iters = 10,
                double delta = 0.001, bool low_memory = true,
                const std::string &candidate_priority = "random",
                std::size_t n_threads = 0, std::size_t block_size = 16384,
                std::size_t grain_size = 1, bool verbose = false,
                const std::string &progress = "bar") {
  DISPATCH_ON_DISTANCES(NND_BUILD_UPDATER);
}

// [[Rcpp::export]]
List nn_descent_query(NumericMatrix reference, IntegerMatrix reference_idx,
                      NumericMatrix query, IntegerMatrix nn_idx,
                      NumericMatrix nn_dist,
                      const std::string &metric = "euclidean",
                      std::size_t max_candidates = 50, std::size_t n_iters = 10,
                      double delta = 0.001, bool low_memory = true,
                      const std::string &candidate_priority = "random",
                      std::size_t n_threads = 0, std::size_t block_size = 16384,
                      std::size_t grain_size = 1, bool verbose = false,
                      const std::string &progress = "bar") {
  DISPATCH_ON_QUERY_DISTANCES(NND_QUERY_UPDATER)
}

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

// #include "rnn_candidatepriority.h"
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
  using NNDImpl = NNDQuerySerial;                                              \
  NNDImpl nnd_impl(reference, query, reference_idx);                           \
  if (low_memory) {                                                            \
    using GraphUpdate = tdoann::upd::Factory<tdoann::upd::QuerySerial>;        \
    NND_PROGRESS()                                                             \
  } else {                                                                     \
    using GraphUpdate = tdoann::upd::Factory<tdoann::upd::QuerySerialHiMem>;   \
    NND_PROGRESS()                                                             \
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
    auto nn_init =
        r_to_graph<typename Distance::Output, typename Distance::Index>(
            nn_idx, nn_dist, data.nrow() - 1);

    Progress progress(n_iters, verbose);
    NNDProgress nnd_progress(progress);

    auto result = tdoann::nnd_build<Distance, GraphUpdate, NNDProgress>(
        data_vec, data.ncol(), nn_init, max_candidates, n_iters, delta,
        nnd_progress, verbose);

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
    auto init_nn =
        r_to_graph<typename Distance::Output, typename Distance::Index>(
            nn_idx, nn_dist, data.nrow() - 1);

    Progress progress(n_iters, verbose);

    auto result =
        tdoann::nnd_build_parallel<Distance, GraphUpdate, Progress, RParallel>(
            data_vec, data.ncol(), init_nn, max_candidates, n_iters, delta,
            progress, n_threads, block_size, grain_size, verbose);

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

  template <typename GraphUpdate, typename Distance, typename Progress,
            typename NNDProgress>
  auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
              std::size_t max_candidates = 50, std::size_t n_iters = 10,
              double delta = 0.001, bool verbose = false) -> List {

    auto ref_vec = r2dvt<Distance>(reference);
    auto query_vec = r2dvt<Distance>(query);
    auto nn_init =
        r_to_graph<typename Distance::Output, typename Distance::Index>(
            nn_idx, nn_dist, reference.nrow() - 1);
    auto ref_idx_vec = r_to_idx<typename Distance::Index>(ref_idx);
    Progress progress(n_iters, verbose);
    NNDProgress nnd_progress(progress);
    RRand rand;

    auto result = tdoann::nnd_query<Distance, GraphUpdate, NNDProgress>(
        ref_vec, reference.ncol(), query_vec, nn_init, ref_idx_vec,
        max_candidates, n_iters, delta, rand, nnd_progress, verbose);

    return graph_to_r(result);
  }
};

// struct NNDQueryParallel {
//   NumericMatrix reference;
//   NumericMatrix query;
//   IntegerMatrix ref_idx;
//
//   std::size_t n_threads;
//   std::size_t block_size;
//   std::size_t grain_size;
//
//   NNDQueryParallel(NumericMatrix reference, NumericMatrix query,
//                    IntegerMatrix ref_idx, std::size_t n_threads,
//                    std::size_t block_size, std::size_t grain_size)
//       : reference(reference), query(query), ref_idx(ref_idx),
//         n_threads(n_threads), block_size(block_size), grain_size(grain_size)
//         {}
//
//   template <typename GraphUpdate, typename Distance, typename
//   CandidatePriority,
//             typename Progress>
//   auto get_nn(IntegerMatrix nn_idx, NumericMatrix nn_dist,
//               CandidatePriority &candidate_priority,
//               std::size_t max_candidates = 50, std::size_t n_iters = 10,
//               double delta = 0.001, bool verbose = false) -> List {
//     auto ref_vec = r2dvt<Distance>(reference);
//     auto query_vec = r2dvt<Distance>(query);
//     auto nn_init = r_to_graph(nn_idx, nn_dist, reference.nrow() - 1);
//     auto ref_idx_vec = r_to_idx<std::size_t>(ref_idx);
//
//     auto result =
//         tdoann::nnd_query_parallel<Distance, GraphUpdate, Progress,
//         RParallel>(
//             ref_vec, reference.ncol(), query_vec, nn_init, ref_idx_vec,
//             max_candidates, n_iters, candidate_priority, delta, n_threads,
//             block_size, grain_size, verbose);
//
//     return graph_to_r(result);
//   }
// };

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
                      NumericMatrix query, IntegerMatrix nn_idx,
                      NumericMatrix nn_dist,
                      const std::string &metric = "euclidean",
                      std::size_t max_candidates = 50, std::size_t n_iters = 10,
                      double delta = 0.001, bool low_memory = true,
                      std::size_t n_threads = 0, std::size_t block_size = 16384,
                      std::size_t grain_size = 1, bool verbose = false,
                      const std::string &progress = "bar") {
  DISPATCH_ON_QUERY_DISTANCES(NND_QUERY_UPDATER)
}

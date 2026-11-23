// BSD 2-Clause License
//
// Copyright 2023 James Melville
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef TDOANN_RPTREESEARCH_H
#define TDOANN_RPTREESEARCH_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "heap.h"
#include "parallel.h"
#include "random.h"

namespace tdoann {

constexpr std::size_t grouped_queries_per_worker = 32;

template <typename Forest, typename Distance, typename Idx>
auto search_forest_queries(const Forest &forest, const Distance &distance,
                           uint32_t n_nbrs,
                           ParallelRandomIntProvider<Idx> &rng_provider,
                           bool cache, std::size_t n_threads,
                           ProgressBase &progress, const Executor &executor,
                           std::size_t queries_per_worker)
    -> NNHeap<typename Distance::Output, Idx> {
  const auto n_queries = distance.get_ny();
  NNHeap<typename Distance::Output, Idx> current_graph(n_queries, n_nbrs);

  const bool grouped = n_threads > 1 && queries_per_worker > 1;
  rng_provider.initialize();
  auto visit = [&](std::size_t i, RandomIntGenerator<Idx> &rng) {
    if (cache) {
      search_forest_cache(forest, distance, static_cast<Idx>(i), rng,
                          current_graph);
    } else {
      search_forest(forest, distance, static_cast<Idx>(i), rng, current_graph);
    }
  };
  auto worker = [&](std::size_t begin, std::size_t end) {
    if (grouped) {
      for (auto i = begin; i < end; ++i) {
        // Match the exclusive end of each original one-query worker range.
        auto rng = rng_provider.get_parallel_instance(i + 1);
        visit(i, *rng);
      }
    } else {
      auto rng = rng_provider.get_parallel_instance(end);
      for (auto i = begin; i < end; ++i) {
        visit(i, *rng);
      }
    }
  };
  progress.set_n_iters(n_queries);
  const auto batch_size = std::max(n_threads, std::size_t{1});
  ExecutionParams exec_params(grouped ? batch_size * queries_per_worker
                                      : batch_size);
  dispatch_work(worker, n_queries, n_threads, exec_params, progress, executor);

  return current_graph;
}

} // namespace tdoann

#endif // TDOANN_RPTREESEARCH_H

//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2021 James Melville
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

// NOLINTBEGIN(modernize-use-trailing-return-type)

#include "rnndescent/random.h"
#include "tdoann/nngraph.h"
#include "tdoann/prepare.h"

#include <Rcpp.h>

#include "rnn_distance.h"
#include "rnn_parallel.h"
#include "rnn_progress.h"

#include <atomic>
#include <chrono>
#include <future>

using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::LogicalMatrix;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

namespace {

template <typename Out, typename Idx>
auto diversify_by_degree(const tdoann::SparseNNGraph<Out, Idx> &graph,
                         const tdoann::BaseDistance<Out, Idx> &distance,
                         double prune_probability, std::size_t n_threads,
                         bool verbose) -> tdoann::SparseNNGraph<Out, Idx> {
  struct Range {
    std::size_t begin;
    std::size_t end;
    uint64_t comparisons;
  };

  // Keep the original ranges and use their end rows as stream IDs. Moving
  // entire ranges preserves the draws assigned to every fractional prune.
  const auto batch_size = tdoann::diversification_rows_per_worker * n_threads;
  std::vector<Range> ranges;
  for (std::size_t begin = 0; begin < graph.n_points; begin += batch_size) {
    const auto end = std::min(begin + batch_size, graph.n_points);
    for (const auto &[first, last] :
         pforr::split_input_range({begin, end}, n_threads, 1)) {
      uint64_t comparisons = 0;
      for (auto row = first; row < last; ++row) {
        const uint64_t degree = graph.n_nbrs(row);
        if (degree > 1) {
          comparisons += degree * (degree - 1) / 2;
        }
      }
      ranges.push_back({first, last, comparisons});
    }
  }
  std::stable_sort(ranges.begin(), ranges.end(),
                   [](const Range &a, const Range &b) {
                     return a.comparisons > b.comparisons;
                   });

  tdoann::SparseNNGraph<Out, Idx> result(graph.row_ptr, graph.col_idx,
                                         graph.dist);
  rnndescent::ParallelRNGAdapter<rnndescent::PcgRand> random;
  random.initialize();
  std::atomic<std::size_t> next{0};
  std::atomic<unsigned> completed{0};
  std::atomic<bool> cancelled{false};
  const auto stop = [&] { return cancelled.load(std::memory_order_relaxed); };
  auto worker = [&](std::size_t, std::size_t) {
    try {
      while (!stop()) {
        const auto i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= ranges.size()) {
          return;
        }
        const auto &range = ranges[i];
        auto rng = random.get_parallel_instance(range.end);
        tdoann::remove_long_edges_impl(graph, distance, *rng, prune_probability,
                                       result, range.begin, range.end, stop);
        if (verbose) {
          completed.fetch_add(1, std::memory_order_relaxed);
        }
      }
    } catch (...) {
      cancelled.store(true, std::memory_order_relaxed);
      throw;
    }
  };

  RParallelExecutor executor;
  RInterruptableProgress interrupt;
  tdoann::ProgressBar progress(ranges.size(), verbose, Rcpp::Rcerr);
  // The future joins on scope exit. Only the calling thread checks R.
  auto work = std::async(std::launch::async, [&] {
    executor.parallel_for(0, n_threads, worker, n_threads, 1);
  });
  try {
    while (work.wait_for(std::chrono::milliseconds(10)) !=
           std::future_status::ready) {
      interrupt.check_interrupt();
      if (verbose) {
        progress.update(completed.load(std::memory_order_relaxed));
      }
    }
    work.get();
    interrupt.check_interrupt();
  } catch (...) {
    cancelled.store(true, std::memory_order_relaxed);
    throw;
  }
  return result;
}

} // namespace

template <typename Out, typename Idx>
List diversify_impl(const tdoann::BaseDistance<Out, Idx> &distance,
                    List graph_list, double prune_probability,
                    std::size_t n_threads, bool verbose,
                    bool balance_work = false) {
  const auto graph = r_to_sparse_graph<Out, Idx>(graph_list);
  if (balance_work && n_threads > 1 &&
      graph.n_points > tdoann::diversification_rows_per_worker * n_threads) {
    return sparse_graph_to_r(diversify_by_degree(
        graph, distance, prune_probability, n_threads, verbose));
  }

  std::optional<tdoann::SparseNNGraph<Out, Idx>> diversified;
  if (n_threads == 0) {
    rnndescent::RRand rand;
    diversified =
        tdoann::remove_long_edges(graph, distance, rand, prune_probability);
  } else {
    RParallelExecutor executor;
    RPProgress progress(1, verbose);
    rnndescent::ParallelRNGAdapter<rnndescent::PcgRand> parallel_rand;
    diversified = tdoann::remove_long_edges(graph, distance, parallel_rand,
                                            prune_probability, n_threads,
                                            progress, executor);
  }
  return sparse_graph_to_r(*diversified);
}

// [[Rcpp::export]]
List rnn_sparse_diversify(const IntegerVector &ind, const IntegerVector &ptr,
                          const NumericVector &data, std::size_t ndim,
                          const List &graph_list, const std::string &metric,
                          double prune_probability, std::size_t n_threads,
                          bool verbose, bool balance_work = false) {
  auto distance_ptr = create_sparse_self_distance(ind, ptr, data, ndim, metric);
  return diversify_impl(*distance_ptr, graph_list, prune_probability, n_threads,
                        verbose, balance_work);
}

// [[Rcpp::export]]
List rnn_diversify(const NumericMatrix &data, const List &graph_list,
                   const std::string &metric, double prune_probability,
                   std::size_t n_threads, bool verbose) {
  auto distance_ptr = create_self_distance(data, metric);
  return diversify_impl(*distance_ptr, graph_list, prune_probability, n_threads,
                        verbose);
}

// [[Rcpp::export]]
List rnn_logical_diversify(const LogicalMatrix &data, const List &graph_list,
                           const std::string &metric, double prune_probability,
                           std::size_t n_threads, bool verbose) {
  auto distance_ptr = create_self_distance(data, metric);
  return diversify_impl(*distance_ptr, graph_list, prune_probability, n_threads,
                        verbose);
}

// [[Rcpp::export]]
List rnn_merge_graph_lists(const List &graph_list1, const List &graph_list2) {
  auto graph1 = r_to_sparse_graph(graph_list1);
  auto graph2 = r_to_sparse_graph(graph_list2);

  auto graph_merged = tdoann::merge_graphs(graph1, graph2);

  return sparse_graph_to_r(graph_merged);
}

// [[Rcpp::export]]
List rnn_degree_prune(const List &graph_list, std::size_t max_degree,
                      std::size_t n_threads) {
  auto graph = r_to_sparse_graph(graph_list);

  RParallelExecutor executor;
  RPProgress progress(1, false);
  auto pruned =
      tdoann::degree_prune(graph, max_degree, n_threads, progress, executor);
  return sparse_graph_to_r(pruned);
}

// NOLINTEND(modernize-use-trailing-return-type)

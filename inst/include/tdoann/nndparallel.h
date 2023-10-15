// BSD 2-Clause License
//
// Copyright 2020 James Melville
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
// OF SUCH DAMAGE.

#ifndef TDOANN_NNDPARALLEL_H
#define TDOANN_NNDPARALLEL_H

#include <array>
#include <mutex>
#include <sstream>

#include "heap.h"
#include "nndcommon.h"
#include "parallel.h"
#include "random.h"

namespace tdoann {

template <typename Distance> class ParallelLocalJoin {
public:
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  using Update = std::tuple<Idx, Idx, DistOut>;

  virtual ~ParallelLocalJoin() = default;

  virtual void generate(const NNDHeap<DistOut, Idx> &current_graph, Idx idx_p,
                        Idx idx_q, std::size_t key) = 0;
  virtual auto apply(NNDHeap<DistOut, Idx> &current_graph) -> std::size_t = 0;

  auto execute(const NNDHeap<DistOut, Idx> &current_graph,
               const NNHeap<DistOut, Idx> &new_nbrs,
               decltype(new_nbrs) &old_nbrs, std::size_t max_candidates,
               std::size_t begin, std::size_t end) {
    for (std::size_t i = begin, idx_offset = begin * max_candidates; i < end;
         i++, idx_offset += max_candidates) {
      for (std::size_t j = 0; j < max_candidates; j++) {
        std::size_t item_p = new_nbrs.idx[idx_offset + j];
        if (item_p == new_nbrs.npos()) {
          continue;
        }
        for (std::size_t k = j; k < max_candidates; k++) {
          std::size_t item_new = new_nbrs.idx[idx_offset + k];
          if (item_new == new_nbrs.npos()) {
            continue;
          }
          this->generate(current_graph, item_p, item_new, i);
        }
        for (std::size_t k = 0; k < max_candidates; k++) {
          std::size_t item_old = old_nbrs.idx[idx_offset + k];
          if (item_old == old_nbrs.npos()) {
            continue;
          }
          this->generate(current_graph, item_p, item_old, i);
        }
      }
    }
  }

  auto execute(NNDHeap<DistOut, Idx> &current_graph,
               const NNHeap<DistOut, Idx> &new_nbrs,
               decltype(new_nbrs) &old_nbrs, NNDProgressBase &progress,
               std::size_t n_threads, Executor &executor) -> std::size_t {
    std::size_t num_updated = 0;
    auto local_join_worker = [&](std::size_t begin, std::size_t end) {
      this->execute(current_graph, new_nbrs, old_nbrs, new_nbrs.n_nbrs, begin,
                    end);
    };
    auto after_local_join = [&](std::size_t, std::size_t) {
      num_updated += this->apply(current_graph);
    };
    ExecutionParams exec_params{16384};
    batch_parallel_for(local_join_worker, after_local_join,
                       current_graph.n_points, n_threads, exec_params,
                       progress.get_base_progress(), executor);
    return num_updated;
  }
};

template <typename Distance>
class LowMemParallelLocalJoin : public ParallelLocalJoin<Distance> {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  using EdgeUpdate = std::tuple<Idx, Idx, DistOut>;

public:
  const Distance &distance;
  std::vector<std::vector<EdgeUpdate>> edge_updates;

  LowMemParallelLocalJoin(const Distance &distance)
      : distance(distance), edge_updates(distance.ny) {}

  void generate(const NNDHeap<DistOut, Idx> &current_graph, Idx idx_p,
                Idx idx_q, std::size_t key) override {
    const auto dist_pq = distance(idx_p, idx_q);
    if (current_graph.accepts_either(idx_p, idx_q, dist_pq)) {
      edge_updates[key].emplace_back(idx_p, idx_q, dist_pq);
    }
  }

  std::size_t apply(NNDHeap<DistOut, Idx> &current_graph) override {
    std::size_t num_updates = 0;
    for (auto &edge_set : edge_updates) {
      for (auto &[idx_p, idx_q, dist_pq] : edge_set) {
        num_updates += current_graph.checked_push_pair(idx_p, dist_pq, idx_q);
      }
      edge_set.clear();
    }
    return num_updates;
  }
};

template <typename Distance>
class CacheParallelLocalJoin : public ParallelLocalJoin<Distance> {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;
  using EdgeUpdate = std::tuple<Idx, Idx, DistOut>;

public:
  const Distance &distance;
  EdgeCache<Idx> cache;
  std::vector<std::vector<EdgeUpdate>> edge_updates;

  CacheParallelLocalJoin(const NNDHeap<DistOut, Idx> &current_graph,
                         const Distance &distance)
      : distance(distance), cache(EdgeCache<Idx>::from_graph(current_graph)),
        edge_updates(current_graph.n_points) {}

  void generate(const NNDHeap<DistOut, Idx> &current_graph, Idx idx_p,
                Idx idx_q, std::size_t key) override {
    auto [idx_pp, idx_qq] = std::minmax(idx_p, idx_q);

    if (cache.contains(idx_pp, idx_qq)) {
      return;
    }

    const auto dist_pq = distance(idx_pp, idx_qq);
    if (current_graph.accepts_either(idx_pp, idx_qq, dist_pq)) {
      edge_updates[key].emplace_back(idx_pp, idx_qq, dist_pq);
    }
  }

  std::size_t apply(NNDHeap<DistOut, Idx> &current_graph) override {
    std::size_t num_updates = 0;

    for (auto &edge_set : edge_updates) {
      for (const auto &[idx_p, idx_q, dist_pq] : edge_set) {

        if (cache.contains(idx_p, idx_q)) {
          continue;
        }

        const bool bad_pd = !current_graph.accepts(idx_p, dist_pq);
        const bool bad_qd = !current_graph.accepts(idx_q, dist_pq);

        if (bad_pd && bad_qd) {
          continue;
        }

        std::size_t local_c = 0;
        if (!bad_pd) {
          current_graph.unchecked_push(idx_p, dist_pq, idx_q);
          local_c++;
        }

        if (idx_p != idx_q && !bad_qd) {
          current_graph.unchecked_push(idx_q, dist_pq, idx_p);
          local_c++;
        }

        if (local_c > 0) {
          cache.insert(idx_p, idx_q);
          num_updates += local_c;
        }
      }
      edge_set.clear();
    }
    return num_updates;
  }
};

template <typename Distance> class LockingHeapAdder {
private:
  using Idx = typename Distance::Index;
  using Out = typename Distance::Output;

  static const constexpr std::size_t n_mutexes = 10;
  std::array<std::mutex, n_mutexes> mutexes;

public:
  LockingHeapAdder() = default;
  LockingHeapAdder(LockingHeapAdder const &) = delete;
  auto operator=(LockingHeapAdder const &) -> LockingHeapAdder & = delete;
  LockingHeapAdder(LockingHeapAdder &&) = delete;
  auto operator=(LockingHeapAdder &&) -> LockingHeapAdder & = delete;
  ~LockingHeapAdder() = default;

  void add(NNHeap<Out, Idx> &nbrs, Idx item_i, Idx item_j, Out dist_ij) {
    {
      std::lock_guard<std::mutex> guard(mutexes[item_i % n_mutexes]);
      nbrs.checked_push(item_i, dist_ij, item_j);
    }
    if (item_i != item_j) {
      std::lock_guard<std::mutex> guard(mutexes[item_j % n_mutexes]);
      nbrs.checked_push(item_j, dist_ij, item_i);
    }
  }
  void add(NNHeap<Out, Idx> &nbrs, Idx item_i, Idx item_j, Out dist_ij,
           Out dist_ji) {
    {
      std::lock_guard<std::mutex> guard(mutexes[item_i % n_mutexes]);
      nbrs.checked_push(item_i, dist_ij, item_j);
    }
    if (item_i != item_j) {
      std::lock_guard<std::mutex> guard(mutexes[item_j % n_mutexes]);
      nbrs.checked_push(item_j, dist_ji, item_i);
    }
  }
};

template <typename Distance>
void build_candidates(
    const NNDHeap<typename Distance::Output, typename Distance::Index>
        &current_graph,
    NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    NNHeap<typename Distance::Output, typename Distance::Index> &old_nbrs,
    ParallelRandomProvider &parallel_rand,
    LockingHeapAdder<Distance> &heap_adder, std::size_t begin,
    std::size_t end) {

  const std::size_t n_nbrs = current_graph.n_nbrs;
  auto rand = parallel_rand.get_parallel_instance(end);

  for (std::size_t i = begin, idx_offset = begin * n_nbrs; i < end;
       i++, idx_offset += n_nbrs) {
    for (std::size_t j = 0, idx_ij = idx_offset; j < n_nbrs; j++, idx_ij++) {
      auto nbr = current_graph.idx[idx_ij];
      uint8_t isn = current_graph.flags[idx_ij];
      auto &nbrs = isn == 1 ? new_nbrs : old_nbrs;
      if (nbr == nbrs.npos()) {
        continue;
      }
      auto rand_weight = rand->unif();
      heap_adder.add(nbrs, i, nbr, rand_weight);
    }
  }
}

template <typename Distance>
void build_candidates(
    const NNDHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    NNHeap<typename Distance::Output, typename Distance::Index> &old_nbrs,
    ParallelRandomProvider &parallel_rand,
    LockingHeapAdder<Distance> &heap_adder, std::size_t n_threads,
    Executor &executor) {

  parallel_rand.initialize();
  auto worker = [&](std::size_t begin, std::size_t end) {
    build_candidates(nn_heap, new_nbrs, old_nbrs, parallel_rand, heap_adder,
                     begin, end);
  };
  batch_parallel_for(worker, nn_heap.n_points, n_threads, executor);
}

template <typename Distance>
void flag_new_candidates(
    NNDHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    const NNHeap<typename Distance::Output, typename Distance::Index> &new_nbrs,
    std::size_t n_threads, Executor &executor) {
  auto worker = [&](std::size_t begin, std::size_t end) {
    flag_retained_new_candidates(nn_heap, new_nbrs, begin, end);
  };
  batch_parallel_for(worker, nn_heap.n_points, n_threads, executor);
}

template <typename Distance>
void nnd_build(
    NNDHeap<typename Distance::Output, typename Distance::Index> &nn_heap,
    ParallelLocalJoin<Distance> &local_join, std::size_t max_candidates,
    std::size_t n_iters, double delta, NNDProgressBase &progress,
    ParallelRandomProvider &parallel_rand, std::size_t n_threads,
    Executor &executor) {
  using DistOut = typename Distance::Output;
  using Idx = typename Distance::Index;

  const std::size_t n_points = nn_heap.n_points;
  const double tol = delta * nn_heap.n_nbrs * n_points;

  LockingHeapAdder<Distance> heap_adder;

  for (std::size_t iter = 0; iter < n_iters; iter++) {
    NNHeap<DistOut, Idx> new_nbrs(n_points, max_candidates);
    decltype(new_nbrs) old_nbrs(n_points, max_candidates);

    build_candidates<Distance>(nn_heap, new_nbrs, old_nbrs, parallel_rand,
                               heap_adder, n_threads, executor);

    flag_new_candidates<Distance>(nn_heap, new_nbrs, n_threads, executor);

    std::size_t num_updated = local_join.execute(nn_heap, new_nbrs, old_nbrs,
                                                 progress, n_threads, executor);

    if (progress.check_interrupt()) {
      break;
    }
    progress.iter_finished();

    bool stop_early = nnd_should_stop(progress, nn_heap, num_updated, tol);
    if (stop_early) {
      break;
    }
  }
}

} // namespace tdoann
#endif // TDOANN_NNDPARALLEL_H

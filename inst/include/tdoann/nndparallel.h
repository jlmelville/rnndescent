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

#include "distancebase.h"
#include "heap.h"
#include "nndcommon.h"
#include "parallel.h"
#include "random.h"

namespace tdoann {

template <typename Out, typename Idx> class ParallelLocalJoin {
  static constexpr auto npos = static_cast<Idx>(-1);

public:
  using Update = std::tuple<Idx, Idx, Out>;

  virtual ~ParallelLocalJoin() = default;

  virtual void generate(const NNDHeap<Out, Idx> &current_graph, Idx idx_p,
                        Idx idx_q, std::size_t key) = 0;
  virtual auto apply(NNDHeap<Out, Idx> &current_graph) -> unsigned long = 0;

  auto execute(const NNDHeap<Out, Idx> &current_graph,
               const NNHeap<Out, Idx> &new_nbrs, decltype(new_nbrs) &old_nbrs,
               std::size_t max_candidates, std::size_t begin, std::size_t end) {
    for (auto i = begin, i_begin = begin * max_candidates; i < end;
         i++, i_begin += max_candidates) {
      for (std::size_t j = 0; j < max_candidates; j++) {
        auto new_j = new_nbrs.idx[i_begin + j];
        if (new_j == npos) {
          continue;
        }

        // (new, new) pairs: loop from j->max_candidates
        for (auto k = j; k < max_candidates; k++) {
          auto new_k = new_nbrs.idx[i_begin + k];
          if (new_k == npos) {
            continue;
          }
          this->generate(current_graph, new_j, new_k, i);
        }

        // (new, old) pairs loop from 0->max_candidates
        for (std::size_t k = 0; k < max_candidates; k++) {
          auto old_k = old_nbrs.idx[i_begin + k];
          if (old_k == npos) {
            continue;
          }
          this->generate(current_graph, new_j, old_k, i);
        }
      }
    }
  }

  auto execute(NNDHeap<Out, Idx> &current_graph,
               const NNHeap<Out, Idx> &new_nbrs, decltype(new_nbrs) &old_nbrs,
               NNDProgressBase &progress, std::size_t n_threads,
               const Executor &executor) -> std::size_t {
    std::size_t num_updates = 0;
    auto local_join_worker = [&](std::size_t begin, std::size_t end) {
      this->execute(current_graph, new_nbrs, old_nbrs, new_nbrs.n_nbrs, begin,
                    end);
    };
    auto after_local_join = [&](std::size_t, std::size_t) {
      num_updates += this->apply(current_graph);
    };
    ExecutionParams exec_params{16384};
    dispatch_work(local_join_worker, after_local_join, current_graph.n_points,
                  n_threads, exec_params, progress.get_base_progress(),
                  executor);
    return num_updates;
  }
};

template <typename Out, typename Idx>
class LowMemParallelLocalJoin : public ParallelLocalJoin<Out, Idx> {
  using EdgeUpdate = std::tuple<Idx, Idx, Out>;

public:
  const BaseDistance<Out, Idx> &distance;
  std::vector<std::vector<EdgeUpdate>> edge_updates;

  LowMemParallelLocalJoin(const BaseDistance<Out, Idx> &distance)
      : distance(distance), edge_updates(distance.get_ny()) {}

  void generate(const NNDHeap<Out, Idx> &current_graph, Idx p, Idx q,
                std::size_t key) override {
    const auto d_pq = distance.calculate(p, q);
    if (current_graph.accepts_either(p, q, d_pq)) {
      edge_updates[key].emplace_back(p, q, d_pq);
    }
  }

  unsigned long apply(NNDHeap<Out, Idx> &current_graph) override {
    unsigned long num_updates = 0UL;
    for (auto &edge_set : edge_updates) {
      for (auto &[p, q, d_pq] : edge_set) {
        num_updates += current_graph.checked_push_pair(p, d_pq, q);
      }
      edge_set.clear();
    }
    return num_updates;
  }
};

template <typename Out, typename Idx>
class CacheParallelLocalJoin : public ParallelLocalJoin<Out, Idx> {
  using EdgeUpdate = std::tuple<Idx, Idx, Out>;

public:
  const BaseDistance<Out, Idx> &distance;
  EdgeCache<Idx> cache;
  std::vector<std::vector<EdgeUpdate>> edge_updates;

  CacheParallelLocalJoin(const NNDHeap<Out, Idx> &current_graph,
                         const BaseDistance<Out, Idx> &distance)
      : distance(distance), cache(EdgeCache<Idx>::from_graph(current_graph)),
        edge_updates(current_graph.n_points) {}

  void generate(const NNDHeap<Out, Idx> &current_graph, Idx idx_p, Idx idx_q,
                std::size_t key) override {
    auto [idx_pp, idx_qq] = std::minmax(idx_p, idx_q);

    if (cache.contains(idx_pp, idx_qq)) {
      return;
    }

    const auto dist_pq = distance.calculate(idx_pp, idx_qq);
    if (current_graph.accepts_either(idx_pp, idx_qq, dist_pq)) {
      edge_updates[key].emplace_back(idx_pp, idx_qq, dist_pq);
    }
  }

  unsigned long apply(NNDHeap<Out, Idx> &current_graph) override {
    unsigned long num_updates = 0;
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

        uint32_t local_c = 0;
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

template <typename Out, typename Idx> class LockingHeapAdder {
private:
  static constexpr std::size_t n_mutexes = 10;
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

  void add(NNHeap<Out, Idx> &nbrs, Idx item_i, Idx item_j, Out weight_i,
           Out weight_j) {
    {
      std::lock_guard<std::mutex> guard(mutexes[item_i % n_mutexes]);
      nbrs.checked_push(item_i, weight_j, item_j);
    }
    if (item_i != item_j) {
      std::lock_guard<std::mutex> guard(mutexes[item_j % n_mutexes]);
      nbrs.checked_push(item_j, weight_i, item_i);
    }
  }
};

template <typename Out, typename Idx>
void build_candidates(const NNDHeap<Out, Idx> &current_graph,
                      NNHeap<Out, Idx> &new_nbrs, decltype(new_nbrs) &old_nbrs,
                      bool weight_by_degree,
                      ParallelRandomProvider &parallel_rand,
                      std::size_t n_threads, const Executor &executor) {
  constexpr auto npos = static_cast<Idx>(-1);
  const std::size_t n_nbrs = current_graph.n_nbrs;
  LockingHeapAdder<Out, Idx> heap_adder;

  auto k_occurrences = weight_by_degree ? count_reverse_neighbors(current_graph)
                                        : std::vector<std::size_t>();

  parallel_rand.initialize();
  auto worker = [&](std::size_t begin, std::size_t end) {
    auto rand = parallel_rand.get_parallel_instance(end);

    for (auto i = begin, idx_offset = begin * n_nbrs; i < end;
         i++, idx_offset += n_nbrs) {
      for (auto idx_ij = idx_offset; idx_ij < idx_offset + n_nbrs; idx_ij++) {
        const auto nbr = current_graph.idx[idx_ij];
        if (nbr == npos) {
          continue;
        }
        auto &nbrs = current_graph.flags[idx_ij] == 1 ? new_nbrs : old_nbrs;
        auto rand_weight = rand->unif();
        if (weight_by_degree) {
          heap_adder.add(nbrs, i, nbr, rand_weight * k_occurrences[i],
                         rand_weight * k_occurrences[nbr]);
        } else {
          heap_adder.add(nbrs, i, nbr, rand_weight);
        }
      }
    }
  };
  dispatch_work(worker, current_graph.n_points, n_threads, executor);
}

template <typename Out, typename Idx>
void flag_new_candidates(NNDHeap<Out, Idx> &current_graph,
                         const NNHeap<Out, Idx> &new_nbrs,
                         std::size_t n_threads, const Executor &executor) {
  auto worker = [&](std::size_t begin, std::size_t end) {
    flag_retained_new_candidates(current_graph, new_nbrs, begin, end);
  };
  dispatch_work(worker, current_graph.n_points, n_threads, executor);
}

template <typename Out, typename Idx>
void nnd_build(NNDHeap<Out, Idx> &current_graph,
               ParallelLocalJoin<Out, Idx> &local_join,
               std::size_t max_candidates, uint32_t n_iters, double delta,
               bool weight_by_degree, NNDProgressBase &progress,
               ParallelRandomProvider &parallel_rand, std::size_t n_threads,
               const Executor &executor) {
  const std::size_t n_points = current_graph.n_points;

  for (auto iter = 0U; iter < n_iters; iter++) {
    NNHeap<Out, Idx> new_nbrs(n_points, max_candidates);
    decltype(new_nbrs) old_nbrs(n_points, max_candidates);

    build_candidates(current_graph, new_nbrs, old_nbrs, weight_by_degree,
                     parallel_rand, n_threads, executor);

    flag_new_candidates(current_graph, new_nbrs, n_threads, executor);

    auto num_updates = local_join.execute(current_graph, new_nbrs, old_nbrs,
                                          progress, n_threads, executor);

    if (nnd_should_stop(progress, current_graph, num_updates, delta)) {
      break;
    }
  }
}

} // namespace tdoann
#endif // TDOANN_NNDPARALLEL_H

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

#ifndef RNND_PARALLEL_H
#define RNND_PARALLEL_H
#include <unordered_set>

#include <Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include "heap.h"
#include "rrand.h"
#include "nndescent.h"

struct LockingCandidatesWorker : public RcppParallel::Worker {
  const NeighborHeap& current_graph;
  const std::size_t n_points;
  const std::size_t n_nbrs;
  const std::size_t max_candidates;
  NeighborHeap new_candidate_neighbors;
  NeighborHeap old_candidate_neighbors;
  tthread::mutex mutex;

  LockingCandidatesWorker(
    const NeighborHeap& current_graph,
    const std::size_t max_candidates
  ) :
    current_graph(current_graph),
    n_points(current_graph.n_points),
    n_nbrs(current_graph.n_nbrs),
    max_candidates(max_candidates),
    new_candidate_neighbors(n_points, max_candidates),
    old_candidate_neighbors(n_points, max_candidates)
  {}

  void operator()(std::size_t begin, std::size_t end) {
    std::unique_ptr<TauRand> rand(nullptr);
    // Each window gets its own PRNG state, to prevent locking inside the loop.
    {
      tthread::lock_guard<tthread::mutex> guard(mutex);
      rand.reset(new TauRand());
    }

    for (std::size_t i = begin; i < end; i++) {
      std::size_t innbrs = i * n_nbrs;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        std::size_t ij = innbrs + j;
        std::size_t idx = current_graph.idx[ij];
        double d = rand->unif();
        bool isn = current_graph.flags[ij] == 1;
        if (isn) {
          {
            tthread::lock_guard<tthread::mutex> guard(mutex);
            new_candidate_neighbors.checked_push(i, d, idx, isn);
            new_candidate_neighbors.checked_push(idx, d, i, isn);
          }
        }
        else {
          {
            tthread::lock_guard<tthread::mutex> guard(mutex);
            old_candidate_neighbors.checked_push(i, d, idx, isn);
            old_candidate_neighbors.checked_push(idx, d, i, isn);
          }
        }
      }
    }
  }
};

// mark any neighbor in the current graph that was retained in the new
// candidates as true
struct NewCandidatesWorker : public RcppParallel::Worker {
  const NeighborHeap& new_candidate_neighbors;
  NeighborHeap& current_graph;
  const std::size_t n_points;
  const std::size_t n_nbrs;
  const std::size_t max_candidates;

  NewCandidatesWorker(
    const NeighborHeap& new_candidate_neighbors,
    NeighborHeap& current_graph
  ) :
    new_candidate_neighbors(new_candidate_neighbors),
    current_graph(current_graph),
    n_points(current_graph.n_points),
    n_nbrs(current_graph.n_nbrs),
    max_candidates(new_candidate_neighbors.n_nbrs)
  {}

  void operator()(std::size_t begin, std::size_t end)
  {
    for (std::size_t i = begin; i < end; i++) {
      std::size_t innbrs = i * n_nbrs;
      std::size_t innbrs_new = i * max_candidates;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        std::size_t ij = innbrs + j;
        std::size_t idx = current_graph.idx[ij];
        for (std::size_t k = 0; k < max_candidates; k++) {
          if (new_candidate_neighbors.idx[innbrs_new + k] == idx) {
            current_graph.flags[ij] = 1;
            break;
          }
        }
      }
    }
  }
};

struct Update {
  const std::size_t p;
  const std::size_t q;
  const double d;

  Update() :
    p(0),
    q(0),
    d(0) {}

  Update(
    const std::size_t p,
    const std::size_t q,
    const double d
  ) :
    p(p),
    q(q),
    d(d)
  {}

  Update(Update&&) = default;
};

template <typename Distance,
          typename GraphUpdaterT>
struct LocalJoinWorker : public RcppParallel::Worker {
  const NeighborHeap& current_graph;
  const NeighborHeap& new_nbrs;
  const NeighborHeap& old_nbrs;
  const Distance& distance;
  const std::size_t n_nbrs;
  const std::size_t max_candidates;
  const GraphUpdaterT& graph_updater;
  std::vector<std::vector<Update>>& updates;

  LocalJoinWorker(
    const NeighborHeap& current_graph,
    const NeighborHeap& new_nbrs,
    const NeighborHeap& old_nbrs,
    const Distance& distance,
    const GraphUpdaterT& graph_updater,
    std::vector<std::vector<Update>>& updates
  ) :
    current_graph(current_graph),
    new_nbrs(new_nbrs),
    old_nbrs(old_nbrs),
    distance(distance),
    n_nbrs(current_graph.n_nbrs),
    max_candidates(new_nbrs.n_nbrs),
    graph_updater(graph_updater),
    updates(updates)
  {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      const std::size_t imaxc = i * max_candidates;
      for (std::size_t j = 0; j < max_candidates; j++) {
        std::size_t p = new_nbrs.idx[imaxc + j];
        if (p == NeighborHeap::npos()) {
          continue;
        }
        const std::size_t pnnbrs = p * n_nbrs;
        for (std::size_t k = j; k < max_candidates; k++) {
          std::size_t q = new_nbrs.idx[imaxc + k];
          if (q == NeighborHeap::npos()) {
            continue;
          }
          graph_updater.generate(current_graph, distance, i, p, q,
                                 n_nbrs, pnnbrs, updates);
        }

        for (std::size_t k = 0; k < max_candidates; k++) {
          std::size_t q = old_nbrs.idx[imaxc + k];
          if (q == NeighborHeap::npos()) {
            continue;
          }
          graph_updater.generate(current_graph, distance, i, p, q, n_nbrs,
                                 pnnbrs, updates);
        }
      }
    }
  }
};


struct GraphCache {
  std::vector<std::unordered_set<std::size_t>> seen;

  GraphCache(const NeighborHeap& neighbor_heap) :
    seen(neighbor_heap.n_points)
  {
    const std::size_t n_points = neighbor_heap.n_points;
    const std::size_t n_nbrs = neighbor_heap.n_nbrs;
    for (std::size_t i = 0; i < n_points; i++) {
      const std::size_t innbrs = i * n_nbrs;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        std::size_t p = neighbor_heap.idx[innbrs + j];
        if (i > p) {
          seen[p].emplace(i);
        }
        else {
          seen[i].emplace(p);
        }
      }
    }
  }

  bool contains(const std::size_t& p,
                const std::size_t& q) const
  {
    return seen[p].find(q) != seen[p].end();
  }

  bool insert(std::size_t p, std::size_t q)
  {
    return !seen[p].emplace(q).second;
  }
};

struct GraphUpdater {
  // Purposely do nothing with the neighbors
  GraphUpdater(const NeighborHeap&) {}

  template<typename Distance>
  void generate(
      const NeighborHeap& current_graph,
      const Distance& distance,
      const std::size_t i,
      const std::size_t p,
      const std::size_t q,
      const std::size_t n_nbrs,
      const std::size_t pnnbrs,
      std::vector<std::vector<Update>>& updates
  ) const
  {
    double d = distance(p, q);
    if (d < current_graph.dist[pnnbrs] || d < current_graph.dist[q * n_nbrs]) {
      updates[i].emplace_back(p, q, d);
    }
  }

  size_t apply(
      NeighborHeap& current_graph,
      std::vector<std::vector<Update>>& updates
  )
  {
    std::size_t c = 0;
    const std::size_t n_points = updates.size();
    for (std::size_t i = 0; i < n_points; i++) {
      const std::size_t n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        const auto& update = updates[i][j];
        c += current_graph.checked_push(update.p, update.d, update.q, true);
        if (update.p != update.q) {
          c += current_graph.checked_push(update.q, update.d, update.p, true);
        }
      }
      updates[i].clear();
    }
    return c;
  }
};


struct GraphUpdaterHiMem {
  GraphCache seen;

  GraphUpdaterHiMem(
    const NeighborHeap& neighbor_heap
  ) :
    seen(neighbor_heap)
  {}

  template<typename Distance>
  void generate(
      const NeighborHeap& current_graph,
      const Distance& distance,
      const std::size_t i,
      const std::size_t p,
      const std::size_t q,
      const std::size_t n_nbrs,
      const std::size_t pnnbrs,
      std::vector<std::vector<Update>>& updates
    ) const
  {
    // canonicalize the order of (p, q) so that qq >= pp
    std::size_t pp = p > q ? q : p;
    std::size_t qq = pp == p ? q : p;

    if (seen.contains(p, q)) {
      return;
    }
    double d = distance(p, q);
    if (d < current_graph.dist[pnnbrs] || d < current_graph.dist[q * n_nbrs]) {
      updates[i].emplace_back(pp, qq, d);
    }
  }

  size_t apply(
      NeighborHeap& current_graph,
      std::vector<std::vector<Update>>& updates
  )
  {
    std::size_t c = 0;
    const std::size_t n_points = updates.size();
    for (std::size_t i = 0; i < n_points; i++) {
      const std::size_t n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        const auto& update = updates[i][j];
        const auto& p = update.p;
        const auto& q = update.q;
        const auto& d = update.d;
        if (seen.insert(p, q)) {
          continue;
        }

        if (d < current_graph.distance(p, 0)) {
          current_graph.unchecked_push(p, d, q, true);
          c += 1;
        }

        if (p != q && d < current_graph.distance(q, 0)) {
          current_graph.unchecked_push(q, d, p, true);
          c += 1;
        }
      }
      updates[i].clear();
    }
    return c;
  }
};

struct UpdateWorker : RcppParallel::Worker {
  NeighborHeap& current_graph;
  std::vector<std::vector<Update>>& updates;
  std::size_t n_updates;
  tthread::mutex mutex;

  UpdateWorker(
    NeighborHeap& current_graph,
    std::vector<std::vector<Update>>& updates
  ) :
    current_graph(current_graph),
    updates(updates),
    n_updates(0)
  {}

  void operator()(std::size_t begin, std::size_t end) {
    std::size_t c = 0;
    for (std::size_t i = begin; i < end; i++) {
      const std::size_t n_updates = updates[i].size();
      for (std::size_t j = 0; j < n_updates; j++) {
        const auto& update = updates[i][j];
        {
          tthread::lock_guard<tthread::mutex> guard(mutex);
          c += current_graph.checked_push(update.p, update.d, update.q, true);
          if (update.p != update.q) {
            c += current_graph.checked_push(update.q, update.d, update.p, true);
          }
        }
      }
    }
    {
      tthread::lock_guard<tthread::mutex> guard(mutex);
      n_updates += c;
    }
  }
};

template <template<typename> class Heap,
          typename Distance,
          typename Rand,
          typename Progress,
          typename GraphUpdaterT>
void nnd_parallel(
    Heap<Distance>& current_graph,
    const std::size_t max_candidates,
    const std::size_t n_iters,
    GraphUpdaterT& graph_updater,
    Rand& rand,
    Progress& progress,
    const double tol,
    std::size_t grain_size = 1,
    const std::size_t block_size = 16384,
    bool verbose = false
)
{
  const std::size_t n_points = current_graph.neighbor_heap.n_points;
  const auto n_blocks = (n_points / block_size) + 1;
  std::vector<std::vector<Update>> updates(n_points);

  for (std::size_t n = 0; n < n_iters; n++) {
    LockingCandidatesWorker candidates_worker(current_graph.neighbor_heap,
                                              max_candidates);
    RcppParallel::parallelFor(0, n_points, candidates_worker, grain_size);
    auto& new_candidate_neighbors = candidates_worker.new_candidate_neighbors;
    auto& old_candidate_neighbors = candidates_worker.old_candidate_neighbors;

    NewCandidatesWorker new_candidates_worker(
        new_candidate_neighbors,
        current_graph.neighbor_heap);
    RcppParallel::parallelFor(0, n_points, new_candidates_worker, grain_size);

    std::size_t c = 0;
    for (std::size_t i = 0; i < n_blocks; i++) {
      const auto block_start = i * block_size;
      const auto block_end = std::min<std::size_t>(n_points, (i + 1) * block_size);

      LocalJoinWorker<Distance, GraphUpdaterT> local_join_worker(
          current_graph.neighbor_heap,
          new_candidate_neighbors,
          old_candidate_neighbors,
          current_graph.weight_measure,
          graph_updater,
          updates
      );
      RcppParallel::parallelFor(block_start, block_end, local_join_worker, grain_size);

      c += graph_updater.apply(current_graph.neighbor_heap, updates);

      if (progress.check_interrupt()) {
        break;
      }
    }
    progress.update(n);

    if (static_cast<double>(c) <= tol) {
      if (verbose) {
        Rcpp::Rcout << "c = " << c << " tol = " << tol << std::endl;
      }
      progress.stopping_early();
      break;
    }
  }
  current_graph.neighbor_heap.deheap_sort();
}

#endif // RNND_PARALLEL_H

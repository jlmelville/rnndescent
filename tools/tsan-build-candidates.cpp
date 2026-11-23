#include <tdoann/distancebin.h>
#include <tdoann/nndparallel.h>
#include <tdoann/nngraph.h>

#include <pforr.h>

#include "rnn_candidate_routing.h"
#include "rnn_nnd_route.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

class ThreadExecutor : public tdoann::Executor {
public:
  void parallel_for(std::size_t begin, std::size_t end,
                    std::function<void(std::size_t, std::size_t)> worker,
                    std::size_t n_threads,
                    std::size_t grain_size) const override {
    pforr::parallel_for(begin, end, worker, n_threads, grain_size);
  }
};

class SplitMixRandom : public tdoann::RandomGenerator {
public:
  explicit SplitMixRandom(std::uint64_t state) : state(state) {}

  auto unif() -> double override {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return static_cast<double>(value >> 11U) * 0x1.0p-53;
  }

private:
  std::uint64_t state;
};

class SplitMixProvider : public tdoann::ParallelRandomProvider {
public:
  void initialize() override {}

  auto get_parallel_instance(std::uint64_t chunk_end)
      -> std::unique_ptr<tdoann::RandomGenerator> override {
    return std::make_unique<SplitMixRandom>(seed ^ chunk_end);
  }

private:
  static constexpr std::uint64_t seed = 0x243f6a8885a308d3ULL;
};

template <typename Heap>
auto validate_and_snapshot(const Heap &heap) -> std::vector<
    std::pair<typename Heap::Index, typename Heap::DistanceOut>> {
  using Index = typename Heap::Index;
  using Distance = typename Heap::DistanceOut;
  using Entry = std::pair<Index, Distance>;

  std::vector<Entry> snapshot;
  snapshot.reserve(heap.idx.size());
  for (Index row = 0; row < heap.n_points; ++row) {
    std::unordered_set<Index> seen;
    std::vector<Entry> row_entries;
    row_entries.reserve(heap.n_nbrs);
    for (Index column = 0; column < heap.n_nbrs; ++column) {
      const auto offset = row * heap.n_nbrs + column;
      const auto index = heap.idx[offset];
      if (index >= heap.n_points || !seen.insert(index).second) {
        throw std::runtime_error("candidate heap membership invariant failed");
      }
      if (column > 0) {
        const auto parent = row * heap.n_nbrs + (column - 1U) / 2U;
        if (heap.dist[parent] < heap.dist[offset]) {
          throw std::runtime_error("candidate max-heap invariant failed");
        }
      }
      row_entries.emplace_back(index, heap.dist[offset]);
    }
    std::sort(row_entries.begin(), row_entries.end());
    snapshot.insert(snapshot.end(), row_entries.begin(), row_entries.end());
  }
  return snapshot;
}

// Use the original stream keys but apply proposals in global source order.
class OrderedExecutor : public tdoann::Executor {
public:
  void parallel_for(std::size_t begin, std::size_t end,
                    std::function<void(std::size_t, std::size_t)> worker,
                    std::size_t threads, std::size_t grain) const override {
    for (const auto &range :
         pforr::split_input_range({begin, end}, threads, grain)) {
      worker(range.first, range.second);
    }
  }
};

void check_compact_routing(tdoann::NNDHeap<float> graph) {
  using Heap = tdoann::NNHeap<float>;
  // A hub, self edges and empty slots across uneven source ranges.
  for (std::uint32_t row = 0; row < graph.n_points; ++row) {
    const auto slot = row * graph.n_nbrs;
    graph.idx[slot] = row;
    graph.idx[slot + 1] = 0;
    graph.idx[slot + 2] = graph.npos();
  }
  OrderedExecutor ordered;
  ThreadExecutor threaded;
  SplitMixProvider rng;
  for (auto workers : {2U, 4U, 8U}) {
    rnndescent::CompactCandidates<float, std::uint32_t> routing;
    for (bool weighted : {false, true, false}) {
      Heap fresh(graph.n_points, 7), old(graph.n_points, 7);
      auto expected_fresh = fresh;
      auto expected_old = old;
      tdoann::build_candidates(graph, expected_fresh, expected_old, weighted,
                               rng, workers, ordered);
      routing.build(graph, fresh, old, weighted, rng, workers, threaded);
      if (fresh.idx != expected_fresh.idx ||
          fresh.dist != expected_fresh.dist || old.idx != expected_old.idx ||
          old.dist != expected_old.dist) {
        throw std::runtime_error("compact candidate IDs or priorities differ");
      }
      for (auto &flag : graph.flags) {
        flag ^= 1;
      }
    }
  }
}

void check_binary_updates() {
  using Index = std::uint32_t;
  // Two full routing windows followed by fewer rows than workers. Repeated
  // feature vectors and endpoints make the order of tied insertions visible.
  constexpr Index rows = 4099;
  tdoann::BitVec bits(rows);
  tdoann::NNHeap<float, Index> fresh(rows, 7), old(rows, 7);
  for (Index row = 0; row < rows; ++row) {
    bits[row] = row % 17;
    for (Index j = 0; j < 6; ++j) {
      const auto slot = row * fresh.n_nbrs + j;
      fresh.idx[slot] = (row + j * 113) % rows;
      old.idx[slot] = j < 2 ? 0 : (row + j * 113 + 1) % rows;
    }
  }
  tdoann::BinarySelfDistanceCalculator<float, Index> distance(
      std::move(bits), rows, 8, tdoann::bjaccard<float, Index>);
  ThreadExecutor executor;
  tdoann::NNDProgress progress(std::make_unique<tdoann::NullProgress>());
  for (auto workers : {2U, 4U, 8U}) {
    tdoann::NNDHeap<float, Index> expected(rows, 4);
    auto actual = expected;
    tdoann::LowMemParallelLocalJoin<float, Index> serial(distance);
    rnndescent::BinaryDirectJoin<float, Index> direct(distance);
    for (int round = 0; round < 2; ++round) {
      const auto n =
          serial.execute(expected, fresh, old, progress, workers, executor);
      const auto changes =
          direct.execute(actual, fresh, old, progress, workers, executor);
      if (n != changes || actual.idx != expected.idx ||
          actual.dist != expected.dist || actual.flags != expected.flags) {
        throw std::runtime_error(
            "binary refinement differs from serial application");
      }
      std::swap(fresh, old);
    }
  }
}

void check_leaf_handoff() {
  ThreadExecutor executor;
  tdoann::NullProgress progress;
  using Index = std::uint32_t;
  tdoann::NNHeap<float, Index> leaves(19, 2);
  // Row 4 has only incoming edges. The first two tied sources win.
  for (Index source : {0U, 1U, 2U}) {
    leaves.checked_push(source, 0.5f, 4);
  }
  leaves.checked_push(5, 0.0f, 5);
  leaves.checked_push(17, 1.0f, 6);
  for (Index row = 6; row < 17; ++row) {
    leaves.checked_push(row, 1.0f, 17);
    leaves.checked_push(row, 1.0f, 5);
  }
  auto sorted = leaves;
  tdoann::sort_heap(sorted, 0, progress, executor);
  tdoann::NNDHeap<float, Index> expected(leaves.n_points, leaves.n_nbrs);
  tdoann::vec_to_knn_heap(expected, sorted.idx, sorted.n_points, sorted.dist, 0,
                          false, progress, executor);
  for (auto threads : {0U, 1U, 4U, 8U}) {
    auto input = leaves;
    const auto graph =
        rnndescent::import_leaf_heap(input, threads, progress, executor);
    if (graph.idx != expected.idx || graph.dist != expected.dist ||
        graph.flags != expected.flags || !graph.contains(4, 0) ||
        !graph.contains(4, 1) || graph.contains(4, 2)) {
      throw std::runtime_error(
          "leaf handoff differs from serial symmetric import");
    }
  }
}

int main() {
  using Distance = float;
  using Index = std::uint32_t;
  using Snapshot = std::vector<std::pair<Index, Distance>>;

  constexpr Index n_points = 257;
  constexpr Index n_neighbors = 32;
  constexpr Index max_candidates = 16;
  constexpr std::size_t n_threads = 4;
  constexpr std::size_t repetitions = 50;

  tdoann::NNDHeap<Distance, Index> current_graph(n_points, n_neighbors);
  for (Index row = 0; row < n_points; ++row) {
    for (Index column = 0; column < n_neighbors; ++column) {
      const auto offset = row * n_neighbors + column;
      current_graph.idx[offset] =
          (row + 1U + (17U * column) % (n_points - 1U)) % n_points;
      current_graph.dist[offset] = static_cast<Distance>(column + 1U);
      current_graph.flags[offset] = static_cast<std::uint8_t>(column % 2U);
    }
  }

  check_compact_routing(current_graph);
  check_binary_updates();
  check_leaf_handoff();
  ThreadExecutor executor;
  SplitMixProvider random_provider;
  std::array<Snapshot, 2> expected;
  std::array<bool, 2> has_expected{false, false};

  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    tdoann::NNHeap<Distance, Index> new_neighbors(n_points, max_candidates);
    tdoann::NNHeap<Distance, Index> old_neighbors(n_points, max_candidates);
    const bool weight_by_degree = repetition % 2U == 0U;
    tdoann::build_candidates(current_graph, new_neighbors, old_neighbors,
                             weight_by_degree, random_provider, n_threads,
                             executor);

    auto snapshot = validate_and_snapshot(new_neighbors);
    auto old_snapshot = validate_and_snapshot(old_neighbors);
    snapshot.insert(snapshot.end(), old_snapshot.begin(), old_snapshot.end());
    const auto mode = static_cast<std::size_t>(weight_by_degree);
    if (!has_expected[mode]) {
      expected[mode] = std::move(snapshot);
      has_expected[mode] = true;
    } else if (snapshot != expected[mode]) {
      throw std::runtime_error("candidate construction is not deterministic");
    }
  }

  std::cout << "parallel candidate construction, binary refinement and leaf "
               "handoff: PASS\n";
}

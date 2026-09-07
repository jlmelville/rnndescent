#include <tdoann/nndparallel.h>

#include <pforr.h>

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

  std::cout << "parallel candidate construction: PASS\n";
}

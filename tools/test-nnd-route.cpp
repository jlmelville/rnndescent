#include "pforr.h"
#include "rnn_nnd_route.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <utility>

namespace {
constexpr uint32_t n_points = 16;

class TestDistance : public tdoann::BaseDistance<float> {
public:
  float calculate(const uint32_t &, const uint32_t &) const override {
    return 0.0F;
  }
  std::size_t get_nx() const override { return n_points; }
  std::size_t get_ny() const override { return n_points; }
};

class TestExecutor : public tdoann::Executor {
public:
  void parallel_for(std::size_t begin, std::size_t end,
                    std::function<void(std::size_t, std::size_t)> worker,
                    std::size_t n_threads,
                    std::size_t grain_size) const override {
    pforr::parallel_for(begin, end, worker, n_threads, grain_size);
  }
};
} // namespace

int main() {
  const TestDistance distance;
  const TestExecutor executor;
  rnndescent::RoutedLowMemParallelLocalJoin<float, uint32_t> join(distance);
  tdoann::NNDHeap<float> serial(n_points, 2);
  for (uint32_t i = 0; i < n_points; ++i) {
    serial.checked_push(i, 0.0F, i);
    serial.checked_push(i, 10.0F, (i + 1) % n_points);
  }
  auto routed = serial;

  // These ties compete for one slot on row 11 from different source ranges.
  join.edge_updates[0] = {{1, 11, 3.0F}, {1, 11, 3.0F}};
  join.edge_updates[2] = {{11, 2, 3.0F}, {11, 11, 1.0F}};
  join.edge_updates[4] = {{4, 5, 9.0F}, {4, 8, 10.0F}};
  join.edge_updates[8] = {{11, 3, 2.0F}, {3, 2, 3.0F}};
  join.edge_updates[10] = {{8, 4, 4.0F}, {2, 1, 0.5F}};
  join.edge_updates[15] = {{5, 11, 1.0F}};

  const std::array<std::pair<std::size_t, std::size_t>, 3> batches{
      {{0, 8}, {8, 15}, {15, 16}}};
  for (const auto &[begin, end] : batches) {
    std::size_t expected = 0;
    for (auto row = begin; row < end; ++row) {
      for (const auto &[p, q, weight] : join.edge_updates[row]) {
        expected += serial.checked_push_pair(p, weight, q);
      }
    }
    const auto count = join.apply_routed(routed, begin, end, 4, executor);
    assert(count == expected);
    assert(routed.idx == serial.idx && routed.dist == serial.dist &&
           routed.flags == serial.flags);
    for (auto row = begin; row < end; ++row) {
      assert(join.edge_updates[row].empty());
    }
    if (begin == 0) {
      assert(routed.contains(11, 1) && !routed.contains(11, 2));
      assert(!join.edge_updates[8].empty());
    }
  }
  std::cout << "destination routing: PASS (ordered heaps across batches)\n";
}

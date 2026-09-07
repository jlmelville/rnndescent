#include "pforr.h"

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr const char *standard_message = "standard worker failure";
constexpr const char *unknown_message =
    "parallel worker threw a non-standard exception";

bool fail(const char *test_name, const std::string &detail) {
  std::cerr << test_name << ": " << detail << "\n";
  return false;
}

bool expect_runtime_error(const char *test_name, const char *expected,
                          const std::function<void()> &action) {
  try {
    action();
  } catch (const std::runtime_error &error) {
    if (error.what() == std::string(expected)) {
      return true;
    }
    return fail(test_name, "unexpected message: " + std::string(error.what()));
  } catch (const std::exception &error) {
    return fail(test_name,
                "unexpected standard exception: " + std::string(error.what()));
  } catch (...) {
    return fail(test_name, "unexpected non-standard exception");
  }
  return fail(test_name, "expected std::runtime_error");
}

bool empty_and_grain_size_boundaries() {
  std::atomic<std::size_t> calls(0);
  auto regular_worker = [&](std::size_t, std::size_t) {
    calls.fetch_add(1, std::memory_order_relaxed);
  };
  auto indexed_worker = [&](std::size_t, std::size_t, std::size_t) {
    calls.fetch_add(1, std::memory_order_relaxed);
  };
  pforr::parallel_for(4, 4, regular_worker, 4, 0);
  pforr::parallel_for_indexed(4, 4, indexed_worker, 4, 0);
  if (calls.load(std::memory_order_relaxed) != 0) {
    return fail("empty ranges", "a worker was called");
  }

  const auto minimum = pforr::split_input_range({4, 7}, 4, 0);
  const auto whole = pforr::split_input_range({4, 7}, 4, 3);
  if (minimum.size() != 3 || minimum[0] != pforr::IndexRange(4, 5) ||
      minimum[1] != pforr::IndexRange(5, 6) ||
      minimum[2] != pforr::IndexRange(6, 7) || whole.size() != 1 ||
      whole[0] != pforr::IndexRange(4, 7)) {
    return fail("grain-size boundaries", "unexpected range partition");
  }
  return true;
}

bool success_paths_cover_uneven_ranges() {
  std::atomic<std::size_t> regular_count(0);
  auto regular_worker = [&](std::size_t begin, std::size_t end) {
    regular_count.fetch_add(end - begin, std::memory_order_relaxed);
  };
  pforr::parallel_for(0, 17, regular_worker, 4);

  std::atomic<std::size_t> indexed_count(0);
  std::atomic<unsigned int> chunk_mask(0);
  auto indexed_worker = [&](std::size_t begin, std::size_t end,
                            std::size_t chunk_id) {
    indexed_count.fetch_add(end - begin, std::memory_order_relaxed);
    chunk_mask.fetch_or(1U << chunk_id, std::memory_order_relaxed);
  };
  pforr::parallel_for_indexed(0, 17, indexed_worker, 4);

  if (regular_count.load(std::memory_order_relaxed) != 17 ||
      indexed_count.load(std::memory_order_relaxed) != 17 ||
      chunk_mask.load(std::memory_order_relaxed) != 0xFU) {
    return fail("success paths", "uneven ranges were not fully processed");
  }
  return true;
}

bool parallel_for_transports_standard_failure_after_join() {
  std::atomic<std::size_t> completed(0);
  auto worker = [&](std::size_t begin, std::size_t end) {
    if (begin == 0) {
      throw std::logic_error(standard_message);
    }
    completed.fetch_add(end - begin, std::memory_order_relaxed);
  };

  const bool transported =
      expect_runtime_error("parallel_for standard failure", standard_message,
                           [&]() { pforr::parallel_for(0, 17, worker, 4); });
  if (completed.load(std::memory_order_relaxed) != 12) {
    return fail("parallel_for standard failure",
                "caller resumed before surviving workers completed");
  }
  return transported;
}

bool parallel_for_transports_unknown_failure_after_join() {
  std::atomic<std::size_t> completed(0);
  auto worker = [&](std::size_t begin, std::size_t end) {
    if (begin == 0) {
      throw 17;
    }
    completed.fetch_add(end - begin, std::memory_order_relaxed);
  };

  const bool transported =
      expect_runtime_error("parallel_for non-standard failure", unknown_message,
                           [&]() { pforr::parallel_for(0, 17, worker, 4); });
  if (completed.load(std::memory_order_relaxed) != 12) {
    return fail("parallel_for non-standard failure",
                "caller resumed before surviving workers completed");
  }
  return transported;
}

bool indexed_parallel_for_transports_standard_failure_after_join() {
  std::atomic<std::size_t> completed(0);
  auto worker = [&](std::size_t begin, std::size_t end, std::size_t chunk_id) {
    if (chunk_id == 0) {
      throw std::logic_error(standard_message);
    }
    completed.fetch_add(end - begin, std::memory_order_relaxed);
  };

  const bool transported =
      expect_runtime_error("indexed standard failure", standard_message, [&]() {
        pforr::parallel_for_indexed(0, 17, worker, 4);
      });
  if (completed.load(std::memory_order_relaxed) != 12) {
    return fail("indexed standard failure",
                "caller resumed before surviving workers completed");
  }
  return transported;
}

bool indexed_parallel_for_transports_unknown_failure_after_join() {
  std::atomic<std::size_t> completed(0);
  auto worker = [&](std::size_t begin, std::size_t end, std::size_t chunk_id) {
    if (chunk_id == 0) {
      throw 23;
    }
    completed.fetch_add(end - begin, std::memory_order_relaxed);
  };

  const bool transported = expect_runtime_error(
      "indexed non-standard failure", unknown_message,
      [&]() { pforr::parallel_for_indexed(0, 17, worker, 4); });
  if (completed.load(std::memory_order_relaxed) != 12) {
    return fail("indexed non-standard failure",
                "caller resumed before surviving workers completed");
  }
  return transported;
}

bool serial_paths_preserve_original_exceptions() {
  auto regular_standard = [](std::size_t, std::size_t) {
    throw std::logic_error(standard_message);
  };
  try {
    pforr::parallel_for(0, 1, regular_standard, 1);
    return fail("serial parallel_for standard failure", "expected exception");
  } catch (const std::logic_error &error) {
    if (error.what() != std::string(standard_message)) {
      return fail("serial parallel_for standard failure", "message changed");
    }
  } catch (...) {
    return fail("serial parallel_for standard failure", "type changed");
  }

  auto regular_unknown = [](std::size_t, std::size_t) { throw 29; };
  try {
    pforr::parallel_for(0, 1, regular_unknown, 1);
    return fail("serial parallel_for non-standard failure",
                "expected exception");
  } catch (int value) {
    if (value != 29) {
      return fail("serial parallel_for non-standard failure", "value changed");
    }
  } catch (...) {
    return fail("serial parallel_for non-standard failure", "type changed");
  }

  auto indexed_standard = [](std::size_t, std::size_t, std::size_t) {
    throw std::invalid_argument(standard_message);
  };
  try {
    pforr::parallel_for_indexed(0, 1, indexed_standard, 1);
    return fail("serial indexed standard failure", "expected exception");
  } catch (const std::invalid_argument &error) {
    if (error.what() != std::string(standard_message)) {
      return fail("serial indexed standard failure", "message changed");
    }
  } catch (...) {
    return fail("serial indexed standard failure", "type changed");
  }

  auto indexed_unknown = [](std::size_t, std::size_t, std::size_t) {
    throw 31;
  };
  try {
    pforr::parallel_for_indexed(0, 1, indexed_unknown, 1);
    return fail("serial indexed non-standard failure", "expected exception");
  } catch (int value) {
    if (value != 31) {
      return fail("serial indexed non-standard failure", "value changed");
    }
  } catch (...) {
    return fail("serial indexed non-standard failure", "type changed");
  }

  return true;
}

bool first_failure_is_retained() {
  pforr::detail::WorkerFailure standard_first;
  const std::logic_error first("first failure");
  const std::runtime_error second("second failure");
  standard_first.capture(first);
  standard_first.capture(second);
  if (!expect_runtime_error("first standard failure", "first failure",
                            [&]() { standard_first.throw_if_set(); })) {
    return false;
  }

  pforr::detail::WorkerFailure unknown_first;
  unknown_first.capture_unknown();
  unknown_first.capture(first);
  return expect_runtime_error("first non-standard failure", unknown_message,
                              [&]() { unknown_first.throw_if_set(); });
}

} // namespace

int main() {
  if (!empty_and_grain_size_boundaries() ||
      !success_paths_cover_uneven_ranges() ||
      !parallel_for_transports_standard_failure_after_join() ||
      !parallel_for_transports_unknown_failure_after_join() ||
      !indexed_parallel_for_transports_standard_failure_after_join() ||
      !indexed_parallel_for_transports_unknown_failure_after_join() ||
      !serial_paths_preserve_original_exceptions() ||
      !first_failure_is_retained()) {
    return 1;
  }
  return 0;
}

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

#ifndef TDOANN_PARALLEL_H
#define TDOANN_PARALLEL_H

#include <functional>
#include <utility>

#include "progressbase.h"

namespace tdoann {

const constexpr std::size_t DEFAULT_NUM_BLOCKS{10};

class Executor {
public:
  virtual ~Executor() = default;
  virtual void
  parallel_for(std::size_t begin, std::size_t end,
               std::function<void(std::size_t, std::size_t)> worker,
               std::size_t n_threads, std::size_t grain_size = 1) = 0;
};

struct ExecutionParams {
  std::size_t batch_size{0};
  std::size_t grain_size{1};

  ExecutionParams() {}

  explicit ExecutionParams(std::size_t bs) : batch_size(bs) {}

  ExecutionParams(std::size_t bs, std::size_t gs)
      : batch_size(bs), grain_size(gs) {}

  // batch_size 0 means "1 batch of size n"
  auto get_batch_size(std::size_t n) const -> std::size_t {
    if (batch_size != 0) {
      return batch_size;
    }
    return n;
  }

  // get the appropriate batch size given n, subject to the grain size
  // (which should always be at least 1)
  auto batch_size_for_n_batches(std::size_t num_batches, std::size_t n)
      -> std::size_t {
    return std::max(n / num_batches, grain_size);
  }
};

class SerialExecutor : public Executor {
public:
  void parallel_for(std::size_t begin, std::size_t end,
                    std::function<void(std::size_t, std::size_t)> worker,
                    std::size_t /* n_threads */,
                    std::size_t /* grain_size */) override {
    worker(begin, end);
  }
};

template <typename Worker>
void batch_parallel_for(Worker &&worker, std::size_t n, std::size_t n_threads,
                        const ExecutionParams &execution_params,
                        ProgressBase &progress, Executor &executor) {
  if (n_threads == 0) {
    batch_serial_for(worker, n, execution_params, progress);
    return;
  }

  const auto batch_size = execution_params.get_batch_size(n);
  const auto grain_size = execution_params.grain_size;

  auto n_batches = (n + batch_size - 1) / batch_size;
  progress.set_n_batches(n_batches);

  std::function<void(std::size_t, std::size_t)> func_worker = worker;

  for (std::size_t i = 0; i < n_batches; i++) {
    auto begin = i * batch_size;
    auto end = std::min(n, begin + batch_size);
    executor.parallel_for(begin, end, func_worker, n_threads, grain_size);
    if (progress.check_interrupt()) {
      break;
    }
    progress.batch_finished();
  }
}

template <typename Worker>
void batch_parallel_for(Worker &&worker, std::size_t n, std::size_t n_threads,
                        ProgressBase &progress, Executor &executor) {
  batch_parallel_for(std::forward<Worker>(worker), n, n_threads, {}, progress,
                     executor);
}

template <typename Worker>
void batch_parallel_for(Worker &&worker, std::size_t n, std::size_t n_threads,
                        Executor &executor) {
  NullProgress progress;
  batch_parallel_for(std::forward<Worker>(worker), n, n_threads, {}, progress,
                     executor);
}

template <typename Worker, typename AfterWorker>
void batch_parallel_for(Worker &&worker, AfterWorker &after_worker,
                        std::size_t n, std::size_t n_threads,
                        const ExecutionParams &execution_params,
                        ProgressBase &progress, Executor &executor) {
  if (n_threads == 0) {
    batch_serial_for(worker, after_worker, n, execution_params, progress);
    return;
  }

  const auto batch_size = execution_params.get_batch_size(n);
  const auto grain_size = execution_params.grain_size;

  auto n_batches = (n + batch_size - 1) / batch_size;
  progress.set_n_batches(n_batches);

  std::function<void(std::size_t, std::size_t)> func_worker = worker;

  for (std::size_t i = 0; i < n_batches; i++) {
    auto begin = i * batch_size;
    auto end = std::min(n, begin + batch_size);
    executor.parallel_for(begin, end, func_worker, n_threads, grain_size);
    if (progress.check_interrupt()) {
      break;
    }
    after_worker(begin, end);
    if (progress.check_interrupt()) {
      break;
    }
    progress.batch_finished();
  }
}

template <typename Worker>
void batch_serial_for(Worker &worker, std::size_t n,
                      const ExecutionParams &execution_params,
                      ProgressBase &progress) {
  const auto &batch_size = execution_params.get_batch_size(n);
  auto n_batches = (n + batch_size - 1) / batch_size;
  progress.set_n_batches(n_batches);
  for (std::size_t i = 0; i < n_batches; i++) {
    auto begin = i * batch_size;
    auto end = std::min(n, begin + batch_size);
    worker(begin, end);
    if (progress.check_interrupt()) {
      break;
    }
    progress.batch_finished();
  }
}

template <typename Worker, typename AfterWorker>
void batch_serial_for(Worker &worker, AfterWorker &after_worker, std::size_t n,
                      const ExecutionParams &execution_params,
                      ProgressBase &progress) {
  const auto &batch_size = execution_params.get_batch_size(n);
  auto n_batches = (n + batch_size - 1) / batch_size;
  progress.set_n_batches(n_batches);
  for (std::size_t i = 0; i < n_batches; i++) {
    auto begin = i * batch_size;
    auto end = std::min(n, begin + batch_size);
    worker(begin, end);
    if (progress.check_interrupt()) {
      break;
    }
    after_worker(begin, end);
    if (progress.check_interrupt()) {
      break;
    }
    progress.batch_finished();
  }
}
} // namespace tdoann

#endif // TDOANN_PARALLEL_H

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

#include "progressbase.h"

namespace tdoann {

const constexpr std::size_t DEFAULT_NUM_BLOCKS{10};

struct NoParallel {
  template <typename Worker>
  static void parallel_for(std::size_t begin, std::size_t end, Worker &worker,
                           std::size_t /* n_threads */,
                           std::size_t /* grain_size */) {
    worker(begin, end);
  }
};

template <typename Parallel = NoParallel, typename Worker>
void batch_parallel_for(Worker &worker, std::size_t n, std::size_t block_size,
                        std::size_t n_threads, std::size_t grain_size,
                        ProgressBase &progress) {
  if (n_threads == 0) {
    batch_serial_for(worker, n, block_size, progress);
    return;
  }

  auto n_blocks = (n / block_size) + 1;
  progress.set_n_blocks(n_blocks);
  for (std::size_t i = 0; i < n_blocks; i++) {
    auto begin = i * block_size;
    auto end = std::min(n, begin + block_size);
    Parallel::parallel_for(begin, end, worker, n_threads, grain_size);
    if (progress.check_interrupt()) {
      break;
    }
    progress.block_finished();
  }
}

template <typename Parallel = NoParallel, typename Worker, typename AfterWorker>
void batch_parallel_for(Worker &worker, AfterWorker &after_worker,
                        std::size_t n, std::size_t block_size,
                        std::size_t n_threads, std::size_t grain_size,
                        ProgressBase &progress) {
  if (n_threads == 0) {
    batch_serial_for(worker, after_worker, n, block_size, progress);
    return;
  }

  auto n_blocks = (n / block_size) + 1;
  progress.set_n_blocks(n_blocks);
  for (std::size_t i = 0; i < n_blocks; i++) {
    auto begin = i * block_size;
    auto end = std::min(n, begin + block_size);
    Parallel::parallel_for(begin, end, worker, n_threads, grain_size);
    if (progress.check_interrupt()) {
      break;
    }
    after_worker(begin, end);
    if (progress.check_interrupt()) {
      break;
    }
    progress.block_finished();
  }
}

template <typename Parallel = NoParallel, typename Worker>
void batch_parallel_for(Worker &worker, std::size_t n, std::size_t n_threads,
                        std::size_t grain_size, ProgressBase &progress) {
  std::size_t block_size = std::max(grain_size, n / DEFAULT_NUM_BLOCKS);
  batch_parallel_for<Parallel>(worker, n, block_size, n_threads, grain_size,
                               progress);
}

template <typename Parallel = NoParallel, typename Worker>
void batch_parallel_for(Worker &worker, std::size_t n, std::size_t n_threads) {
  NullProgress progress;
  std::size_t grain_size = 1;
  std::size_t block_size = std::max(grain_size, n / DEFAULT_NUM_BLOCKS);
  batch_parallel_for<Parallel>(worker, n, block_size, n_threads, grain_size,
                               progress);
}

template <typename Worker>
void batch_serial_for(Worker &worker, std::size_t n, std::size_t block_size,
                      ProgressBase &progress) {
  auto n_blocks = (n / block_size) + 1;
  progress.set_n_blocks(n_blocks);
  for (std::size_t i = 0; i < n_blocks; i++) {
    auto begin = i * block_size;
    auto end = std::min(n, begin + block_size);
    worker(begin, end);
    if (progress.check_interrupt()) {
      break;
    }
    progress.block_finished();
  }
}

template <typename Worker, typename AfterWorker>
void batch_serial_for(Worker &worker, AfterWorker &after_worker, std::size_t n,
                      std::size_t block_size, ProgressBase &progress) {
  auto n_blocks = (n / block_size) + 1;
  progress.set_n_blocks(n_blocks);
  for (std::size_t i = 0; i < n_blocks; i++) {
    auto begin = i * block_size;
    auto end = std::min(n, begin + block_size);
    worker(begin, end);
    if (progress.check_interrupt()) {
      break;
    }
    after_worker(begin, end);
    if (progress.check_interrupt()) {
      break;
    }
    progress.block_finished();
  }
}
} // namespace tdoann

#endif // TDOANN_PARALLEL_H

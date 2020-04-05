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

#include "progress.h"

namespace tdoann {

struct NoParallel {
  template <typename Worker>
  static void parallel_for(std::size_t begin, std::size_t end, Worker &worker,
                           std::size_t, std::size_t) {
    worker(begin, end);
  }
};

struct BatchParallelWorker {
  void after_parallel(std::size_t, std::size_t) {}
};

template <typename Parallel, typename Progress, typename Worker>
void batch_parallel_for(Worker &worker, Progress &progress, std::size_t n,
                        std::size_t n_threads, std::size_t block_size,
                        std::size_t grain_size) {
  auto n_blocks = (n / block_size) + 1;
  progress.set_n_blocks(n_blocks);
  for (std::size_t i = 0; i < n_blocks; i++) {
    auto begin = i * block_size;
    auto end = std::min(n, begin + block_size);
    Parallel::parallel_for(begin, end, worker, n_threads, grain_size);
    TDOANN_BREAKIFINTERRUPTED();
    worker.after_parallel(begin, end);
    TDOANN_BLOCKFINISHED();
  }
}

template <typename Progress, typename Worker>
void batch_serial_for(Worker &worker, Progress &progress, std::size_t n,
                      std::size_t block_size) {
  auto n_blocks = (n / block_size) + 1;
  progress.set_n_blocks(n_blocks);
  for (std::size_t i = 0; i < n_blocks; i++) {
    auto begin = i * block_size;
    auto end = std::min(n, begin + block_size);
    worker(begin, end);
    TDOANN_BLOCKFINISHED();
  }
}

} // namespace tdoann

#endif // TDOANN_PARALLEL_H

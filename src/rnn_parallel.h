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

// Generic parallel helper code

#ifndef RNN_PARALLEL_H
#define RNN_PARALLEL_H

#include <Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>

struct ParallelOnlyWorker {
  RcppParallel::Worker &parallel_worker;

  ParallelOnlyWorker(RcppParallel::Worker &parallel_worker)
      : parallel_worker(parallel_worker){};

  void after_parallel(std::size_t begin, std::size_t end) {}
};

template <typename Progress>
void batch_parallel_for(RcppParallel::Worker &worker, Progress &progress,
                        std::size_t n, std::size_t block_size,
                        std::size_t grain_size) {
  bool dont_care_if_interrupted = false;
  batch_parallel_for(worker, progress, n, block_size, grain_size,
                     dont_care_if_interrupted);
}

template <typename Progress>
void batch_parallel_for(RcppParallel::Worker &worker, Progress &progress,
                        std::size_t n, std::size_t block_size,
                        std::size_t grain_size, bool &interrupted) {
  ParallelOnlyWorker parallel_only_worker(worker);
  batch_parallel_for(parallel_only_worker, progress, n, block_size, grain_size,
                     interrupted);
}

template <typename BatchParallelWorker, typename Progress>
void batch_parallel_for(BatchParallelWorker &rnn_worker, Progress &progress,
                        std::size_t n, std::size_t block_size,
                        std::size_t grain_size, bool &interrupted) {
  interrupted = false;
  if (n <= block_size) {
    RcppParallel::parallelFor(0, n, rnn_worker.parallel_worker, grain_size);
    if (progress.check_interrupt()) {
      interrupted = true;
      return;
    }
    rnn_worker.after_parallel(0, n);
    if (progress.check_interrupt()) {
      interrupted = true;
      return;
    }
  } else {
    const auto n_blocks = (n / block_size) + 1;
    for (std::size_t i = 0; i < n_blocks; i++) {
      const auto begin = i * block_size;
      const auto end = std::min(n, begin + block_size);

      RcppParallel::parallelFor(begin, end, rnn_worker.parallel_worker,
                                grain_size);
      if (progress.check_interrupt()) {
        interrupted = true;
        break;
      }
      rnn_worker.after_parallel(begin, end);
      if (progress.check_interrupt()) {
        interrupted = true;
        break;
      }
    }
  }
}

#endif // RNN_PARALLEL_H

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

#include "tdoann/heap.h"
#include "tdoann/progress.h"
#include "tdoann/typedefs.h"

#include "RcppPerpendicular.h"

struct BatchParallelWorker {
  void after_parallel(std::size_t begin, std::size_t end) {}
};

template <typename Progress, typename Worker>
void batch_parallel_for(Worker &rnn_worker, Progress &progress, std::size_t n,
                        std::size_t n_threads, std::size_t block_size,
                        std::size_t grain_size) {
  auto n_blocks = (n / block_size) + 1;
  progress.set_n_blocks(n_blocks);
  for (std::size_t i = 0; i < n_blocks; i++) {
    auto begin = i * block_size;
    auto end = std::min(n, begin + block_size);
    RcppPerpendicular::parallel_for(begin, end, rnn_worker, n_threads,
                                    grain_size);
    TDOANN_BREAKIFINTERRUPTED();
    rnn_worker.after_parallel(begin, end);
    TDOANN_BLOCKFINISHED();
  }
}

template <typename Progress, typename Worker>
void batch_serial_for(Worker &rnn_worker, Progress &progress, std::size_t n,
                      std::size_t block_size) {
  auto n_blocks = (n / block_size) + 1;
  progress.set_n_blocks(n_blocks);
  for (std::size_t i = 0; i < n_blocks; i++) {
    auto begin = i * block_size;
    auto end = std::min(n, begin + block_size);
    rnn_worker(begin, end);
    TDOANN_BLOCKFINISHED();
  }
}

#endif // RNN_PARALLEL_H

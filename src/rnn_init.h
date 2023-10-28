//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2023 James Melville
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

#ifndef RNN_INIT_H
#define RNN_INIT_H

#include <Rcpp.h>

#include "tdoann/distancebase.h"
#include "tdoann/heap.h"
#include "tdoann/randnbrs.h"

#include "rnndescent/random.h"

#include "rnn_parallel.h"
#include "rnn_progress.h"
#include "rnn_util.h"

// fill up heap, replacing any "missing" data with randomly chosen neighbors
template <typename NbrHeap>
void fill_random(NbrHeap &current_graph,
                 const tdoann::BaseDistance<typename NbrHeap::DistanceOut,
                                            typename NbrHeap::Index> &distance,
                 std::size_t n_threads, bool verbose) {
  if (verbose) {
    tsmessage() << "Filling graph with random neighbors (where needed)"
                << "\n";
  }

  rnndescent::ParallelIntRNGAdapter<typename NbrHeap::Index,
                                    rnndescent::DQIntSampler>
      rng_provider;
  RParallelExecutor executor;
  RPProgress fill_progress(false);
  tdoann::fill_random(current_graph, distance, rng_provider, n_threads,
                      fill_progress, executor);
  if (verbose) {
    tsmessage() << "Done\n";
  }
}

#endif // RNN_UTIL_H

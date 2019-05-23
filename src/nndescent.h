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

#ifndef RNND_NNDESCENT_H
#define RNND_NNDESCENT_H

#include "heap.h"

template <typename Heap,
          typename Rand>
ArrayHeap build_candidates(Heap& current_graph,
                         std::size_t max_candidates,
                         const std::size_t npoints,
                         const std::size_t nnbrs) {
  Rand rand;

  ArrayHeap candidate_neighbors(npoints, max_candidates);

  for (std::size_t i = 0; i < npoints; i++) {
    for (std::size_t j = 0; j < nnbrs; j++) {
      if (current_graph.idx[i][j] < 0) {
        continue;
      }
      std::size_t idx = current_graph.idx[i][j];
      bool isn = current_graph.flags[i][j];
      double d = rand.unif();

      candidate_neighbors.add_pair(i, idx, d, isn);
      current_graph.flags[i][j] = false;
    }
  }
  return candidate_neighbors;
}


template <typename Heap,
          typename Distance,
          typename Rand,
          typename Progress>
void nnd(
    Heap& heap,
    const std::size_t max_candidates,
    const std::size_t n_iters,
    const std::size_t npoints,
    const std::size_t nnbrs,
    Distance distance,
    Rand rand,
    Progress progress,
    const double rho,
    const double tol,
    bool verbose)
{
  for (std::size_t n = 0; n < n_iters; n++) {
    if (verbose) {
      progress.iter(n, n_iters, heap);
    }

    ArrayHeap candidate_neighbors = build_candidates<Heap, Rand>(heap, max_candidates, npoints, nnbrs);

    std::size_t c = 0;
    for (std::size_t i = 0; i < npoints; i++) {
      for (std::size_t j = 0; j < max_candidates; j++) {
        int p = candidate_neighbors.idx[i][j];
        if (p < 0 || rand.unif() < rho) {
          continue;
        }

        for (std::size_t k = 0; k < max_candidates; k++) {
          std::size_t q = candidate_neighbors.idx[i][k];
          if (q == Heap::npos || (!candidate_neighbors.flags[i][j] &&
              !candidate_neighbors.flags[i][k])) {
              continue;
          }
          double d = distance(p, q);
          c += heap.add_pair(p, q, d, true);
        }
      }
      progress.check_interrupt();
    }
    if (static_cast<double>(c) <= tol) {
      if (verbose) {
        progress.converged(c, tol);
      }
      break;
    }
  }
}


#endif // RNND_NNDESCENT_H

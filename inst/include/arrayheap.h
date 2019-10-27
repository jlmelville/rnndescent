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

#ifndef RNND_ARRAYHEAP_H
#define RNND_ARRAYHEAP_H

#include "heap.h"

// Checks for duplicates by iterating over the entire array of stored indexes
template <typename WeightMeasure>
struct ArrayHeap
{
  NeighborHeap neighbor_heap;
  WeightMeasure weight_measure;

  ArrayHeap(
    WeightMeasure& weight_measure,
    const std::size_t n_points,
    const std::size_t size) :
    neighbor_heap(n_points, size),
    weight_measure(weight_measure)
  {}

  ArrayHeap(const ArrayHeap&) = default;
  ~ArrayHeap() = default;
  ArrayHeap& operator=(const ArrayHeap &) = default;

  std::size_t add_pair(
      std::size_t i,
      std::size_t j,
      bool flag)
  {
    double d = weight_measure(i, j);

    std::size_t c = push(i, d, j, flag);
    if (i != j) {
      c += push(j, d, i, flag);
    }

    return c;
  }

  std::size_t add_pair_asymm(
      std::size_t i,
      std::size_t j,
      bool flag)
  {
    if (contains(i, j)) {
      return 0;
    }

    double weight = weight_measure(i, j);

    if (weight >= neighbor_heap.distance(i, 0)) {
      return 0;
    }

    return neighbor_heap.unchecked_push(i, weight, j, flag);
  }

  bool contains(std::size_t row, std::size_t index) const
  {
    const std::size_t n_nbrs = neighbor_heap.n_nbrs;
    const std::size_t rnnbrs = row * n_nbrs;
    for (std::size_t i = 0; i < n_nbrs; i++) {
      if (index == neighbor_heap.index(rnnbrs + i)) {
        return true;
      }
    }
    return false;
  }

  std::size_t push(
      std::size_t row,
      double weight,
      std::size_t index,
      bool flag)
  {
    if (weight >= neighbor_heap.distance(row, 0)) {
      return 0;
    }

    // break if we already have this element
    if (contains(row, index)) {
      return 0;
    }

    return neighbor_heap.unchecked_push(row, weight, index, flag);
  }
};

template <typename Rand>
struct RandomWeight
{
  Rand rand;

  RandomWeight(Rand& rand) : rand(rand) { }

  double operator()(
      std::size_t i,
      std::size_t j)
  {
    return rand.unif();
  }
};

template <typename Rand>
using RandomHeap = ArrayHeap<RandomWeight<Rand>>;

#endif // NND_ARRAYHEAP_H

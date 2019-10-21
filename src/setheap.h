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

#ifndef RNND_SETHEAP_H
#define RNND_SETHEAP_H

#include <unordered_set>

#include <boost/functional/hash.hpp>

#include "heap.h"

using pair = std::pair<std::size_t, std::size_t>;


// Checks for duplicates by storing a set of already-seen pairs. Takes up more
// memory but might be faster if lots of duplicate pairs are expected
template <typename WeightMeasure>
struct SetHeap
{
  NeighborHeap neighbor_heap;
  WeightMeasure weight_measure;
  std::unordered_set<std::pair<std::size_t, std::size_t>,
                     boost::hash<pair>> seen;

  SetHeap(WeightMeasure& weight_measure,
          const std::size_t n_points,
          const std::size_t size)
    : neighbor_heap(n_points, size),
      weight_measure(weight_measure),
      seen()
    {}

  SetHeap(const SetHeap&) = default;
  ~SetHeap() = default;
  SetHeap& operator=(const SetHeap &) = default;

  std::size_t add_pair(
      std::size_t i,
      std::size_t j,
      bool flag)
  {
    if (i > j) {
      std::swap(i, j);
    }

    pair p(i, j);
    if (seen.find(p) != seen.end()) {
      return 0;
    }
    seen.insert(p);

    double d = weight_measure(i, j);

    std::size_t c = 0;
    if (d < neighbor_heap.distance(i, 0)) {
      c += neighbor_heap.unchecked_push(i, d, j, flag);
    }
    if (i != j && d < neighbor_heap.distance(j, 0)) {
      c += neighbor_heap.unchecked_push(j, d, i, flag);
    }

    return c;
  }

  std::size_t add_pair_asymm(
      std::size_t i,
      std::size_t j,
      bool flag)
  {
    pair p(i, j);
    if (seen.find(p) != seen.end()) {
      return 0;
    }
    seen.insert(p);

    double d = weight_measure(i, j);

    std::size_t c = 0;
    if (d < neighbor_heap.distance(i, 0)) {
      c += neighbor_heap.unchecked_push(i, d, j, flag);
    }

    return c;
  }
};

#endif // RNND_SETHEAP_H

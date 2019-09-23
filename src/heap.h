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

#ifndef RNND_HEAP_H
#define RNND_HEAP_H

#include <limits>

// Base class storing neighbor data as a series of heaps
struct NeighborHeap
{
  // used in analogy with std::string::npos as used in std::string::find
  // to represent not found
  static constexpr std::size_t npos = static_cast<std::size_t>(-1);

  const std::size_t n_points;
  const std::size_t n_nbrs;
  std::vector<std::size_t> idx;
  std::vector<double> dist;
  std::vector<char> flags;

  NeighborHeap(
    const std::size_t n_points,
    const std::size_t n_nbrs) :
    n_points(n_points),
    n_nbrs(n_nbrs),
    idx(n_points * n_nbrs, npos),
    dist(n_points * n_nbrs, std::numeric_limits<double>::max()),
    flags(n_points * n_nbrs, 0)
  { }

  unsigned int unchecked_push(
      std::size_t row,
      double weight,
      std::size_t index,
      bool flag)
  {
    const std::size_t r0 = row * n_nbrs;

    // insert val at position zero
    dist[r0] = weight;
    idx[r0] = index;
    flags[r0] = flag ? 1 : 0;

    // descend the heap, swapping values until the max heap criterion is met
    std::size_t i = 0;
    std::size_t i_swap = 0;
    while (true) {
      std::size_t ic1 = 2 * i + 1;
      std::size_t ic2 = ic1 + 1;

      if (ic1 >= n_nbrs) {
        break;
      }
      else if (ic2 >= n_nbrs) {
        if (dist[r0 + ic1] >= weight) {
          i_swap = ic1;
        }
        else {
          break;
        }
      }
      else if (dist[r0 + ic1] >= dist[r0 + ic2]) {
        if (weight < dist[r0 + ic1]) {
          i_swap = ic1;
        }
        else {
          break;
        }
      }
      else {
        if (weight < dist[r0 + ic2]) {
          i_swap = ic2;
        }
        else {
          break;
        }
      }

      dist[r0 + i] = dist[r0 + i_swap];
      idx[r0 + i] = idx[r0 + i_swap];
      flags[r0 + i] = flags[r0 + i_swap];

      i = i_swap;
    }

    dist[r0 + i] = weight;
    idx[r0 + i] = index;
    flags[r0 + i] = flag ? 1 : 0;

    return 1;
  }

  void deheap_sort() {
    for (std::size_t i = 0; i < n_points; i++) {
      const std::size_t r0 = i * n_nbrs;
      for (std::size_t j = 0; j < n_nbrs - 1; j++) {
        std::swap(idx[r0], idx[r0 + (n_nbrs - j - 1)]);
        std::swap(dist[r0], dist[r0 + (n_nbrs - j - 1)]);
        siftdown(r0, n_nbrs - j - 1, 0);
      }
    }
  }

  void siftdown(const std::size_t r0,
                const std::size_t len,
                std::size_t elt) {

    while (elt * 2 + 1 < len) {
      std::size_t left_child = elt * 2 + 1;
      std::size_t right_child = left_child + 1;
      std::size_t swap = elt;

      if (dist[r0 + swap] < dist[r0 + left_child]) {
        swap = left_child;
      }

      if (right_child < len && dist[r0 + swap] < dist[r0 + right_child]) {
        swap = right_child;
      }

      if (swap == elt) {
        break;
      }
      else {
        std::swap(dist[r0 + elt], dist[r0 + swap]);
        std::swap(idx[r0 + elt], idx[r0 + swap]);
        elt = swap;
      }
    }
  }

  std::size_t index(std::size_t i, std::size_t j) const {
    return idx[i * n_nbrs + j];
  }
  std::size_t& index(std::size_t i, std::size_t j) {
    return idx[i * n_nbrs + j];
  }

  double distance(std::size_t i, std::size_t j) const {
    return dist[i * n_nbrs + j];
  }
  double& distance(std::size_t i, std::size_t j) {
    return dist[i * n_nbrs + j];
  }

  char flag(std::size_t i, std::size_t j) const {
    return flags[i * n_nbrs + j];
  }
  char& flag(std::size_t i, std::size_t j) {
    return flags[i * n_nbrs + j];
  }
};

constexpr std::size_t NeighborHeap::npos;


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

  unsigned int add_pair(
      std::size_t i,
      std::size_t j,
      bool flag)
  {
    double d = weight_measure(i, j);

    unsigned int c = 0;
    c += push(i, d, j, flag);
    if (i != j) {
      c += push(j, d, i, flag);
    }

    return c;
  }

  unsigned int push(
      std::size_t row,
      double weight,
      std::size_t index,
      bool flag)
  {
    if (weight >= neighbor_heap.distance(row, 0)) {
      return 0;
    }

    // break if we already have this element
    const std::size_t n_nbrs = neighbor_heap.n_nbrs;
    for (std::size_t i = 0; i < n_nbrs; i++) {
      if (index == neighbor_heap.index(row, i)) {
        return 0;
      }
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

#endif // RNDD_HEAP_H

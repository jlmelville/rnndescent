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

  std::size_t n_points;
  std::size_t n_nbrs;
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

  NeighborHeap(const NeighborHeap&) = default;
  ~NeighborHeap() = default;
  NeighborHeap& operator=(const NeighborHeap &) = default;

  std::size_t checked_push(
      std::size_t row,
      double weight,
      std::size_t idx,
      bool flag)
  {
    if (weight >= distance(row, 0)) {
      return 0;
    }

    // break if we already have this element
    for (std::size_t i = 0; i < n_nbrs; i++) {
      if (idx == index(row, i)) {
        return 0;
      }
    }

    return unchecked_push(row, weight, idx, flag);
  }

  std::size_t unchecked_push(
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
      const std::size_t ic1 = 2 * i + 1;
      const std::size_t ic2 = ic1 + 1;

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

      const std::size_t r0i = r0 + i;
      const std::size_t r0is = r0 + i_swap;
      dist[r0i] = dist[r0is];
      idx[r0i] = idx[r0is];
      flags[r0i] = flags[r0is];

      i = i_swap;
    }

    const std::size_t r0i = r0 + i;
    dist[r0i] = weight;
    idx[r0i] = index;
    flags[r0i] = flag ? 1 : 0;

    return 1;
  }

  void deheap_sort() {
    const std::size_t nnbrs1 = n_nbrs - 1;

    for (std::size_t i = 0; i < n_points; i++) {
      const std::size_t r0 = i * n_nbrs;
      for (std::size_t j = 0; j < nnbrs1; j++) {
        const std::size_t r0nn1 = r0 + nnbrs1 - j;
        std::swap(idx[r0], idx[r0nn1]);
        std::swap(dist[r0], dist[r0nn1]);
        siftdown(r0, nnbrs1 - j);
      }
    }
  }

  void siftdown(const std::size_t r0,
                const std::size_t len) {
    std::size_t elt = 0;
    std::size_t e21 = elt * 2 + 1;

    while (e21 < len) {
      const std::size_t left_child = e21;
      const std::size_t right_child = left_child + 1;
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
      e21 = elt * 2 + 1;
    }
  }

  std::size_t index(std::size_t i, std::size_t j) const {
    return idx[i * n_nbrs + j];
  }
  std::size_t& index(std::size_t i, std::size_t j) {
    return idx[i * n_nbrs + j];
  }
  std::size_t index(std::size_t ij) const {
    return idx[ij];
  }
  std::size_t& index(std::size_t ij) {
    return idx[ij];
  }

  double distance(std::size_t i, std::size_t j) const {
    return dist[i * n_nbrs + j];
  }
  double& distance(std::size_t i, std::size_t j) {
    return dist[i * n_nbrs + j];
  }
  double distance(std::size_t ij) const {
    return dist[ij];
  }
  double& distance(std::size_t ij) {
    return dist[ij];
  }

  char flag(std::size_t i, std::size_t j) const {
    return flags[i * n_nbrs + j];
  }
  char& flag(std::size_t i, std::size_t j) {
    return flags[i * n_nbrs + j];
  }
  char flag(std::size_t ij) const {
    return flags[ij];
  }
  char& flag(std::size_t ij) {
    return flags[ij];
  }

  void df(std::size_t i, std::size_t j,
          double& distance, bool& flag) const {
    auto pos = i * n_nbrs + j;
    distance = dist[pos];
    flag = flags[pos] == 1;
  }
  void df(std::size_t ij, double& distance, bool& flag) const {
    distance = dist[ij];
    flag = flags[ij] == 1;
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

#endif // RNDD_HEAP_H

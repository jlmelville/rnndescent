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

  std::vector<std::vector<std::size_t>> idx;
  std::vector<std::vector<double>> dist;
  std::vector<std::vector<bool>> flags; // vector of bool, yes ugh

  NeighborHeap(
    const std::size_t n_points,
    const std::size_t size)
  {
    for (std::size_t i = 0; i < n_points; i++) {
      idx.push_back(std::vector<std::size_t>(size, npos));
      dist.push_back(std::vector<double>(size,
                                         std::numeric_limits<double>::max()));
      flags.push_back(std::vector<bool>(size, false));
    }
  }

  unsigned int unchecked_push(
      std::size_t row,
      double weight,
      std::size_t index,
      bool flag)
  {
    std::vector<std::size_t>& indices = idx[row];
    std::vector<double>& weights = dist[row];
    std::vector<bool>& is_new = flags[row];

    const std::size_t n_nbrs = indices.size();

    // insert val at position zero
    weights[0] = weight;
    indices[0] = index;
    is_new[0] = flag;

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
        if (weights[ic1] >= weight) {
          i_swap = ic1;
        }
        else {
          break;
        }
      }
      else if (weights[ic1] >= weights[ic2]) {
        if (weight < weights[ic1]) {
          i_swap = ic1;
        }
        else {
          break;
        }
      }
      else {
        if (weight < weights[ic2]) {
          i_swap = ic2;
        }
        else {
          break;
        }
      }

      weights[i] = weights[i_swap];
      indices[i] = indices[i_swap];
      is_new[i] = is_new[i_swap];

      i = i_swap;
    }

    weights[i] = weight;
    indices[i] = index;
    is_new[i] = flag;

    return 1;
  }

  void deheap_sort() {
    const std::size_t npoints = idx.size();

    for (std::size_t i = 0; i < npoints; i++) {
      std::vector<std::size_t>& ind_heap = idx[i];
      std::vector<double>& dist_heap = dist[i];

      const std::size_t nnbrs = ind_heap.size();
      for (std::size_t j = 0; j < nnbrs - 1; j++) {
        std::swap(ind_heap[0], ind_heap[nnbrs - j - 1]);
        std::swap(dist_heap[0], dist_heap[nnbrs - j - 1]);
        siftdown(dist_heap, ind_heap, nnbrs - j - 1, 0);
      }
    }
  }

  void siftdown(std::vector<double>& dist_heap,
                std::vector<std::size_t>& ind_heap,
                const std::size_t len,
                std::size_t elt) {

    while (elt * 2 + 1 < len) {
      std::size_t left_child = elt * 2 + 1;
      std::size_t right_child = left_child + 1;
      std::size_t swap = elt;

      if (dist_heap[swap] < dist_heap[left_child]) {
        swap = left_child;
      }

      if (right_child < len && dist_heap[swap] < dist_heap[right_child]) {
        swap = right_child;
      }

      if (swap == elt) {
        break;
      }
      else {
        std::swap(dist_heap[elt], dist_heap[swap]);
        std::swap(ind_heap[elt], ind_heap[swap]);
        elt = swap;
      }
    }
  }
};

constexpr std::size_t NeighborHeap::npos;

///////////////////////////////////////

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
    if (weight >= neighbor_heap.dist[row][0]) {
      return 0;
    }

    // break if we already have this element
    std::vector<std::size_t>& indices = neighbor_heap.idx[row];
    const std::size_t n_nbrs = indices.size();
    for (std::size_t i = 0; i < n_nbrs; i++) {
      if (index == indices[i]) {
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

  RandomWeight() { }

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

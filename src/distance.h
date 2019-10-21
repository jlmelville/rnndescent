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

#ifndef RNND_DISTANCE_H
#define RNND_DISTANCE_H

#include <bitset>
#include <cmath>
#include <vector>

template <typename In, typename Out>
struct Euclidean
{
  Euclidean(const std::vector<In>& data, std::size_t ndim)
    : data(data), ndim(ndim) { }

  Out operator()(std::size_t i, std::size_t j) {
    Out sum = 0.0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      const Out diff = data[di + d] - data[dj + d];
      sum += diff * diff;
    }

    return std::sqrt(sum);
  }

  std::vector<In> data;
  std::size_t ndim;

  typedef In in_type;
};

template <typename In, typename Out>
struct L2
{
  L2(const std::vector<In>& data, std::size_t ndim)
    : data(data), ndim(ndim) { }

  Out operator()(std::size_t i, std::size_t j) {
    Out sum = 0.0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      const Out diff = data[di + d] - data[dj + d];
      sum += diff * diff;
    }

    return sum;
  }

  std::vector<In> data;
  std::size_t ndim;

  typedef In in_type;
};



template <typename In, typename Out>
struct Cosine
{
  Cosine(const std::vector<In>& _data, std::size_t ndim)
    : data(_data), ndim(ndim)

  {
    // normalize data on input
    const std::size_t npoints = data.size() / ndim;
    for (std::size_t i = 0; i < npoints; i++) {
      const std::size_t di = ndim * i;
      In norm = 0.0;

      for (std::size_t d = 0; d < ndim; d++) {
        const auto val = data[di + d];
        norm += val * val;
      }
      norm = 1.0 / (std::sqrt(norm) + 1e-30);
      for (std::size_t d = 0; d < ndim; d++) {
        data[di + d] *= norm;
      }
    }
  }

  Out operator()(std::size_t i, std::size_t j) {
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    Out sum = 0.0;
    for (std::size_t d = 0; d < ndim; d++) {
      sum += data[di + d] * data[dj + d];
    }

    return 1.0 - sum;
  }

  std::vector<In> data;
  std::size_t ndim;

  typedef In in_type;
};


template <typename In, typename Out>
struct Manhattan
{

  Manhattan(const std::vector<In>& data, std::size_t ndim)
    : data(data), ndim(ndim) { }

  Out operator()(std::size_t i, std::size_t j) {
    Out sum = 0.0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      sum += std::abs(data[di + d] - data[dj + d]);
    }

    return sum;
  }

  std::vector<In> data;
  std::size_t ndim;

  typedef In in_type;
};

template<typename In, typename Out>
struct Hamming
{
  Hamming(const std::vector<In>& vdata, std::size_t vndim) {
    // Instead of storing each bit as an element, we will pack them
    // into a series of 64-bit bitsets. Possibly compilers are smart enough
    // to use built in integer popcount routines for the bitset count()
    // method.
    std::bitset<64> bits;
    std::size_t bit_count = 0;
    std::size_t vd_count = 0;

    for (std::size_t i = 0; i < vdata.size(); i++) {
      if (bit_count == 64 || vd_count == vndim) {
        // filled up current bitset
        data.push_back(bits);
        bit_count = 0;
        bits.reset();

        if (vd_count == vndim) {
          // end of item
          vd_count = 0;
        }
      }
      bits[bit_count] = vdata[i];

      ++vd_count;
      ++bit_count;
    }
    if (bit_count > 0) {
      data.push_back(bits);
    }

    ndim = std::ceil(vndim / 64.0);
  }

  Out operator()(std::size_t i, std::size_t j) {
    Out sum = 0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      sum += (data[di + d] ^ data[dj + d]).count();
    }

    return sum;
  }

  std::vector<std::bitset<64>> data;
  std::size_t ndim;

  typedef In in_type;
};

#endif // RNND_DISTANCE_H

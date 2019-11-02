// BSD 2-Clause License
//
// Copyright 2019 James Melville
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// OF SUCH DAMAGE.

#ifndef NND_DISTANCE_H
#define NND_DISTANCE_H

#include <bitset>
#include <cmath>
#include <vector>

template <typename In, typename Out>
struct Euclidean
{
  Euclidean(const std::vector<In>& data, std::size_t ndim)
    : data(data), ndim(ndim) { }

  Out operator()(std::size_t i, std::size_t j) const {
    Out sum = 0.0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      const Out diff = data[di + d] - data[dj + d];
      sum += diff * diff;
    }

    return std::sqrt(sum);
  }

  const std::vector<In> data;
  const std::size_t ndim;

  typedef In in_type;
};

template <typename In, typename Out>
struct L2
{
  L2(const std::vector<In>& data, std::size_t ndim)
    : data(data), ndim(ndim) { }

  Out operator()(std::size_t i, std::size_t j) const {
    Out sum = 0.0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      const Out diff = data[di + d] - data[dj + d];
      sum += diff * diff;
    }

    return sum;
  }

  const std::vector<In> data;
  const std::size_t ndim;

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

  Out operator()(std::size_t i, std::size_t j) const {
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    Out sum = 0.0;
    for (std::size_t d = 0; d < ndim; d++) {
      sum += data[di + d] * data[dj + d];
    }

    return 1.0 - sum;
  }

  std::vector<In> data;
  const std::size_t ndim;

  typedef In in_type;
};


template <typename In, typename Out>
struct Manhattan
{
  Manhattan(const std::vector<In>& data, std::size_t ndim)
    : data(data), ndim(ndim) { }

  Out operator()(std::size_t i, std::size_t j) const {
    Out sum = 0.0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      sum += std::abs(data[di + d] - data[dj + d]);
    }

    return sum;
  }

  const std::vector<In> data;
  const std::size_t ndim;

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

  Out operator()(std::size_t i, std::size_t j) const {
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

#endif // NND_DISTANCE_H

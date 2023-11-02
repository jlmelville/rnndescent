// BSD 2-Clause License
//
// Copyright 2023 James Melville
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

#ifndef TDOANN_SPARSE_H
#define TDOANN_SPARSE_H

#include <algorithm>
#include <cmath>
#include <iterator>
#include <vector>

#include "distancebase.h"

namespace tdoann {

template <typename Out>
std::pair<std::vector<std::size_t>, std::vector<Out>>
sparse_sum(typename std::vector<std::size_t>::const_iterator ind1_start,
           typename std::vector<Out>::const_iterator data1_start,
           typename std::vector<std::size_t>::const_iterator ind2_start,
           typename std::vector<Out>::const_iterator data2_start,
           std::size_t ndim) {
  const auto result_size = 2 * ndim;
  std::vector<std::size_t> result_ind(result_size);
  std::vector<Out> result_data(result_size);

  std::size_t i1 = 0;
  std::size_t i2 = 0;
  std::size_t nnz = 0;

  // pass through both index lists
  while (i1 < ndim && i2 < ndim) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      auto val = *(data1_start + i1) + *(data2_start + i2);
      if (val != 0) {
        result_ind[nnz] = j1;
        result_data[nnz] = val;
        ++nnz;
      }
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      auto val = *(data1_start + i1);
      if (val != 0) {
        result_ind[nnz] = j1;
        result_data[nnz] = val;
        ++nnz;
      }
      ++i1;
    } else {
      auto val = *(data2_start + i2);
      if (val != 0) {
        result_ind[nnz] = j2;
        result_data[nnz] = val;
        ++nnz;
      }
      ++i2;
    }
  }

  // pass over the tails
  while (i1 < ndim) {
    auto j1 = *(ind1_start + i1);
    auto val = *(data1_start + i1);
    if (val != 0) {
      result_ind[nnz] = j1;
      result_data[nnz] = val;
      ++nnz;
    }
    ++i1;
  }

  while (i2 < ndim) {
    auto j2 = *(ind2_start + i2);
    auto val = *(data2_start + i2);
    if (val != 0) {
      result_ind[nnz] = j2;
      result_data[nnz] = val;
      ++nnz;
    }
    ++i2;
  }

  // truncate to the correct length in case there were zeros created
  result_ind.resize(nnz);
  result_data.resize(nnz);

  return {result_ind, result_data};
}

template <typename Out>
std::pair<std::vector<std::size_t>, std::vector<Out>>
sparse_diff(typename std::vector<std::size_t>::const_iterator ind1_start,
            typename std::vector<Out>::const_iterator data1_start,
            typename std::vector<std::size_t>::const_iterator ind2_start,
            typename std::vector<Out>::const_iterator data2_start,
            std::size_t ndim) {
  std::vector<Out> neg_data2(ndim);
  std::transform(data2_start, data2_start + ndim, neg_data2.begin(),
                 std::negate<Out>());

  return sparse_sum<Out>(ind1_start, data1_start, ind2_start, neg_data2.begin(),
                         ndim);
}

template <typename Out>
Out sparse_l2sqr(typename std::vector<std::size_t>::const_iterator ind1_start,
                 typename std::vector<Out>::const_iterator data1_start,
                 typename std::vector<std::size_t>::const_iterator ind2_start,
                 typename std::vector<Out>::const_iterator data2_start,
                 std::size_t ndim) {
  auto diff =
      sparse_diff<Out>(ind1_start, data1_start, ind2_start, data2_start, ndim);
  auto &aux_data = diff.second;

  Out result = Out();
  for (const auto &val : aux_data) {
    result += val * val;
  }

  return result;
}

template <typename Out>
Out sparse_euclidean(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    typename std::vector<Out>::const_iterator data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    typename std::vector<Out>::const_iterator data2_start, std::size_t ndim) {
  return std::sqrt(sparse_l2sqr<Out>(ind1_start, data1_start, ind2_start,
                                     data2_start, ndim));
}

template <typename In, typename Out, typename Idx = uint32_t>
class SparseL2SqrSelfDistance : public BaseDistance<Out, Idx> {
public:
  SparseL2SqrSelfDistance(std::vector<std::size_t> &&x_ind,
                          std::vector<In> &&x_data, std::size_t ndim,
                          std::size_t nx)
      : x_ind(std::move(x_ind)), x_data(std::move(x_data)), nx(nx), ndim(ndim) {
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    const std::size_t dj = this->ndim * j;

    return sparse_l2sqr<Out>(
        this->x_ind.begin() + di, this->x_data.begin() + di,
        this->x_ind.begin() + dj, this->x_data.begin() + dj, this->ndim);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return nx; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<In> x_data;
  std::size_t nx;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class SparseEuclideanSelfDistance : public BaseDistance<Out, Idx> {
public:
  SparseEuclideanSelfDistance(std::vector<std::size_t> &&x_ind,
                              std::vector<In> &&x_data, std::size_t ndim,
                              std::size_t nx)
      : x_ind(std::move(x_ind)), x_data(std::move(x_data)), nx(nx), ndim(ndim) {
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    const std::size_t dj = this->ndim * j;

    return sparse_euclidean<Out>(
        this->x_ind.begin() + di, this->x_data.begin() + di,
        this->x_ind.begin() + dj, this->x_data.begin() + dj, this->ndim);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return nx; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<In> x_data;
  std::size_t nx;
  std::size_t ndim;
};

} // namespace tdoann

#endif // TDOANN_SPARSE_H

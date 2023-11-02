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

// TODO: if we assume ar1 and ar2 are already sorted we could use std::set_union
inline std::vector<std::size_t> arr_union(const std::vector<std::size_t> &ar1,
                                          const std::vector<std::size_t> &ar2) {
  // Concatenate the two arrays
  std::vector<std::size_t> union_result(ar1.size() + ar2.size());
  std::copy(ar1.begin(), ar1.end(), union_result.begin());
  std::copy(ar2.begin(), ar2.end(), union_result.begin() + ar1.size());

  std::sort(union_result.begin(), union_result.end());

  // Remove duplicate elements
  union_result.erase(std::unique(union_result.begin(), union_result.end()),
                     union_result.end());

  return union_result;
}

template <typename DataIt>
std::pair<std::vector<std::size_t>,
          std::vector<typename std::iterator_traits<DataIt>::value_type>>
sparse_mul(typename std::vector<std::size_t>::const_iterator ind1_start,
           DataIt data1_start,
           typename std::vector<std::size_t>::const_iterator ind2_start,
           DataIt data2_start, std::size_t ndim) {
  std::vector<std::size_t> result_ind;
  std::vector<typename std::iterator_traits<DataIt>::value_type> result_data;
  std::size_t i1 = 0;
  std::size_t i2 = 0;
  while (i1 < ndim && i2 < ndim) {
    auto j1 = *(ind1_start + i1);
    auto j2 = *(ind2_start + i2);
    if (j1 == j2) {
      auto val = *(data1_start + i1) * *(data2_start + i2);
      if (val != 0) {
        result_ind.push_back(j1);
        result_data.push_back(val);
      }
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      ++i1;
    } else {
      ++i2;
    }
  }
  return {result_ind, result_data};
}

template <typename Out, typename DataIt>
std::pair<std::vector<std::size_t>, std::vector<Out>>
sparse_diff(typename std::vector<std::size_t>::const_iterator ind1_start,
            DataIt data1_start,
            typename std::vector<std::size_t>::const_iterator ind2_start,
            DataIt data2_start, std::size_t ndim) {
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
      auto val = static_cast<Out>(*(data1_start + i1) - *(data2_start + i2));
      if (val != 0) {
        result_ind[nnz] = j1;
        result_data[nnz] = val;
        ++nnz;
      }
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      auto val = static_cast<Out>(*(data1_start + i1));
      if (val != 0) {
        result_ind[nnz] = j1;
        result_data[nnz] = val;
        ++nnz;
      }
      ++i1;
    } else {
      auto val = static_cast<Out>(-*(data2_start + i2));
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
    auto val = static_cast<Out>(*(data1_start + i1));
    if (val != 0) {
      result_ind[nnz] = j1;
      result_data[nnz] = val;
      ++nnz;
    }
    ++i1;
  }

  while (i2 < ndim) {
    auto j2 = *(ind2_start + i2);
    auto val = static_cast<Out>(-*(data2_start + i2));
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

template <typename Out, typename DataIt>
Out sparse_l2sqr(typename std::vector<std::size_t>::const_iterator ind1_start,
                 DataIt data1_start,
                 typename std::vector<std::size_t>::const_iterator ind2_start,
                 DataIt data2_start, std::size_t ndim) {

  Out sum{0};

  std::size_t i1 = 0;
  std::size_t i2 = 0;

  while (i1 < ndim && i2 < ndim) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      auto val = static_cast<Out>(*(data1_start + i1) - *(data2_start + i2));
      sum += val * val;
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      auto val = static_cast<Out>(*(data1_start + i1));
      sum += val * val;
      ++i1;
    } else {
      auto val = static_cast<Out>(*(data2_start + i2));
      sum += val * val;
      ++i2;
    }
  }

  // pass over the tails
  while (i1 < ndim) {
    auto val = static_cast<Out>(*(data1_start + i1));
    sum += val * val;
    ++i1;
  }

  while (i2 < ndim) {
    auto val = static_cast<Out>(*(data2_start + i2));
    sum += val * val;
    ++i2;
  }

  return sum;
}

template <typename Out, typename DataIt>
Out sparse_euclidean(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    DataIt data2_start, std::size_t ndim) {
  return std::sqrt(sparse_l2sqr<Out>(ind1_start, data1_start, ind2_start,
                                     data2_start, ndim));
}

template <typename Out, typename DataIt>
Out sparse_manhattan(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    DataIt data2_start, std::size_t ndim) {

  Out result = Out();

  std::size_t i1 = 0;
  std::size_t i2 = 0;

  while (i1 < ndim && i2 < ndim) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      auto val =
          std::abs(static_cast<Out>(*(data1_start + i1) - *(data2_start + i2)));
      result += val;
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      result += std::abs(static_cast<Out>(*(data1_start + i1)));
      ++i1;
    } else {
      result += std::abs(static_cast<Out>(*(data2_start + i2)));
      ++i2;
    }
  }

  while (i1 < ndim) {
    result += std::abs(static_cast<Out>(*(data1_start + i1)));
    ++i1;
  }

  while (i2 < ndim) {
    result += std::abs(static_cast<Out>(*(data2_start + i2)));
    ++i2;
  }

  return result;
}

template <typename Out, typename DataIt>
Out sparse_hamming(typename std::vector<std::size_t>::const_iterator ind1_start,
                   DataIt data1_start,
                   typename std::vector<std::size_t>::const_iterator ind2_start,
                   DataIt data2_start, std::size_t ndim,
                   std::size_t n_features) {

  std::size_t i1 = 0;
  std::size_t i2 = 0;
  std::size_t num_not_equal = 0;

  while (i1 < ndim && i2 < ndim) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      if (*(data1_start + i1) != *(data2_start + i2)) {
        ++num_not_equal;
      }
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      ++num_not_equal;
      ++i1;
    } else {
      ++num_not_equal;
      ++i2;
    }
  }

  while (i1 < ndim) {
    ++num_not_equal;
    ++i1;
  }

  while (i2 < ndim) {
    ++num_not_equal;
    ++i2;
  }

  // Normalize by the total number of features
  return static_cast<Out>(num_not_equal) / n_features;
}

#include <cmath>
#include <cstddef>
#include <vector>

template <typename Out, typename DataIt>
Out sparse_cosine(typename std::vector<std::size_t>::const_iterator ind1_start,
                  DataIt data1_start,
                  typename std::vector<std::size_t>::const_iterator ind2_start,
                  DataIt data2_start, std::size_t ndim) {

  Out dot_product{0};
  Out norm1{0};
  Out norm2{0};

  std::size_t i1 = 0;
  std::size_t i2 = 0;

  // pass through both index lists
  while (i1 < ndim && i2 < ndim) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      const auto val1 = static_cast<Out>(*(data1_start + i1));
      const auto val2 = static_cast<Out>(*(data2_start + i2));
      dot_product += val1 * val2;
      norm1 += val1 * val1;
      norm2 += val2 * val2;
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      const auto val1 = static_cast<Out>(*(data1_start + i1));
      norm1 += val1 * val1;
      ++i1;
    } else {
      const auto val2 = static_cast<Out>(*(data2_start + i2));
      norm2 += val2 * val2;
      ++i2;
    }
  }

  while (i1 < ndim) {
    const auto val1 = static_cast<Out>(*(data1_start + i1));
    norm1 += val1 * val1;
    ++i1;
  }

  while (i2 < ndim) {
    const auto val2 = static_cast<Out>(*(data2_start + i2));
    norm2 += val2 * val2;
    ++i2;
  }

  norm1 = std::sqrt(norm1);
  norm2 = std::sqrt(norm2);

  if (norm1 == 0.0 && norm2 == 0.0) {
    return Out(0);
  } else if (norm1 == 0.0 || norm2 == 0.0) {
    return Out(1);
  } else {
    return Out(1) - (dot_product / (norm1 * norm2));
  }
}

template <typename Out, typename DataIt>
Out sparse_correlation(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    DataIt data2_start, std::size_t ndim, std::size_t n_features) {
  Out mu_x{0};
  Out mu_y{0};
  Out dot_product{0};

  if (ndim == 0) {
    return (n_features == 0) ? Out(0) : Out(1);
  }

  for (std::size_t i = 0; i < ndim; ++i) {
    mu_x += *(data1_start + i);
    mu_y += *(data2_start + i);
  }
  mu_x /= n_features;
  mu_y /= n_features;

  std::vector<Out> shifted_data1(ndim);
  std::vector<Out> shifted_data2(ndim);
  for (std::size_t i = 0; i < ndim; ++i) {
    shifted_data1[i] = *(data1_start + i) - mu_x;
    shifted_data2[i] = *(data2_start + i) - mu_y;
  }

  Out norm1 =
      std::sqrt(std::inner_product(shifted_data1.begin(), shifted_data1.end(),
                                   shifted_data1.begin(), Out(0)) +
                (n_features - ndim) * mu_x * mu_x);
  Out norm2 =
      std::sqrt(std::inner_product(shifted_data2.begin(), shifted_data2.end(),
                                   shifted_data2.begin(), Out(0)) +
                (n_features - ndim) * mu_y * mu_y);

  auto dot_prod = sparse_mul(ind1_start, shifted_data1.begin(), ind2_start,
                             shifted_data2.begin(), ndim);
  auto &dot_prod_inds = dot_prod.first;
  auto &dot_prod_data = dot_prod.second;

  std::unordered_set<std::size_t> common_indices(dot_prod_inds.begin(),
                                                 dot_prod_inds.end());

  for (auto val : dot_prod_data) {
    dot_product += val;
  }

  for (std::size_t i = 0; i < ndim; ++i) {
    if (common_indices.find(*(ind1_start + i)) == common_indices.end()) {
      dot_product -= shifted_data1[i] * mu_y;
    }
    if (common_indices.find(*(ind2_start + i)) == common_indices.end()) {
      dot_product -= shifted_data2[i] * mu_x;
    }
  }

  auto all_indices =
      arr_union(std::vector<std::size_t>(ind1_start, ind1_start + ndim),
                std::vector<std::size_t>(ind2_start, ind2_start + ndim));
  dot_product += mu_x * mu_y * (n_features - all_indices.size());

  if (norm1 == 0.0 && norm2 == 0.0) {
    return Out(0);
  } else if (dot_product == 0.0) {
    return Out(1);
  } else {
    return Out(1) - (dot_product / (norm1 * norm2));
  }
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

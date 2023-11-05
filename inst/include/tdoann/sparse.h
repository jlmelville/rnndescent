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
#include <unordered_set>
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

template <typename Out, typename DataIt>
std::pair<std::vector<std::size_t>, std::vector<Out>>
sparse_sum(typename std::vector<std::size_t>::const_iterator ind1_start,
           std::size_t ind1_size, DataIt data1_start,
           typename std::vector<std::size_t>::const_iterator ind2_start,
           std::size_t ind2_size, DataIt data2_start) {

  std::vector<std::size_t> result_ind;
  result_ind.reserve(ind1_size + ind2_size);

  std::vector<Out> result_data;
  result_data.reserve(ind1_size + ind2_size);

  std::size_t i1 = 0;
  std::size_t i2 = 0;

  constexpr Out zero(0);

  // Pass through both index lists
  while (i1 < ind1_size && i2 < ind2_size) {
    auto j1 = *(ind1_start + i1);
    auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      auto val = *(data1_start + i1) + *(data2_start + i2);
      if (val != zero) {
        result_ind.push_back(j1);
        result_data.push_back(val);
      }
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      auto val = *(data1_start + i1);
      if (val != zero) {
        result_ind.push_back(j1);
        result_data.push_back(val);
      }
      ++i1;
    } else {
      auto val = *(data2_start + i2);
      if (val != zero) {
        result_ind.push_back(j2);
        result_data.push_back(val);
      }
      ++i2;
    }
  }

  // tails
  while (i1 < ind1_size) {
    auto j1 = *(ind1_start + i1);
    auto val = *(data1_start + i1);
    if (val != zero) {
      result_ind.push_back(j1);
      result_data.push_back(val);
    }
    ++i1;
  }

  while (i2 < ind2_size) {
    auto j2 = *(ind2_start + i2);
    auto val = *(data2_start + i2);
    if (val != zero) {
      result_ind.push_back(j2);
      result_data.push_back(val);
    }
    ++i2;
  }

  return {result_ind, result_data};
}

template <typename Out, typename DataIt>
std::pair<std::vector<std::size_t>, std::vector<Out>>
sparse_diff(typename std::vector<std::size_t>::const_iterator ind1_start,
            std::size_t ind1_size, DataIt data1_start,
            typename std::vector<std::size_t>::const_iterator ind2_start,
            std::size_t ind2_size, DataIt data2_start) {

  std::vector<std::size_t> result_ind;
  result_ind.reserve(ind1_size + ind2_size);

  std::vector<Out> result_data;
  result_data.reserve(ind1_size + ind2_size);

  std::size_t i1 = 0;
  std::size_t i2 = 0;

  constexpr Out zero(0);

  while (i1 < ind1_size && i2 < ind2_size) {
    auto j1 = *(ind1_start + i1);
    auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      auto val = *(data1_start + i1) - *(data2_start + i2);
      if (val != zero) {
        result_ind.push_back(j1);
        result_data.push_back(val);
      }
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      auto val = *(data1_start + i1);
      if (val != zero) {
        result_ind.push_back(j1);
        result_data.push_back(val);
      }
      ++i1;
    } else {
      auto val = -*(data2_start + i2); // negation
      if (val != zero) {
        result_ind.push_back(j2);
        result_data.push_back(val);
      }
      ++i2;
    }
  }

  // tails
  while (i1 < ind1_size) {
    auto j1 = *(ind1_start + i1);
    auto val = *(data1_start + i1);
    if (val != zero) {
      result_ind.push_back(j1);
      result_data.push_back(val);
    }
    ++i1;
  }

  while (i2 < ind2_size) {
    auto j2 = *(ind2_start + i2);
    auto val = -*(data2_start + i2); // negation
    if (val != zero) {
      result_ind.push_back(j2);
      result_data.push_back(val);
    }
    ++i2;
  }

  return {result_ind, result_data};
}

template <typename Out, typename DataIt>
std::pair<std::vector<std::size_t>, std::vector<Out>>
sparse_mul(typename std::vector<std::size_t>::const_iterator ind1_start,
           std::size_t ind1_size, DataIt data1_start,
           typename std::vector<std::size_t>::const_iterator ind2_start,
           std::size_t ind2_size, DataIt data2_start) {
  std::vector<std::size_t> result_ind;
  std::vector<Out> result_data;

  std::size_t i1 = 0;
  std::size_t i2 = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      auto val = static_cast<Out>(*(data1_start + i1) * *(data2_start + i2));
      if (val != Out(0)) {
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
Out sparse_l2sqr(typename std::vector<std::size_t>::const_iterator ind1_start,
                 std::size_t ind1_size, DataIt data1_start,
                 typename std::vector<std::size_t>::const_iterator ind2_start,
                 std::size_t ind2_size, DataIt data2_start) {
  Out sum{0};

  std::size_t i1 = 0;
  std::size_t i2 = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
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
  while (i1 < ind1_size) {
    auto val = static_cast<Out>(*(data1_start + i1));
    sum += val * val;
    ++i1;
  }

  while (i2 < ind2_size) {
    auto val = static_cast<Out>(*(data2_start + i2));
    sum += val * val;
    ++i2;
  }

  return sum;
}

template <typename Out, typename DataIt>
Out sparse_euclidean(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start) {
  return std::sqrt(sparse_l2sqr<Out>(ind1_start, ind1_size, data1_start,
                                     ind2_start, ind2_size, data2_start));
}

template <typename Out, typename DataIt>
Out sparse_manhattan(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start) {

  Out result = Out();

  std::size_t i1 = 0;
  std::size_t i2 = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
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

  while (i1 < ind1_size) {
    result += std::abs(static_cast<Out>(*(data1_start + i1)));
    ++i1;
  }

  while (i2 < ind2_size) {
    result += std::abs(static_cast<Out>(*(data2_start + i2)));
    ++i2;
  }

  return result;
}

template <typename Out, typename DataIt>
Out sparse_hamming(typename std::vector<std::size_t>::const_iterator ind1_start,
                   std::size_t ind1_size, DataIt data1_start,
                   typename std::vector<std::size_t>::const_iterator ind2_start,
                   std::size_t ind2_size, DataIt data2_start) {

  std::size_t i1 = 0;
  std::size_t i2 = 0;
  std::size_t num_not_equal = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
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

  while (i1 < ind1_size) {
    ++num_not_equal;
    ++i1;
  }

  while (i2 < ind2_size) {
    ++num_not_equal;
    ++i2;
  }

  return num_not_equal;
}

template <typename Out, typename DataIt>
Out sparse_cosine(typename std::vector<std::size_t>::const_iterator ind1_start,
                  std::size_t ind1_size, DataIt data1_start,
                  typename std::vector<std::size_t>::const_iterator ind2_start,
                  std::size_t ind2_size, DataIt data2_start) {

  Out dot_product{0};
  Out norm1{0};
  Out norm2{0};

  std::size_t i1 = 0;
  std::size_t i2 = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
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

  // Handle remaining elements
  while (i1 < ind1_size) {
    const auto val1 = static_cast<Out>(*(data1_start + i1));
    norm1 += val1 * val1;
    ++i1;
  }

  while (i2 < ind2_size) {
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
Out sparse_alternative_cosine(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start) {

  const Out FLOAT32_MAX = std::numeric_limits<float>::max();

  auto sparse_mul_result = sparse_mul<Out>(ind1_start, ind1_size, data1_start,
                                           ind2_start, ind2_size, data2_start);
  auto &aux_data = sparse_mul_result.second;

  Out result{0};
  Out norm_x{0};
  Out norm_y{0};
  std::size_t dim = aux_data.size();

  for (std::size_t i = 0; i < ind1_size; ++i) {
    Out val = static_cast<Out>(*(data1_start + i));
    norm_x += val * val;
  }

  for (std::size_t i = 0; i < ind2_size; ++i) {
    Out val = static_cast<Out>(*(data2_start + i));
    norm_y += val * val;
  }

  norm_x = std::sqrt(norm_x);
  norm_y = std::sqrt(norm_y);

  for (std::size_t i = 0; i < dim; ++i) {
    result += aux_data[i];
  }

  if (norm_x == 0.0 && norm_y == 0.0) {
    return Out(0);
  } else if (norm_x == 0.0 || norm_y == 0.0) {
    return FLOAT32_MAX;
  } else if (result <= 0.0) {
    return FLOAT32_MAX;
  } else {
    result = (norm_x * norm_y) / result;
    return std::log2(result);
  }
}

template <typename Out, typename DataIt>
Out sparse_correlation(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t ndim) {

  Out mu_x{0};
  Out mu_y{0};
  Out dot_product{0};

  if (ind1_size == 0 && ind2_size == 0) {
    return (ndim == 0) ? Out(0) : Out(1);
  }

  for (std::size_t i = 0; i < ind1_size; ++i) {
    mu_x += *(data1_start + i);
  }
  for (std::size_t i = 0; i < ind2_size; ++i) {
    mu_y += *(data2_start + i);
  }

  mu_x /= ndim;
  mu_y /= ndim;

  std::vector<Out> shifted_data1(ind1_size);
  std::vector<Out> shifted_data2(ind2_size);

  for (std::size_t i = 0; i < ind1_size; ++i) {
    shifted_data1[i] = *(data1_start + i) - mu_x;
  }
  for (std::size_t i = 0; i < ind2_size; ++i) {
    shifted_data2[i] = *(data2_start + i) - mu_y;
  }

  Out norm1 =
      std::sqrt(std::inner_product(shifted_data1.begin(), shifted_data1.end(),
                                   shifted_data1.begin(), Out(0)) +
                (ndim - ind1_size) * mu_x * mu_x);
  Out norm2 =
      std::sqrt(std::inner_product(shifted_data2.begin(), shifted_data2.end(),
                                   shifted_data2.begin(), Out(0)) +
                (ndim - ind2_size) * mu_y * mu_y);

  auto dot_prod = sparse_mul<Out>(ind1_start, ind1_size, shifted_data1.begin(),
                                  ind2_start, ind2_size, shifted_data2.begin());
  auto &dot_prod_inds = dot_prod.first;
  auto &dot_prod_data = dot_prod.second;

  std::unordered_set<std::size_t> common_indices(dot_prod_inds.begin(),
                                                 dot_prod_inds.end());

  for (auto val : dot_prod_data) {
    dot_product += val;
  }

  for (std::size_t i = 0; i < ind1_size; ++i) {
    if (common_indices.find(*(ind1_start + i)) == common_indices.end()) {
      dot_product -= shifted_data1[i] * mu_y;
    }
  }
  for (std::size_t i = 0; i < ind2_size; ++i) {
    if (common_indices.find(*(ind2_start + i)) == common_indices.end()) {
      dot_product -= shifted_data2[i] * mu_x;
    }
  }

  auto all_indices =
      arr_union(std::vector<std::size_t>(ind1_start, ind1_start + ind1_size),
                std::vector<std::size_t>(ind2_start, ind2_start + ind2_size));
  dot_product += mu_x * mu_y * (ndim - all_indices.size());

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
                          std::vector<std::size_t> &&x_ptr,
                          std::vector<In> &&x_data, std::size_t nx,
                          std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), ndim(ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->x_ptr[j];

    return sparse_l2sqr<Out>(this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
                             this->x_data.begin() + di,
                             this->x_ind.begin() + dj, this->x_ptr[j + 1] - dj,
                             this->x_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return nx; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class SparseEuclideanSelfDistance : public BaseDistance<Out, Idx> {
public:
  SparseEuclideanSelfDistance(std::vector<std::size_t> &&x_ind,
                              std::vector<std::size_t> &&x_ptr,
                              std::vector<In> &&x_data, std::size_t nx,
                              std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), ndim(ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->x_ptr[j];

    return sparse_euclidean<Out>(
        this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
        this->x_data.begin() + di, this->x_ind.begin() + dj,
        this->x_ptr[j + 1] - dj, this->x_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return nx; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class SparseManhattanSelfDistance : public BaseDistance<Out, Idx> {
public:
  SparseManhattanSelfDistance(std::vector<std::size_t> &&x_ind,
                              std::vector<std::size_t> &&x_ptr,
                              std::vector<In> &&x_data, std::size_t nx,
                              std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), ndim(ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->x_ptr[j];

    return sparse_manhattan<Out>(
        this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
        this->x_data.begin() + di, this->x_ind.begin() + dj,
        this->x_ptr[j + 1] - dj, this->x_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return nx; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class SparseHammingSelfDistance : public BaseDistance<Out, Idx> {
public:
  SparseHammingSelfDistance(std::vector<std::size_t> &&x_ind,
                            std::vector<std::size_t> &&x_ptr,
                            std::vector<In> &&x_data, std::size_t nx,
                            std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), ndim(ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->x_ptr[j];

    return sparse_hamming<Out>(
        this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
        this->x_data.begin() + di, this->x_ind.begin() + dj,
        this->x_ptr[j + 1] - dj, this->x_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return nx; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class SparseCosineSelfDistance : public BaseDistance<Out, Idx> {
public:
  SparseCosineSelfDistance(std::vector<std::size_t> &&x_ind,
                           std::vector<std::size_t> &&x_ptr,
                           std::vector<In> &&x_data, std::size_t nx,
                           std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), ndim(ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->x_ptr[j];

    return sparse_cosine<Out>(this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
                              this->x_data.begin() + di,
                              this->x_ind.begin() + dj, this->x_ptr[j + 1] - dj,
                              this->x_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return nx; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class SparseAlternativeCosineSelfDistance : public BaseDistance<Out, Idx> {
public:
  SparseAlternativeCosineSelfDistance(std::vector<std::size_t> &&x_ind,
                                      std::vector<std::size_t> &&x_ptr,
                                      std::vector<In> &&x_data, std::size_t nx,
                                      std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), ndim(ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->x_ptr[j];

    return sparse_alternative_cosine<Out>(
        this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
        this->x_data.begin() + di, this->x_ind.begin() + dj,
        this->x_ptr[j + 1] - dj, this->x_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return nx; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class SparseCorrelationSelfDistance : public BaseDistance<Out, Idx> {
public:
  SparseCorrelationSelfDistance(std::vector<std::size_t> &&x_ind,
                                std::vector<std::size_t> &&x_ptr,
                                std::vector<In> &&x_data, std::size_t nx,
                                std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), ndim(ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->x_ptr[j];

    return sparse_correlation<Out>(
        this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
        this->x_data.begin() + di, this->x_ind.begin() + dj,
        this->x_ptr[j + 1] - dj, this->x_data.begin() + dj, this->ndim);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return nx; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class SparseL2SqrQueryDistance : public BaseDistance<Out, Idx> {
public:
  SparseL2SqrQueryDistance(std::vector<std::size_t> &&x_ind,
                           std::vector<std::size_t> &&x_ptr,
                           std::vector<In> &&x_data, std::size_t nx,
                           std::vector<std::size_t> &&y_ind,
                           std::vector<std::size_t> &&y_ptr,
                           std::vector<In> &&y_data, std::size_t ny,
                           std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), y_ind(std::move(y_ind)),
        y_ptr(std::move(y_ptr)), y_data(std::move(y_data)), ny(ny), ndim(ndim) {
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->y_ptr[j];

    return sparse_l2sqr<Out>(this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
                             this->x_data.begin() + di,
                             this->y_ind.begin() + dj, this->y_ptr[j + 1] - dj,
                             this->y_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return ny; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::vector<std::size_t> y_ind;
  std::vector<std::size_t> y_ptr;
  std::vector<In> y_data;
  std::size_t ny;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class SparseEuclideanQueryDistance : public BaseDistance<Out, Idx> {
public:
  SparseEuclideanQueryDistance(std::vector<std::size_t> &&x_ind,
                               std::vector<std::size_t> &&x_ptr,
                               std::vector<In> &&x_data, std::size_t nx,
                               std::vector<std::size_t> &&y_ind,
                               std::vector<std::size_t> &&y_ptr,
                               std::vector<In> &&y_data, std::size_t ny,
                               std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), y_ind(std::move(y_ind)),
        y_ptr(std::move(y_ptr)), y_data(std::move(y_data)), ny(ny), ndim(ndim) {
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->y_ptr[j];

    return sparse_euclidean<Out>(
        this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
        this->x_data.begin() + di, this->y_ind.begin() + dj,
        this->y_ptr[j + 1] - dj, this->y_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return ny; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::vector<std::size_t> y_ind;
  std::vector<std::size_t> y_ptr;
  std::vector<In> y_data;
  std::size_t ny;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class SparseManhattanQueryDistance : public BaseDistance<Out, Idx> {
public:
  SparseManhattanQueryDistance(std::vector<std::size_t> &&x_ind,
                               std::vector<std::size_t> &&x_ptr,
                               std::vector<In> &&x_data, std::size_t nx,
                               std::vector<std::size_t> &&y_ind,
                               std::vector<std::size_t> &&y_ptr,
                               std::vector<In> &&y_data, std::size_t ny,
                               std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), y_ind(std::move(y_ind)),
        y_ptr(std::move(y_ptr)), y_data(std::move(y_data)), ny(ny), ndim(ndim) {
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->y_ptr[j];

    return sparse_manhattan<Out>(
        this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
        this->x_data.begin() + di, this->y_ind.begin() + dj,
        this->y_ptr[j + 1] - dj, this->y_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return ny; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::vector<std::size_t> y_ind;
  std::vector<std::size_t> y_ptr;
  std::vector<In> y_data;
  std::size_t ny;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class SparseHammingQueryDistance : public BaseDistance<Out, Idx> {
public:
  SparseHammingQueryDistance(std::vector<std::size_t> &&x_ind,
                             std::vector<std::size_t> &&x_ptr,
                             std::vector<In> &&x_data, std::size_t nx,
                             std::vector<std::size_t> &&y_ind,
                             std::vector<std::size_t> &&y_ptr,
                             std::vector<In> &&y_data, std::size_t ny,
                             std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), y_ind(std::move(y_ind)),
        y_ptr(std::move(y_ptr)), y_data(std::move(y_data)), ny(ny), ndim(ndim) {
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->y_ptr[j];

    return sparse_hamming<Out>(
        this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
        this->x_data.begin() + di, this->y_ind.begin() + dj,
        this->y_ptr[j + 1] - dj, this->y_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return ny; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::vector<std::size_t> y_ind;
  std::vector<std::size_t> y_ptr;
  std::vector<In> y_data;
  std::size_t ny;
  std::size_t ndim;
};
template <typename In, typename Out, typename Idx = uint32_t>
class SparseCosineQueryDistance : public BaseDistance<Out, Idx> {
public:
  SparseCosineQueryDistance(std::vector<std::size_t> &&x_ind,
                            std::vector<std::size_t> &&x_ptr,
                            std::vector<In> &&x_data, std::size_t nx,
                            std::vector<std::size_t> &&y_ind,
                            std::vector<std::size_t> &&y_ptr,
                            std::vector<In> &&y_data, std::size_t ny,
                            std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), y_ind(std::move(y_ind)),
        y_ptr(std::move(y_ptr)), y_data(std::move(y_data)), ny(ny), ndim(ndim) {
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->y_ptr[j];

    return sparse_cosine<Out>(this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
                              this->x_data.begin() + di,
                              this->y_ind.begin() + dj, this->y_ptr[j + 1] - dj,
                              this->y_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return ny; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::vector<std::size_t> y_ind;
  std::vector<std::size_t> y_ptr;
  std::vector<In> y_data;
  std::size_t ny;
  std::size_t ndim;
};
template <typename In, typename Out, typename Idx = uint32_t>
class SparseAlternativeCosineQueryDistance : public BaseDistance<Out, Idx> {
public:
  SparseAlternativeCosineQueryDistance(std::vector<std::size_t> &&x_ind,
                                       std::vector<std::size_t> &&x_ptr,
                                       std::vector<In> &&x_data, std::size_t nx,
                                       std::vector<std::size_t> &&y_ind,
                                       std::vector<std::size_t> &&y_ptr,
                                       std::vector<In> &&y_data, std::size_t ny,
                                       std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), y_ind(std::move(y_ind)),
        y_ptr(std::move(y_ptr)), y_data(std::move(y_data)), ny(ny), ndim(ndim) {
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->y_ptr[j];

    return sparse_alternative_cosine<Out>(
        this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
        this->x_data.begin() + di, this->y_ind.begin() + dj,
        this->y_ptr[j + 1] - dj, this->y_data.begin() + dj);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return ny; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::vector<std::size_t> y_ind;
  std::vector<std::size_t> y_ptr;
  std::vector<In> y_data;
  std::size_t ny;
  std::size_t ndim;
};
template <typename In, typename Out, typename Idx = uint32_t>
class SparseCorrelationQueryDistance : public BaseDistance<Out, Idx> {
public:
  SparseCorrelationQueryDistance(std::vector<std::size_t> &&x_ind,
                                 std::vector<std::size_t> &&x_ptr,
                                 std::vector<In> &&x_data, std::size_t nx,
                                 std::vector<std::size_t> &&y_ind,
                                 std::vector<std::size_t> &&y_ptr,
                                 std::vector<In> &&y_data, std::size_t ny,
                                 std::size_t ndim)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(nx), y_ind(std::move(y_ind)),
        y_ptr(std::move(y_ptr)), y_data(std::move(y_data)), ny(ny), ndim(ndim) {
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->x_ptr[i];
    const std::size_t dj = this->y_ptr[j];

    return sparse_correlation<Out>(
        this->x_ind.begin() + di, this->x_ptr[i + 1] - di,
        this->x_data.begin() + di, this->y_ind.begin() + dj,
        this->y_ptr[j + 1] - dj, this->y_data.begin() + dj, this->ndim);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return ny; }

private:
  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::vector<std::size_t> y_ind;
  std::vector<std::size_t> y_ptr;
  std::vector<In> y_data;
  std::size_t ny;
  std::size_t ndim;
};

} // namespace tdoann

#endif // TDOANN_SPARSE_H

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

#include "distance.h"
#include "distancebase.h"

namespace tdoann {

template <typename Out, typename DataIt>
Out sparse_squared_euclidean(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */);

inline std::vector<std::size_t> arr_union(const std::vector<std::size_t> &ar1,
                                          const std::vector<std::size_t> &ar2) {
  std::vector<std::size_t> union_result;
  union_result.reserve(ar1.size() + ar2.size());

  std::set_union(ar1.begin(), ar1.end(), ar2.begin(), ar2.end(),
                 std::back_inserter(union_result));

  return union_result;
}

template <typename It>
std::size_t fast_intersection_size(It ind1_start, std::size_t ind1_size,
                                   It ind2_start, std::size_t ind2_size) {
  if (ind1_size == 0 || ind2_size == 0) {
    return 0;
  }

  It i1 = ind1_start;
  It i2 = ind2_start;
  std::size_t result = 0;

  while (i1 < ind1_start + ind1_size && i2 < ind2_start + ind2_size) {
    if (*i1 == *i2) {
      ++result;
      ++i1;
      ++i2;
    } else if (*i1 < *i2) {
      ++i1;
    } else {
      ++i2;
    }
  }

  return result;
}

template <typename In, typename DataIt>
std::pair<std::vector<In>, std::vector<In>>
dense_union(typename std::vector<std::size_t>::const_iterator ind1_start,
            std::size_t ind1_size, DataIt data1_start,
            typename std::vector<std::size_t>::const_iterator ind2_start,
            std::size_t ind2_size, DataIt data2_start) {

  std::vector<In> result_data1, result_data2;

  auto i1 = ind1_start;
  auto d1 = data1_start;
  auto i2 = ind2_start;
  auto d2 = data2_start;
  while (i1 < ind1_start + ind1_size && i2 < ind2_start + ind2_size) {
    if (*i1 == *i2) {
      result_data1.push_back(*d1);
      result_data2.push_back(*d2);
      ++i1;
      ++d1;
      ++i2;
      ++d2;
    } else if (*i1 < *i2) {
      result_data1.push_back(*d1);
      result_data2.push_back(In{});
      ++i1;
      ++d1;
    } else {
      result_data1.push_back(In{});
      result_data2.push_back(*d2);
      ++i2;
      ++d2;
    }
  }

  // Process remaining elements
  while (i1 < ind1_start + ind1_size) {
    result_data1.push_back(*d1);
    result_data2.push_back(In{});
    ++i1;
    ++d1;
  }
  while (i2 < ind2_start + ind2_size) {
    result_data1.push_back(In{});
    result_data2.push_back(*d2);
    ++i2;
    ++d2;
  }

  return {result_data1, result_data2};
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
Out sparse_bray_curtis(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {

  double numerator = 0.0;
  double denominator = 0.0;
  std::size_t i1 = 0;
  std::size_t i2 = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      auto val1 = *(data1_start + i1);
      auto val2 = *(data2_start + i2);
      numerator += std::abs(val1 - val2);
      denominator += std::abs(val1 + val2);
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      numerator += std::abs(*(data1_start + i1));
      denominator += std::abs(*(data1_start + i1));
      ++i1;
    } else {
      numerator += std::abs(*(data2_start + i2));
      denominator += std::abs(*(data2_start + i2));
      ++i2;
    }
  }

  while (i1 < ind1_size) {
    numerator += std::abs(*(data1_start + i1));
    denominator += std::abs(*(data1_start + i1));
    ++i1;
  }

  while (i2 < ind2_size) {
    numerator += std::abs(*(data2_start + i2));
    denominator += std::abs(*(data2_start + i2));
    ++i2;
  }

  return denominator == 0.0
             ? Out{}
             : static_cast<Out>(numerator) / static_cast<Out>(denominator);
}

template <typename Out, typename DataIt>
Out sparse_canberra(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {

  std::size_t i1 = 0, i2 = 0;
  Out result = 0.0;

  while (i1 < ind1_size && i2 < ind2_size) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      auto abs_val1 = std::abs(*(data1_start + i1));
      auto abs_val2 = std::abs(*(data2_start + i2));
      auto denom = abs_val1 + abs_val2;
      if (denom > Out{}) {
        result += std::abs(abs_val1 - abs_val2) / denom;
      }
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      if (*(data1_start + i1) != Out{}) {
        result += 1.0;
      }
      ++i1;
    } else {
      if (*(data2_start + i2) != Out{}) {
        result += 1.0;
      }
      ++i2;
    }
  }

  while (i1 < ind1_size) {
    if (*(data1_start + i1) != Out{}) {
      result += 1.0;
    }
    ++i1;
  }

  while (i2 < ind2_size) {
    if (*(data2_start + i2) != Out{}) {
      result += 1.0;
    }
    ++i2;
  }

  return result;
}

template <typename Out, typename DataIt>
Out sparse_chebyshev(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {

  std::size_t i1 = 0, i2 = 0;
  Out max_diff = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      max_diff = std::max(max_diff,
                          std::abs(*(data1_start + i1) - *(data2_start + i2)));
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      max_diff = std::max(max_diff, std::abs(*(data1_start + i1)));
      ++i1;
    } else {
      max_diff = std::max(max_diff, std::abs(*(data2_start + i2)));
      ++i2;
    }
  }

  while (i1 < ind1_size) {
    max_diff = std::max(max_diff, std::abs(*(data1_start + i1)));
    ++i1;
  }

  while (i2 < ind2_size) {
    max_diff = std::max(max_diff, std::abs(*(data2_start + i2)));
    ++i2;
  }

  return max_diff;
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
template <typename Out, typename DataIt>
Out sparse_cosine(typename std::vector<std::size_t>::const_iterator ind1_start,
                  std::size_t ind1_size, DataIt data1_start,
                  typename std::vector<std::size_t>::const_iterator ind2_start,
                  std::size_t ind2_size, DataIt data2_start,
                  std::size_t /* ndim */) {

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
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {

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
Out sparse_dice(typename std::vector<std::size_t>::const_iterator ind1_start,
                std::size_t ind1_size, DataIt /* data1_start */,
                typename std::vector<std::size_t>::const_iterator ind2_start,
                std::size_t ind2_size, DataIt /* data2_start */,
                std::size_t /* ndim */) {

  std::size_t num_true_true =
      fast_intersection_size(ind1_start, ind1_size, ind2_start, ind2_size);
  std::size_t num_non_zero = ind1_size + ind2_size - num_true_true;
  std::size_t num_not_equal = num_non_zero - num_true_true;

  if (num_not_equal == 0) {
    return Out{};
  } else {
    return static_cast<Out>(static_cast<double>(num_not_equal) /
                            (2 * num_true_true + num_not_equal));
  }
}

template <typename Out, typename DataIt>
Out sparse_dot(typename std::vector<std::size_t>::const_iterator ind1_start,
               std::size_t ind1_size, DataIt data1_start,
               typename std::vector<std::size_t>::const_iterator ind2_start,
               std::size_t ind2_size, DataIt data2_start,
               std::size_t /* ndim */) {

  Out result = 0;
  std::size_t i1 = 0;
  std::size_t i2 = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      result += static_cast<Out>(*(data1_start + i1) * *(data2_start + i2));
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      ++i1;
    } else {
      ++i2;
    }
  }

  if (result <= Out{}) {
    return 1.0;
  } else {
    return 1.0 - result;
  }
}

template <typename Out, typename DataIt>
auto sparse_alternative_dot(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {
  Out result = 0;
  std::size_t i1 = 0;
  std::size_t i2 = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      result += static_cast<Out>(*(data1_start + i1) * *(data2_start + i2));
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      ++i1;
    } else {
      ++i2;
    }
  }

  if (result <= 0.0) {
    return std::numeric_limits<Out>::max();
  }

  return -std::log2(result);
}

template <typename Out, typename DataIt>
Out sparse_euclidean(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t ndim) {
  return std::sqrt(sparse_squared_euclidean<Out>(ind1_start, ind1_size,
                                                 data1_start, ind2_start,
                                                 ind2_size, data2_start, ndim));
}

template <typename Out, typename DataIt>
Out sparse_hamming(typename std::vector<std::size_t>::const_iterator ind1_start,
                   std::size_t ind1_size, DataIt data1_start,
                   typename std::vector<std::size_t>::const_iterator ind2_start,
                   std::size_t ind2_size, DataIt data2_start,
                   std::size_t ndim) {
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

  num_not_equal += (ind1_size - i1) + (ind2_size - i2);

  return static_cast<Out>(static_cast<double>(num_not_equal) / ndim);
}

template <typename Out, typename DataIt>
Out sparse_hellinger(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {

  double result = 0.0;
  double l1_norm_x = 0.0;
  double l1_norm_y = 0.0;
  std::size_t i1 = 0;
  std::size_t i2 = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      result += std::sqrt(*(data1_start + i1) * *(data2_start + i2));
      l1_norm_x += *(data1_start + i1);
      l1_norm_y += *(data2_start + i2);
      ++i1;
      ++i2;
    } else {
      if (j1 < j2) {
        l1_norm_x += *(data1_start + i1);
        ++i1;
      } else {
        l1_norm_y += *(data2_start + i2);
        ++i2;
      }
    }
  }

  while (i1 < ind1_size) {
    l1_norm_x += *(data1_start + i1);
    ++i1;
  }

  while (i2 < ind2_size) {
    l1_norm_y += *(data2_start + i2);
    ++i2;
  }

  if (l1_norm_x == 0 && l1_norm_y == 0) {
    return Out{};
  } else if (l1_norm_x == 0 || l1_norm_y == 0) {
    return static_cast<Out>(1.0);
  } else {
    return std::sqrt(static_cast<Out>(1.0) -
                     result / std::sqrt(l1_norm_x * l1_norm_y));
  }
}

template <typename Out, typename DataIt>
Out sparse_alternative_hellinger(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {

  double result = 0.0;
  double l1_norm_x = 0.0;
  double l1_norm_y = 0.0;

  // Calculate L1 norms
  for (std::size_t i = 0; i < ind1_size; ++i) {
    l1_norm_x += *(data1_start + i);
  }
  for (std::size_t j = 0; j < ind2_size; ++j) {
    l1_norm_y += *(data2_start + j);
  }

  // Calculate the product and its square root sum
  std::size_t i1 = 0, i2 = 0;
  while (i1 < ind1_size && i2 < ind2_size) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      result += std::sqrt(*(data1_start + i1) * *(data2_start + i2));
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      ++i1;
    } else {
      ++i2;
    }
  }

  if (l1_norm_x == 0 && l1_norm_y == 0) {
    return Out{};
  } else if (l1_norm_x == 0 || l1_norm_y == 0 || result <= 0) {
    return std::numeric_limits<Out>::max();
  } else {
    result = std::sqrt(l1_norm_x * l1_norm_y) / result;
    return static_cast<Out>(std::log2(result));
  }
}

template <typename Out, typename DataIt>
Out sparse_jaccard(typename std::vector<std::size_t>::const_iterator ind1_start,
                   std::size_t ind1_size, DataIt /* data1_start */,
                   typename std::vector<std::size_t>::const_iterator ind2_start,
                   std::size_t ind2_size, DataIt /* data2_start */,
                   std::size_t /* ndim */) {

  std::size_t num_equal =
      fast_intersection_size(ind1_start, ind1_size, ind2_start, ind2_size);
  std::size_t num_non_zero = ind1_size + ind2_size - num_equal;

  if (num_non_zero == 0) {
    return Out{};
  } else {
    return static_cast<Out>(static_cast<double>(num_non_zero - num_equal) /
                            num_non_zero);
  }
}

template <typename Out, typename DataIt>
Out sparse_alternative_jaccard(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt /* data1_start */,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt /* data2_start */, std::size_t /* ndim */) {

  std::size_t num_equal =
      fast_intersection_size(ind1_start, ind1_size, ind2_start, ind2_size);
  std::size_t num_non_zero = ind1_size + ind2_size - num_equal;

  if (num_non_zero == 0) {
    return Out{};
  } else if (num_equal == 0) {
    return std::numeric_limits<Out>::max();
  } else {
    return static_cast<Out>(
        -std::log2(static_cast<double>(num_equal) / num_non_zero));
  }
}

template <typename Out, typename DataIt>
Out sparse_jensen_shannon_divergence(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {

  auto [dense_data1, dense_data2] = dense_union<Out, DataIt>(
      ind1_start, ind1_size, data1_start, ind2_start, ind2_size, data2_start);

  return jensen_shannon_divergence<Out>(dense_data1.begin(), dense_data1.end(),
                                        dense_data2.begin());
}

template <typename Out, typename DataIt>
Out sparse_kulsinski(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt /* data1_start */,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt /* data2_start */, std::size_t ndim) {

  std::size_t num_true_true =
      fast_intersection_size(ind1_start, ind1_size, ind2_start, ind2_size);
  std::size_t num_non_zero = ind1_size + ind2_size - num_true_true;
  std::size_t num_not_equal = num_non_zero - num_true_true;

  if (num_not_equal == 0) {
    return Out{};
  } else {
    return static_cast<Out>(
        static_cast<double>(num_not_equal - num_true_true + ndim) /
        (num_not_equal + ndim));
  }
}

template <typename DataIt>
std::pair<std::vector<double>, double>
sparse_rankdata(typename std::vector<std::size_t>::const_iterator ind_start,
                std::size_t ind_size, DataIt data_start, std::size_t ndim) {
  // Rank the non-zero data using dense rankdata function
  auto ranks = rankdata(data_start, data_start + ind_size);

  // Now account for the zeros - we use averaging for breaking ties
  // the average of all N zeros is (N + 1) / 2
  std::size_t num_zeros = ndim - ind_size;

  // Negative value ranks can't be affected by the zero ranks, so we only
  // need to adjust the ranks of any positive values so they rank below the
  // zeros
  // also count the sum of the non-zero ranks while we are looping
  double nz_rank_sum = 0.0;
  for (size_t i = 0; i < ind_size; ++i) {
    if (*(data_start + i) > 0) {
      ranks[i] += num_zeros;
    }
    nz_rank_sum += ranks[i];
  }

  // as long as we always break ties with averaging, we know the total sum of
  // ranks no matter how many ties
  double total_rank_sum = ndim * (ndim + 1) / 2.0;

  // Zero rank average = sum of zero rank / total number of zero elements
  // use a negative value as a sentinel just in case we get a dense vector
  double zero_rank =
      ndim == ind_size ? -1.0 : (total_rank_sum - nz_rank_sum) / num_zeros;

  return {ranks, zero_rank};
}

template <typename Out, typename DataIt>
Out sparse_spearmanr(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t ndim) {

  // Calculate the mean of ranks
  double mean = (ndim + 1) / 2.0;

  auto [x_rank, x_rank0] =
      sparse_rankdata(ind1_start, ind1_size, data1_start, ndim);
  auto [y_rank, y_rank0] =
      sparse_rankdata(ind2_start, ind2_size, data2_start, ndim);

  const auto xc0 = x_rank0 < 0 ? 0.0 : x_rank0 - mean;
  const auto yc0 = y_rank0 < 0 ? 0.0 : y_rank0 - mean;
  const auto xc02 = xc0 * xc0;
  const auto yc02 = yc0 * yc0;

  double sum_xc2 = 0.0;
  double sum_yc2 = 0.0;
  double sum_xcyc = 0.0;

  std::size_t i1 = 0;
  std::size_t i2 = 0;
  // the size of the union of the indices
  std::size_t n = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      const auto xc = x_rank[i1] - mean;
      const auto yc = y_rank[i2] - mean;
      sum_xc2 += xc * xc;
      sum_yc2 += yc * yc;
      sum_xcyc += xc * yc;
      ++i1;
      ++i2;
      ++n;
    } else if (j1 < j2) {
      const auto xc = x_rank[i1] - mean;
      sum_xc2 += xc * xc;
      sum_yc2 += yc02;
      sum_xcyc += xc * yc0;
      ++i1;
      ++n;
    } else {
      const auto yc = y_rank[i2] - mean;
      sum_yc2 += yc * yc;
      sum_xc2 += xc02;
      sum_xcyc += xc0 * yc;
      ++i2;
      ++n;
    }
  }

  // Pass over the tails
  while (i1 < ind1_size) {
    const auto xc = x_rank[i1] - mean;
    sum_xc2 += xc * xc;
    sum_yc2 += yc02;
    sum_xcyc += xc * yc0;
    ++i1;
    ++n;
  }
  while (i2 < ind2_size) {
    const auto yc = y_rank[i2] - mean;
    sum_yc2 += yc * yc;
    sum_xc2 += xc02;
    sum_xcyc += xc0 * yc;
    ++i2;
    ++n;
  }

  std::size_t nzero_tail = ndim - n;
  sum_xc2 += xc02 * nzero_tail;
  sum_yc2 += yc02 * nzero_tail;
  sum_xcyc += xc0 * yc0 * nzero_tail;

  return 1.0 - (sum_xcyc / std::sqrt(sum_xc2 * sum_yc2));
}

template <typename Out, typename DataIt>
Out sparse_squared_euclidean(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {
  Out sum{0};

  std::size_t i1 = 0;
  std::size_t i2 = 0;

  while (i1 < ind1_size && i2 < ind2_size) {
    const auto j1 = *(ind1_start + i1);
    const auto j2 = *(ind2_start + i2);

    if (j1 == j2) {
      auto val = *(data1_start + i1) - *(data2_start + i2);
      sum += val * val;
      ++i1;
      ++i2;
    } else if (j1 < j2) {
      auto val = *(data1_start + i1);
      sum += val * val;
      ++i1;
    } else {
      auto val = *(data2_start + i2);
      sum += val * val;
      ++i2;
    }
  }

  // pass over the tails
  while (i1 < ind1_size) {
    auto val = *(data1_start + i1);
    sum += val * val;
    ++i1;
  }

  while (i2 < ind2_size) {
    auto val = *(data2_start + i2);
    sum += val * val;
    ++i2;
  }

  return static_cast<Out>(sum);
}

template <typename Out, typename DataIt>
Out sparse_manhattan(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {

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
Out sparse_matching(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt /* data1_start */,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt /* data2_start */, std::size_t ndim) {

  std::size_t num_true_true =
      fast_intersection_size(ind1_start, ind1_size, ind2_start, ind2_size);
  std::size_t num_non_zero = ind1_size + ind2_size - num_true_true;
  std::size_t num_not_equal = num_non_zero - num_true_true;

  return static_cast<Out>(static_cast<double>(num_not_equal) / ndim);
}

template <typename Out, typename DataIt>
Out sparse_rogers_tanimoto(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt /* data1_start */,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt /* data2_start */, std::size_t ndim) {

  std::size_t num_true_true =
      fast_intersection_size(ind1_start, ind1_size, ind2_start, ind2_size);
  std::size_t num_non_zero = ind1_size + ind2_size - num_true_true;
  std::size_t num_not_equal = num_non_zero - num_true_true;

  return static_cast<Out>((2.0 * num_not_equal) / (ndim + num_not_equal));
}

template <typename Out, typename DataIt>
Out sparse_russell_rao(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt /* data1_start */,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt /* data2_start */, std::size_t ndim) {

  std::size_t num_true_true = 0;
  auto i1 = ind1_start;
  auto i2 = ind2_start;

  while (i1 < ind1_start + ind1_size && i2 < ind2_start + ind2_size) {
    if (*i1 == *i2) {
      ++num_true_true;
      ++i1;
      ++i2;
    } else if (*i1 < *i2) {
      ++i1;
    } else {
      ++i2;
    }
  }

  if (num_true_true == ind1_size && num_true_true == ind2_size) {
    return Out{};
  } else {
    return static_cast<Out>(static_cast<double>(ndim - num_true_true) / ndim);
  }
}

template <typename Out, typename DataIt>
Out sparse_sokal_michener(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt /* data1_start */,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt /* data2_start */, std::size_t ndim) {

  std::size_t num_true_true =
      fast_intersection_size(ind1_start, ind1_size, ind2_start, ind2_size);
  std::size_t num_non_zero = ind1_size + ind2_size - num_true_true;
  std::size_t num_not_equal = num_non_zero - num_true_true;

  return static_cast<Out>(static_cast<double>(num_not_equal + num_not_equal) /
                          (ndim + num_not_equal));
}

template <typename Out, typename DataIt>
Out sparse_sokal_sneath(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt /* data1_start */,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt /* data2_start */, std::size_t /* ndim */) {

  std::size_t num_true_true =
      fast_intersection_size(ind1_start, ind1_size, ind2_start, ind2_size);
  std::size_t num_non_zero = ind1_size + ind2_size - num_true_true;
  std::size_t num_not_equal = num_non_zero - num_true_true;

  if (num_not_equal == 0) {
    return Out{};
  } else {
    return static_cast<Out>(num_not_equal /
                            (0.5 * num_true_true + num_not_equal));
  }
}

template <typename Out, typename DataIt>
Out sparse_true_angular(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {

  Out result = 0.0;
  Out norm_x = 0.0;
  Out norm_y = 0.0;

  auto i1 = ind1_start;
  auto d1 = data1_start;
  auto i2 = ind2_start;
  auto d2 = data2_start;

  while (i1 < ind1_start + ind1_size && i2 < ind2_start + ind2_size) {
    if (*i1 == *i2) {
      result += *d1 * *d2;
      norm_x += *d1 * *d1;
      norm_y += *d2 * *d2;
      ++i1;
      ++d1;
      ++i2;
      ++d2;
    } else {
      if (*i1 < *i2) {
        norm_x += *d1 * *d1;
        ++i1;
        ++d1;
      } else {
        norm_y += *d2 * *d2;
        ++i2;
        ++d2;
      }
    }
  }

  while (i1 < ind1_start + ind1_size) {
    norm_x += *d1 * *d1;
    ++i1;
    ++d1;
  }
  while (i2 < ind2_start + ind2_size) {
    norm_y += *d2 * *d2;
    ++i2;
    ++d2;
  }

  if (norm_x == 0.0 && norm_y == 0.0) {
    return 0.0;
  } else if (norm_x == 0.0 || norm_y == 0.0 || result <= 0.0) {
    return std::numeric_limits<Out>::max();
  } else {
    result /= std::sqrt(norm_x) * std::sqrt(norm_y);
    result = std::clamp(result, Out(-1), Out(1));
    result = std::acos(result) / M_PI;
    return 1.0 - result;
  }
}

template <typename Out, typename DataIt>
Out sparse_tsss(typename std::vector<std::size_t>::const_iterator ind1_start,
                std::size_t ind1_size, DataIt data1_start,
                typename std::vector<std::size_t>::const_iterator ind2_start,
                std::size_t ind2_size, DataIt data2_start, std::size_t ndim) {

  Out d_euc_squared = 0.0;
  Out d_cos = 0.0;
  Out norm_x = 0.0;
  Out norm_y = 0.0;

  auto i1 = ind1_start, i2 = ind2_start;
  auto d1 = data1_start, d2 = data2_start;

  while (i1 < ind1_start + ind1_size && i2 < ind2_start + ind2_size) {
    if (*i1 == *i2) {
      Out diff = *d1 - *d2;
      d_euc_squared += diff * diff;
      d_cos += *d1 * *d2;
      norm_x += *d1 * *d1;
      norm_y += *d2 * *d2;
      ++i1;
      ++d1;
      ++i2;
      ++d2;
    } else {
      if (*i1 < *i2) {
        norm_x += *d1 * *d1;
        d_euc_squared += *d1 * *d1;
        ++i1;
        ++d1;
      } else {
        norm_y += *d2 * *d2;
        d_euc_squared += *d2 * *d2;
        ++i2;
        ++d2;
      }
    }
  }

  // Remaining elements in vectors
  while (i1 < ind1_start + ind1_size) {
    norm_x += *d1 * *d1;
    d_euc_squared += *d1 * *d1;
    ++i1;
    ++d1;
  }
  while (i2 < ind2_start + ind2_size) {
    norm_y += *d2 * *d2;
    d_euc_squared += *d2 * *d2;
    ++i2;
    ++d2;
  }

  norm_x = std::sqrt(norm_x);
  norm_y = std::sqrt(norm_y);
  Out magnitude_difference = std::abs(norm_x - norm_y);
  d_cos /= norm_x * norm_y;
  d_cos = std::clamp(d_cos, Out(-1), Out(1));
  Out theta = std::acos(d_cos) + (M_PI / 18.0); // Add 10 degrees in radians

  Out sector =
      std::pow((std::sqrt(d_euc_squared) + magnitude_difference), 2) * theta;
  Out triangle = norm_x * norm_y * std::sin(theta) / 4.0;

  return triangle * sector;
}

template <typename Out, typename DataIt>
Out sparse_yule(typename std::vector<std::size_t>::const_iterator ind1_start,
                std::size_t ind1_size, DataIt /* data1_start */,
                typename std::vector<std::size_t>::const_iterator ind2_start,
                std::size_t ind2_size, DataIt /* data2_start */,
                std::size_t ndim) {

  std::size_t num_true_true = 0;
  std::size_t num_true_false = 0;
  std::size_t num_false_true = 0;

  auto i1 = ind1_start;
  auto i2 = ind2_start;

  while (i1 < ind1_start + ind1_size && i2 < ind2_start + ind2_size) {
    if (*i1 == *i2) {
      ++num_true_true;
      ++i1;
      ++i2;
    } else if (*i1 < *i2) {
      ++num_true_false;
      ++i1;
    } else {
      ++num_false_true;
      ++i2;
    }
  }

  num_true_false += (ind1_start + ind1_size) - i1;
  num_false_true += (ind2_start + ind2_size) - i2;

  std::size_t num_false_false =
      ndim - num_true_true - num_true_false - num_false_true;

  if (num_true_false == 0 || num_false_true == 0) {
    return Out{};
  } else {
    return static_cast<Out>(2.0 * num_true_false * num_false_true) /
           static_cast<Out>(num_true_true * num_false_false +
                            num_true_false * num_false_true);
  }
}

template <typename Out, typename DataIt>
Out sparse_symmetric_kl_divergence(
    typename std::vector<std::size_t>::const_iterator ind1_start,
    std::size_t ind1_size, DataIt data1_start,
    typename std::vector<std::size_t>::const_iterator ind2_start,
    std::size_t ind2_size, DataIt data2_start, std::size_t /* ndim */) {

  auto [dense_data1, dense_data2] = dense_union<Out, DataIt>(
      ind1_start, ind1_size, data1_start, ind2_start, ind2_size, data2_start);

  return symmetric_kl_divergence<Out>(dense_data1.begin(), dense_data1.end(),
                                      dense_data2.begin());
}

template <typename In>
void sparse_normalize(const std::vector<std::size_t> &ind,
                      const std::vector<std::size_t> &ptr,
                      std::vector<In> &data, std::size_t ndim) {

  constexpr double MIN_NORM = 1e-30;

  for (std::size_t i = 0; i < ptr.size() - 1; ++i) {
    auto data_start = data.begin() + ptr[i];
    auto data_end = data.begin() + ptr[i + 1];

    double norm =
        std::sqrt(std::inner_product(data_start, data_end, data_start, 0.0)) +
        MIN_NORM;

    std::transform(data_start, data_end, data_start,
                   [norm](In val) { return val / norm; });
  }
}

template <typename In, typename Out, typename Idx = uint32_t>
class SparseVectorDistance : public BaseDistance<Out, Idx> {
public:
  using SizeTypeIterator = typename std::vector<std::size_t>::const_iterator;
  using DataIterator = typename std::vector<In>::const_iterator;
  using SparseObs = std::tuple<SizeTypeIterator, std::size_t, DataIterator>;

  virtual ~SparseVectorDistance() = default;

  // return data pointing at the ith data point
  virtual std::tuple<SizeTypeIterator, std::size_t, DataIterator>
  get_x(Idx i) const = 0;
  virtual std::tuple<SizeTypeIterator, std::size_t, DataIterator>
  get_y(Idx i) const = 0;
};

template <typename In, typename Out, typename Idx>
struct DistanceTraits<std::unique_ptr<SparseVectorDistance<In, Out, Idx>>> {
  using Input = In;
  using Output = Out;
  using Index = Idx;
};

using SizeIt = typename std::vector<std::size_t>::const_iterator;
template <typename In, typename Out>
using SparseDistanceFunc = Out (*)(SizeIt, std::size_t, DataIt<In>, SizeIt,
                                   std::size_t, DataIt<In>, std::size_t);
template <typename In>
using SparsePreprocessFunc = void (*)(const std::vector<std::size_t> &,
                                      const std::vector<std::size_t> &,
                                      std::vector<In> &, std::size_t);

template <typename In, typename Out, typename Idx>
class SparseSelfDistanceCalculator : public SparseVectorDistance<In, Out, Idx> {
public:
  SparseSelfDistanceCalculator(
      std::vector<std::size_t> &&ind, std::vector<std::size_t> &&ptr,
      std::vector<In> &&data, std::size_t ndim,
      SparseDistanceFunc<In, Out> distance_func,
      SparsePreprocessFunc<In> preprocess_func = nullptr)
      : x_ind(std::move(ind)), x_ptr(std::move(ptr)), x_data(std::move(data)),
        nx(x_ptr.size() - 1), ndim(ndim), distance_func(distance_func) {
    if (preprocess_func) {
      preprocess_func(x_ind, x_ptr, x_data, ndim);
    }
  }

  virtual ~SparseSelfDistanceCalculator() = default;

  std::tuple<SizeIt, std::size_t, DataIt<In>> get_x(Idx i) const override {
    auto ind_start = x_ind.cbegin() + x_ptr[i];
    auto ind_end = x_ind.cbegin() + x_ptr[i + 1];
    auto data_start = x_data.cbegin() + x_ptr[i];
    auto ind_size = ind_end - ind_start;

    return std::make_tuple(ind_start, ind_size, data_start);
  }

  std::tuple<SizeIt, std::size_t, DataIt<In>> get_y(Idx i) const override {
    return get_x(i);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return nx; }

  Out calculate(const Idx &i, const Idx &j) const override {
    auto [ind1_start, ind1_size, data1_start] = this->get_x(i);
    auto [ind2_start, ind2_size, data2_start] = this->get_x(j);

    return distance_func(ind1_start, ind1_size, data1_start, ind2_start,
                         ind2_size, data2_start, this->ndim);
  }

  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;
  std::size_t ndim;
  SparseDistanceFunc<In, Out> distance_func;
};

template <typename In, typename Out, typename Idx>
class SparseQueryDistanceCalculator
    : public SparseVectorDistance<In, Out, Idx> {
public:
  SparseQueryDistanceCalculator(
      std::vector<std::size_t> &&x_ind, std::vector<std::size_t> &&x_ptr,
      std::vector<In> &&x_data, std::vector<std::size_t> &&y_ind,
      std::vector<std::size_t> &&y_ptr, std::vector<In> &&y_data,
      std::size_t ndim, SparseDistanceFunc<In, Out> distance_func,
      SparsePreprocessFunc<In> preprocess_func = nullptr)
      : x_ind(std::move(x_ind)), x_ptr(std::move(x_ptr)),
        x_data(std::move(x_data)), nx(this->x_ptr.size() - 1),
        y_ind(std::move(y_ind)), y_ptr(std::move(y_ptr)),
        y_data(std::move(y_data)), ny(this->y_ptr.size() - 1), ndim(ndim),
        distance_func(distance_func) {
    if (preprocess_func) {
      preprocess_func(this->x_ind, this->x_ptr, this->x_data, ndim);
      preprocess_func(this->y_ind, this->y_ptr, this->y_data, ndim);
    }
  }

  virtual ~SparseQueryDistanceCalculator() = default;

  std::tuple<SizeIt, std::size_t, DataIt<In>> get_x(Idx i) const override {
    auto ind_start = x_ind.cbegin() + x_ptr[i];
    auto ind_end = x_ind.cbegin() + x_ptr[i + 1];
    auto data_start = x_data.cbegin() + x_ptr[i];
    auto ind_size = ind_end - ind_start;

    return std::make_tuple(ind_start, ind_size, data_start);
  }

  std::tuple<SizeIt, std::size_t, DataIt<In>> get_y(Idx i) const override {
    auto ind_start = y_ind.cbegin() + y_ptr[i];
    auto ind_end = y_ind.cbegin() + y_ptr[i + 1];
    auto data_start = y_data.cbegin() + y_ptr[i];
    auto ind_size = ind_end - ind_start;
    return std::make_tuple(ind_start, ind_size, data_start);
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return ny; }

  Out calculate(const Idx &i, const Idx &j) const override {
    auto [ind1_start, ind1_size, data1_start] = this->get_x(i);
    auto [ind2_start, ind2_size, data2_start] = this->get_y(j);

    return distance_func(ind1_start, ind1_size, data1_start, ind2_start,
                         ind2_size, data2_start, this->ndim);
  }

  std::vector<std::size_t> x_ind;
  std::vector<std::size_t> x_ptr;
  std::vector<In> x_data;
  std::size_t nx;

  std::vector<std::size_t> y_ind;
  std::vector<std::size_t> y_ptr;
  std::vector<In> y_data;
  std::size_t ny;

  std::size_t ndim;

  SparseDistanceFunc<In, Out> distance_func;
};

} // namespace tdoann

#endif // TDOANN_SPARSE_H

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

#ifndef TDOANN_DISTANCE_H
#define TDOANN_DISTANCE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

// NOLINTBEGIN(readability-identifier-length)

namespace tdoann {

// distance functions

// functions are organized alphabetically, requiring forward declaration

template <typename Out, typename It>
Out squared_euclidean(const It xbegin, const It xend, const It ybegin);
template <typename It> std::vector<double> rankdata(It begin, It end);

template <typename Out, typename It>
Out bray_curtis(const It xbegin, const It xend, const It ybegin) {
  Out numerator = 0.0;
  Out denominator = 0.0;

  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    numerator += std::abs(*xit - *yit);
    denominator += std::abs(*xit + *yit);
  }

  if (denominator > Out{0}) {
    return numerator / denominator;
  } else {
    return Out{0};
  }
}

template <typename Out, typename It>
Out canberra(const It xbegin, const It xend, const It ybegin) {
  Out result = 0.0;
  for (auto xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    const auto xi = *xit;
    const auto yi = *yit;
    const auto denominator = std::abs(xi) + std::abs(yi);

    if (denominator > Out{0}) {
      result += std::abs(xi - yi) / denominator;
    }
  }
  return result;
}

template <typename Out, typename It>
Out chebyshev(const It xbegin, const It xend, const It ybegin) {
  Out result = 0;
  for (auto xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    result = std::max(result, std::abs(*xit - *yit));
  }
  return result;
}

template <typename Out, typename It>
inline Out correlation(const It xbegin, const It xend, const It ybegin) {
  // calculate mean
  Out xmu = 0.0;
  Out ymu = 0.0;
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    xmu += *xit;
    ymu += *yit;
  }
  const auto n = std::distance(xbegin, xend);
  xmu /= n;
  ymu /= n;

  // cosine on mean centered data
  Out res = 0.0;
  Out normx = 0.0;
  Out normy = 0.0;
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    Out x = *xit - xmu;
    Out y = *yit - ymu;
    res += x * y;
    normx += x * x;
    normy += y * y;
  }

  constexpr Out zero = 0.0;
  if (normx == zero && normy == zero) {
    return zero;
  }
  constexpr Out one = 1.0;
  if (normx == zero || normy == zero) {
    return one;
  }
  return one - (res / std::sqrt(normx * normy));
}

template <typename Out, typename It>
Out cosine(const It xbegin, const It xend, const It ybegin) {
  Out result = 0.0;
  Out norm_x = 0.0;
  Out norm_y = 0.0;

  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    Out x = *xit;
    Out y = *yit;
    result += x * y;
    norm_x += x * x;
    norm_y += y * y;
  }

  if (norm_x == 0.0 && norm_y == 0.0) {
    return 0.0;
  }
  if (norm_x == 0.0 || norm_y == 0.0) {
    return 1.0;
  }
  return 1.0 - (result / std::sqrt(norm_x * norm_y));
}

template <typename Out, typename It>
Out alternative_cosine(const It xbegin, const It xend, const It ybegin) {
  Out result = 0.0;
  Out norm_x = 0.0;
  Out norm_y = 0.0;

  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    Out x = *xit;
    Out y = *yit;
    result += x * y;
    norm_x += x * x;
    norm_y += y * y;
  }

  if (norm_x == 0.0 && norm_y == 0.0) {
    return 0.0;
  }

  Out max_value = std::numeric_limits<Out>::max();
  if (norm_x == 0.0 || norm_y == 0.0 || result <= 0.0) {
    return max_value;
  }

  result = std::sqrt(norm_x * norm_y) / result;
  return std::log2(result);
}

template <typename Out, typename It> auto dice(It xbegin, It xend, It ybegin) {
  std::size_t num_true_true = 0;
  std::size_t num_not_equal = 0;
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    bool x_true = *xit != 0;
    bool y_true = *yit != 0;
    num_true_true += x_true && y_true;
    num_not_equal += x_true != y_true;
  }

  if (num_not_equal == 0) {
    return Out{0};
  } else {
    return static_cast<Out>(static_cast<double>(num_not_equal) /
                            (2 * num_true_true + num_not_equal));
  }
}

template <typename Out, typename It>
auto dot(const It xbegin, const It xend, const It ybegin) {
  Out result = 0;
  for (auto xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    result += (*xit) * (*yit);
  }

  if (result <= 0.0) {
    return static_cast<Out>(1.0);
  } else {
    return static_cast<Out>(1.0) - result;
  }
}

template <typename Out, typename It>
auto alternative_dot(const It xbegin, const It xend, const It ybegin) {
  Out result = 0;
  for (auto xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    result += (*xit) * (*yit);
  }

  if (result <= 0.0) {
    return std::numeric_limits<Out>::max();
  }

  return -std::log2(result);
}

template <typename Out, typename It>
Out euclidean(const It xbegin, const It xend, const It ybegin) {
  return std::sqrt(squared_euclidean<Out>(xbegin, xend, ybegin));
}

template <typename Out, typename It>
Out hamming(const It xbegin, const It xend, const It ybegin) {
  Out sum{0};
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    sum += *xit != *yit;
  }
  return static_cast<Out>(static_cast<double>(sum) /
                          std::distance(xbegin, xend));
}

template <typename Out, typename It>
auto hellinger(const It xbegin, const It xend, const It ybegin) {
  Out result = 0.0;
  Out l1_norm_x = 0.0;
  Out l1_norm_y = 0.0;

  for (auto xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    result += std::sqrt((*xit) * (*yit));
    l1_norm_x += *xit;
    l1_norm_y += *yit;
  }

  if (l1_norm_x == 0 && l1_norm_y == 0) {
    return Out{0};
  } else if (l1_norm_x == 0 || l1_norm_y == 0) {
    return static_cast<Out>(1.0);
  } else {
    return std::sqrt(static_cast<Out>(1.0) -
                     result / std::sqrt(l1_norm_x * l1_norm_y));
  }
}

template <typename Out, typename It>
auto alternative_hellinger(const It xbegin, const It xend, const It ybegin) {
  Out result = 0.0;
  Out l1_norm_x = 0.0;
  Out l1_norm_y = 0.0;

  for (auto xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    result += std::sqrt((*xit) * (*yit));
    l1_norm_x += *xit;
    l1_norm_y += *yit;
  }

  constexpr Out FLOAT32_MAX = std::numeric_limits<Out>::max();

  if (l1_norm_x == 0 && l1_norm_y == 0) {
    return Out{0};
  } else if (l1_norm_x == 0 || l1_norm_y == 0 || result <= 0) {
    return FLOAT32_MAX;
  } else {
    result = std::sqrt(l1_norm_x * l1_norm_y) / result;
    return std::log2(result);
  }
}

template <typename Out, typename It>
auto inner_product(const It xbegin, const It xend, const It ybegin) {
  Out sum{0};
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    sum += *xit * *yit;
  }
  return std::max(1 - sum, Out{0});
}

template <typename Out, typename It>
auto jaccard(It xbegin, It xend, It ybegin) {
  std::size_t num_non_zero = 0;
  std::size_t num_equal = 0;
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    bool x_true = *xit != 0;
    bool y_true = *yit != 0;
    num_non_zero += x_true || y_true;
    num_equal += x_true && y_true;
  }

  if (num_non_zero == 0) {
    return Out{0};
  } else {
    return static_cast<Out>(static_cast<double>(num_non_zero - num_equal) /
                            num_non_zero);
  }
}

template <typename Out, typename It>
Out alternative_jaccard(It xbegin, It xend, It ybegin) {
  std::size_t num_non_zero = 0;
  std::size_t num_equal = 0;
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    bool x_true = *xit != 0;
    bool y_true = *yit != 0;
    num_non_zero += x_true || y_true;
    num_equal += x_true && y_true;
  }

  if (num_non_zero == 0) {
    return Out{0};
  }
  if (num_equal == 0) {
    return std::numeric_limits<Out>::max();
  }
  return -std::log2(static_cast<double>(num_equal) / num_non_zero);
}

template <typename Out, typename It>
Out jensen_shannon_divergence(It xbegin, It xend, It ybegin) {
  const std::size_t ndim = std::distance(xbegin, xend);
  Out l1_norm_x = 0;
  Out l1_norm_y = 0;

  for (std::size_t i = 0; i < ndim; ++i) {
    l1_norm_x += std::abs(*(xbegin + i));
    l1_norm_y += std::abs(*(ybegin + i));
  }

  constexpr Out FLOAT32_EPS = std::numeric_limits<float>::epsilon();
  l1_norm_x += FLOAT32_EPS * ndim;
  l1_norm_y += FLOAT32_EPS * ndim;

  Out result = 0.0;
  for (std::size_t i = 0; i < ndim; ++i) {
    const Out xi = *(xbegin + i) + FLOAT32_EPS;
    const Out yi = *(ybegin + i) + FLOAT32_EPS;
    const Out m = 0.5 * (xi / l1_norm_x + yi / l1_norm_y);

    if (xi > FLOAT32_EPS) {
      result += 0.5 * (xi / l1_norm_x) * std::log((xi / l1_norm_x) / m);
    }
    if (yi > FLOAT32_EPS) {
      result += 0.5 * (yi / l1_norm_y) * std::log((yi / l1_norm_y) / m);
    }
  }

  return result;
}

template <typename Out, typename It>
auto kulsinski(It xbegin, It xend, It ybegin) {
  std::size_t num_true_true = 0;
  std::size_t num_not_equal = 0;
  std::size_t length = std::distance(xbegin, xend);
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    bool x_true = *xit != 0;
    bool y_true = *yit != 0;
    num_true_true += (x_true && y_true);
    num_not_equal += (x_true != y_true);
  }

  if (num_not_equal == 0) {
    return Out{0};
  } else {
    return static_cast<Out>(
        static_cast<double>(num_not_equal - num_true_true + length) /
        (num_not_equal + length));
  }
}

template <typename Out, typename It>
Out manhattan(const It xbegin, const It xend, const It ybegin) {
  Out sum{0};
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    sum += std::abs(*xit - *yit);
  }
  return sum;
}

template <typename Out, typename It>
auto matching(It xbegin, It xend, It ybegin) {
  std::size_t num_not_equal = 0;
  std::size_t length = std::distance(xbegin, xend);
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    bool x_true = *xit != 0;
    bool y_true = *yit != 0;
    num_not_equal += x_true != y_true;
  }

  return static_cast<Out>(static_cast<double>(num_not_equal) / length);
}

template <typename Out, typename It>
auto rogers_tanimoto(It xbegin, It xend, It ybegin) {
  std::size_t num_not_equal = 0;
  std::size_t length = std::distance(xbegin, xend);
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    num_not_equal += (*xit != 0) != (*yit != 0);
  }

  return static_cast<Out>((2.0 * num_not_equal) / (length + num_not_equal));
}

template <typename Out, typename It>
auto russell_rao(It xbegin, It xend, It ybegin) {
  std::size_t num_true_true = 0;
  std::size_t num_x_nonzero = 0;
  std::size_t num_y_nonzero = 0;
  std::size_t length = std::distance(xbegin, xend);

  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    bool x_nonzero = *xit != 0;
    bool y_nonzero = *yit != 0;
    num_x_nonzero += x_nonzero;
    num_y_nonzero += y_nonzero;
    num_true_true += x_nonzero && y_nonzero;
  }

  if (num_true_true == num_x_nonzero && num_true_true == num_y_nonzero) {
    return Out{0};
  } else {
    return static_cast<Out>(length - num_true_true) / static_cast<Out>(length);
  }
}

template <typename Out, typename It>
Out sokal_michener(It xbegin, It xend, It ybegin) {
  std::size_t num_not_equal = 0;
  std::size_t num_equal = 0;

  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    bool x_true = *xit != 0;
    bool y_true = *yit != 0;
    num_not_equal += (x_true != y_true);
    num_equal += (x_true == y_true);
  }

  // the same as (2.0 * num_not_equal) / (ndim + num_not_equal)
  // but avoid having to calculate ndim as std::difference(xbegin, xend);
  const double nne2 = num_not_equal + num_not_equal;
  return static_cast<Out>(nne2 / (num_equal + nne2));
}

template <typename Out, typename It>
auto sokal_sneath(It xbegin, It xend, It ybegin) {
  std::size_t num_true_true = 0;
  std::size_t num_not_equal = 0;

  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    bool x_true = *xit != 0;
    bool y_true = *yit != 0;
    num_true_true += x_true && y_true;
    num_not_equal += x_true != y_true;
  }

  if (num_not_equal == 0) {
    return Out{0};
  } else {
    return static_cast<Out>(num_not_equal) /
           static_cast<Out>(0.5 * num_true_true + num_not_equal);
  }
}

template <typename Out, typename It>
Out spearmanr(It xbegin, It xend, It ybegin) {
  auto x_rank = rankdata(xbegin, xend);
  auto y_rank = rankdata(ybegin, ybegin + std::distance(xbegin, xend));

  return correlation<Out>(x_rank.begin(), x_rank.end(), y_rank.begin());
}

template <typename Out, typename It>
Out squared_euclidean(const It xbegin, const It xend, const It ybegin) {
  Out sum{0};
  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    const Out diff = *xit - *yit;
    sum += diff * diff;
  }
  return sum;
}

template <typename Out, typename It>
Out symmetric_kl_divergence(It xbegin, It xend, It ybegin) {
  const std::size_t ndim = std::distance(xbegin, xend);
  Out l1_norm_x = 0;
  Out l1_norm_y = 0;

  // First pass to calculate L1 norms
  for (std::size_t i = 0; i < ndim; ++i) {
    l1_norm_x += std::abs(*(xbegin + i));
    l1_norm_y += std::abs(*(ybegin + i));
  }

  constexpr Out FLOAT32_EPS = std::numeric_limits<float>::epsilon();
  l1_norm_x += FLOAT32_EPS * ndim;
  l1_norm_y += FLOAT32_EPS * ndim;

  Out result = 0.0;
  for (std::size_t i = 0; i < ndim; ++i) {
    const Out xi = *(xbegin + i) + FLOAT32_EPS;
    const Out yi = *(ybegin + i) + FLOAT32_EPS;
    const Out pdf_xi = xi / l1_norm_x;
    const Out pdf_yi = yi / l1_norm_y;

    if (pdf_xi > FLOAT32_EPS) {
      result += pdf_xi * std::log(pdf_xi / pdf_yi);
    }
    if (pdf_yi > FLOAT32_EPS) {
      result += pdf_yi * std::log(pdf_yi / pdf_xi);
    }
  }

  return result;
}

template <typename Out, typename It>
Out true_angular(It xbegin, It xend, It ybegin) {
  Out result = 0.0;
  Out norm_x = 0.0;
  Out norm_y = 0.0;

  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    result += *xit * *yit;
    norm_x += *xit * *xit;
    norm_y += *yit * *yit;
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

template <typename Out, typename It> Out tsss(It xbegin, It xend, It ybegin) {
  Out d_euc_squared = 0.0;
  Out d_cos = 0.0;
  Out norm_x = 0.0;
  Out norm_y = 0.0;

  for (auto xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    Out diff = *xit - *yit;
    d_euc_squared += diff * diff;
    d_cos += *xit * *yit;
    norm_x += *xit * *xit;
    norm_y += *yit * *yit;
  }

  norm_x = std::sqrt(norm_x);
  norm_y = std::sqrt(norm_y);
  Out magnitude_difference = std::abs(norm_x - norm_y);
  d_cos /= norm_x * norm_y;
  // very real chance of d_cos being outside [-1, 1] range and causing acos
  // to give a NaN when comparing a point with itself
  d_cos = std::clamp(d_cos, Out(-1), Out(1));
  Out theta = std::acos(d_cos) + (M_PI / 18.0); // Add 10 degrees in radians

  Out sector =
      std::pow((std::sqrt(d_euc_squared) + magnitude_difference), 2) * theta;
  Out triangle = norm_x * norm_y * std::sin(theta) / 4.0;

  return triangle * sector;
}

template <typename Out, typename It> auto yule(It xbegin, It xend, It ybegin) {
  std::size_t num_true_true = 0;
  std::size_t num_true_false = 0;
  std::size_t num_false_true = 0;
  std::size_t num_false_false = 0;

  for (It xit = xbegin, yit = ybegin; xit != xend; ++xit, ++yit) {
    bool x_true = *xit != 0;
    bool y_true = *yit != 0;
    num_true_true += x_true && y_true;
    num_true_false += x_true && !y_true;
    num_false_true += !x_true && y_true;
  }

  num_false_false = std::distance(xbegin, xend) - num_true_true -
                    num_true_false - num_false_true;

  if (num_true_false == 0 || num_false_true == 0) {
    return Out{0};
  } else {
    return static_cast<Out>(2.0 * num_true_false * num_false_true) /
           static_cast<Out>(num_true_true * num_false_false +
                            num_true_false * num_false_true);
  }
}

template <typename It> std::vector<double> rankdata(It begin, It end) {
  std::vector<double> ranks(std::distance(begin, end));
  std::vector<size_t> indices(ranks.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(),
            [&](size_t a, size_t b) { return *(begin + a) < *(begin + b); });

  // Calculate dense ranks
  for (size_t i = 0; i < indices.size(); ++i) {
    ranks[indices[i]] = i + 1;
  }

  // Handle ties by averaging ranks
  for (size_t i = 0; i < ranks.size();) {
    size_t j = i;
    double sum_ranks = 0.0;
    while (j < ranks.size() && *(begin + indices[i]) == *(begin + indices[j])) {
      sum_ranks += ranks[indices[j]];
      ++j;
    }

    double average_rank = sum_ranks / (j - i);
    for (size_t k = i; k < j; ++k) {
      ranks[indices[k]] = average_rank;
    }

    i = j;
  }

  return ranks;
}

// Note that this is done *in-place* to avoid unnecessary copying
template <typename T> void normalize(std::vector<T> &vec, std::size_t ndim) {
  constexpr T MIN_NORM = 1e-30;
  for (auto start_it = vec.begin(); start_it != vec.end(); start_it += ndim) {
    T norm = std::sqrt(std::inner_product(start_it, start_it + ndim, start_it,
                                          T{0})) +
             MIN_NORM;
    std::transform(start_it, start_it + ndim, start_it,
                   [norm](T val) { return val / norm; });
  }
}

// Note that this is done *in-place* to avoid unnecessary copying
template <typename T> void mean_center(std::vector<T> &vec, std::size_t ndim) {
  for (auto start_it = vec.begin(); start_it != vec.end(); start_it += ndim) {
    T mu = std::accumulate(start_it, start_it + ndim, T{0}) / ndim;
    std::transform(start_it, start_it + ndim, start_it,
                   [mu](T val) { return val - mu; });
  }
}

template <typename T>
void mean_center_and_normalize(std::vector<T> &vec, std::size_t ndim) {
  mean_center(vec, ndim);
  normalize(vec, ndim);
}

} // namespace tdoann
#endif // TDOANN_DISTANCE_H
// NOLINTEND(readability-identifier-length)

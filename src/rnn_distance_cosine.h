//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#ifndef RNN_DISTANCE_COSINE_H
#define RNN_DISTANCE_COSINE_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "tdoann/distancebase.h"

#include "rnn_distance_batch.h"
#include "rnn_util.h"

namespace rnndescent {

enum class CosineBuildStage { leaf, refinement };

template <CosineBuildStage Stage>
class CosineSelfDistance final
    : public tdoann::SelfDistanceCalculator<RNN_DEFAULT_IN, RNN_DEFAULT_DIST,
                                            RNN_DEFAULT_IDX> {
  using Base = tdoann::SelfDistanceCalculator<RNN_DEFAULT_IN, RNN_DEFAULT_DIST,
                                              RNN_DEFAULT_IDX>;
  using BatchDistance = DenseDistanceFour<RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>;
  std::vector<float> norms;

public:
  CosineSelfDistance(std::vector<float> &&data, std::size_t dimension)
      : Base(std::move(data), dimension, tdoann::cosine<float, Iterator>),
        norms(cosine_norms(x.data(), nx, ndim)) {}

  auto batch_kernel() const -> BatchDistance {
    return {[this](RNN_DEFAULT_IDX p, const BatchDistance::Targets &targets) {
      auto scores = BatchDistance::run<DenseBatchMetric::raw_dot>(
          x.data(), ndim, p, targets);
      for (std::size_t j = 0; j < targets.size(); ++j) {
        const auto q = targets[j];
        scores[j] = norms[p] < 0 || norms[q] < 0
                        ? calculate(p, q)
                        : from_dot(scores[j], norms[p], norms[q]);
      }
      return scores;
    }};
  }

  float calculate(const RNN_DEFAULT_IDX &i,
                  const RNN_DEFAULT_IDX &j) const override {
    const float norm_i = norms[i];
    const float norm_j = norms[j];
    auto first = get_x(i);
    const auto end = first + ndim;
    auto second = get_x(j);
    if (norm_i < 0 || norm_j < 0) {
      return distance_func(first, end, second);
    }
    float dot = 0;
    if constexpr (Stage == CosineBuildStage::refinement) {
      dot = tdoann::sum_feature_products(first, end, second, 0.0f,
                                         std::multiplies<>{});
    } else {
      for (; first != end; ++first, ++second) {
        const float a = *first;
        const float b = *second;
        dot += a * b;
      }
    }
    return from_dot(dot, norm_i, norm_j);
  }

private:
  static auto cosine_norms(const float *data, std::size_t count,
                           std::size_t dimension) -> std::vector<float> {
    std::vector<float> norms(count);
    for (std::size_t row = 0; row < count; ++row) {
      float squared = 0;
      const auto *values = data + row * dimension;
      // Keep the indexed scan and feature order for float rounding parity.
      for (std::size_t j = 0; j < dimension; ++j) {
        squared += values[j] * values[j];
      }
      // Keep the original kernel for zero and near the limits of float
      // arithmetic, where rounding can change underflow or overflow.
      constexpr float lower = 4 * std::numeric_limits<float>::min();
      constexpr float upper = std::numeric_limits<float>::max() / 4;
      norms[row] =
          squared >= lower && squared <= upper ? std::sqrt(squared) : -1.0f;
    }
    return norms;
  }

  static float from_dot(float dot, float norm_i, float norm_j) {
    return std::max(1.0f - dot / (norm_i * norm_j), 0.0f);
  }
};

} // namespace rnndescent

#endif // RNN_DISTANCE_COSINE_H

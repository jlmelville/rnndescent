//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#ifndef RNN_DISTANCE_BATCH_H
#define RNN_DISTANCE_BATCH_H

#include <array>
#include <cmath>
#include <cstddef>

#include "tdoann/distance.h"

namespace rnndescent {

enum class DenseBatchMetric {
  squared_euclidean,
  euclidean,
  inner_product,
  dot,
  alternative_dot,
  manhattan
};

template <typename Out, typename Idx> struct DenseDistanceFour {
  using Targets = std::array<Idx, 4>;
  using Distances = std::array<Out, 4>;
  using Evaluate = Distances (*)(const float *, std::size_t, Idx,
                                 const Targets &);

  const float *data{nullptr};
  std::size_t dimension{0};
  Evaluate evaluate{nullptr};

  auto calculate(Idx p, const Targets &q) const -> Distances {
    return evaluate(data, dimension, p, q);
  }

  template <DenseBatchMetric Metric>
  static auto run(const float *data, std::size_t dimension, Idx p,
                  const Targets &q) -> Distances {
    return run_from<Metric>(data + p * dimension, data, dimension, q);
  }

  template <DenseBatchMetric Metric>
  static auto run_from(const float *x, const float *data, std::size_t dimension,
                       const Targets &q) -> Distances {
    const auto *y0 = data + q[0] * dimension;
    const auto *y1 = data + q[1] * dimension;
    const auto *y2 = data + q[2] * dimension;
    const auto *y3 = data + q[3] * dimension;
    Out sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    // Share the anchor load and keep each candidate's feature order.
    for (std::size_t j = 0; j < dimension; ++j) {
      const auto anchor = x[j];
      if constexpr (Metric == DenseBatchMetric::squared_euclidean ||
                    Metric == DenseBatchMetric::euclidean) {
        const auto d0 = anchor - y0[j];
        const auto d1 = anchor - y1[j];
        const auto d2 = anchor - y2[j];
        const auto d3 = anchor - y3[j];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
      } else if constexpr (Metric == DenseBatchMetric::manhattan) {
        sum0 += std::abs(anchor - y0[j]);
        sum1 += std::abs(anchor - y1[j]);
        sum2 += std::abs(anchor - y2[j]);
        sum3 += std::abs(anchor - y3[j]);
      } else {
        sum0 += anchor * y0[j];
        sum1 += anchor * y1[j];
        sum2 += anchor * y2[j];
        sum3 += anchor * y3[j];
      }
    }
    return {finish<Metric>(sum0), finish<Metric>(sum1), finish<Metric>(sum2),
            finish<Metric>(sum3)};
  }

private:
  template <DenseBatchMetric Metric> static auto finish(Out sum) -> Out {
    if constexpr (Metric == DenseBatchMetric::euclidean) {
      return std::sqrt(sum);
    } else if constexpr (Metric == DenseBatchMetric::inner_product) {
      return tdoann::finish_inner_product(sum);
    } else if constexpr (Metric == DenseBatchMetric::dot) {
      return tdoann::finish_dot(sum);
    } else if constexpr (Metric == DenseBatchMetric::alternative_dot) {
      return tdoann::finish_alternative_dot(sum);
    } else {
      return sum;
    }
  }
};

} // namespace rnndescent

#endif // RNN_DISTANCE_BATCH_H

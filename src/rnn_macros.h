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

#ifndef RNN_MACROS_H
#define RNN_MACROS_H

#include <Rcpp.h>

#include "tdoann/distance.h"

#define DISPATCH_ON_DISTANCES(NEXT_MACRO)                                      \
  if (metric == "euclidean") {                                                 \
    using Distance = tdoann::EuclideanSelf<float, float>;                      \
    NEXT_MACRO()                                                               \
  } else if (metric == "l2sqr") {                                              \
    using Distance = tdoann::L2SqrSelf<float, float>;                          \
    NEXT_MACRO()                                                               \
  } else if (metric == "cosine") {                                             \
    using Distance = tdoann::CosineSelf<float, float>;                         \
    NEXT_MACRO()                                                               \
  } else if (metric == "manhattan") {                                          \
    using Distance = tdoann::ManhattanSelf<float, float>;                      \
    NEXT_MACRO()                                                               \
  } else if (metric == "bhamming") {                                           \
    using Distance = tdoann::BHammingSelf<uint8_t, std::size_t>;               \
    NEXT_MACRO()                                                               \
  } else if (metric == "hamming") {                                            \
    using Distance = tdoann::HammingSelf<float, std::size_t>;                  \
    NEXT_MACRO()                                                               \
  } else if (metric == "correlation") {                                        \
    using Distance = tdoann::CorrelationSelf<float, float>;                    \
    NEXT_MACRO()                                                               \
  } else {                                                                     \
    Rcpp::stop("Bad metric");                                                  \
  }

#define DISPATCH_ON_QUERY_DISTANCES(NEXT_MACRO)                                \
  if (metric == "euclidean") {                                                 \
    using Distance = tdoann::EuclideanQuery<float, float>;                     \
    NEXT_MACRO()                                                               \
  } else if (metric == "l2sqr") {                                              \
    using Distance = tdoann::L2SqrQuery<float, float>;                         \
    NEXT_MACRO()                                                               \
  } else if (metric == "cosine") {                                             \
    using Distance = tdoann::CosineQuery<float, float>;                        \
    NEXT_MACRO()                                                               \
  } else if (metric == "manhattan") {                                          \
    using Distance = tdoann::ManhattanQuery<float, float>;                     \
    NEXT_MACRO()                                                               \
  } else if (metric == "bhamming") {                                           \
    using Distance = tdoann::BHammingQuery<uint8_t, std::size_t>;              \
    NEXT_MACRO()                                                               \
  } else if (metric == "hamming") {                                            \
    using Distance = tdoann::HammingQuery<float, std::size_t>;                 \
    NEXT_MACRO()                                                               \
  } else if (metric == "correlation") {                                        \
    using Distance = tdoann::CorrelationQuery<float, float>;                   \
    NEXT_MACRO()                                                               \
  } else {                                                                     \
    Rcpp::stop("Bad metric");                                                  \
  }

#endif // #RNN_MACROS_H

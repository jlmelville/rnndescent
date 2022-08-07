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
  using In = float;                                                            \
  using Out = float;                                                           \
  using It = std::vector<In>::const_iterator;                                  \
  if (metric == "euclidean") {                                                 \
    using Distance = tdoann::SelfDistance<In, Out, It, tdoann::euclidean>;     \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "l2sqr") {                                                     \
    using Distance = tdoann::SelfDistance<In, Out, It, tdoann::l2sqr>;         \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "cosine") {                                                    \
    using Distance = tdoann::SelfDistance<In, Out, It, tdoann::inner_product,  \
                                          tdoann::normalize>;                  \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "manhattan") {                                                 \
    using Distance = tdoann::SelfDistance<In, Out, It, tdoann::manhattan>;     \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "bhamming") {                                                  \
    using Distance = tdoann::BHammingSelf<uint8_t, std::size_t>;               \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "hamming") {                                                   \
    using Distance = tdoann::SelfDistance<In, Out, It, tdoann::hamming>;       \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "correlation") {                                               \
    using Distance = tdoann::SelfDistance<In, Out, It, tdoann::inner_product,  \
                                          tdoann::normalize_center>;           \
    NEXT_MACRO()                                                               \
  }                                                                            \
  Rcpp::stop("Bad metric");

#define DISPATCH_ON_QUERY_DISTANCES(NEXT_MACRO)                                \
  using In = float;                                                            \
  using Out = float;                                                           \
  using It = std::vector<In>::const_iterator;                                  \
  if (metric == "euclidean") {                                                 \
    using Distance = tdoann::QueryDistance<In, Out, It, tdoann::euclidean>;    \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "l2sqr") {                                                     \
    using Distance = tdoann::QueryDistance<In, Out, It, tdoann::l2sqr>;        \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "cosine") {                                                    \
    using Distance = tdoann::QueryDistance<In, Out, It, tdoann::inner_product, \
                                           tdoann::normalize>;                 \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "manhattan") {                                                 \
    using Distance = tdoann::QueryDistance<In, Out, It, tdoann::manhattan>;    \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "bhamming") {                                                  \
    using Distance = tdoann::BHammingQuery<uint8_t, std::size_t>;              \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "hamming") {                                                   \
    using Distance = tdoann::QueryDistance<In, Out, It, tdoann::hamming>;      \
    NEXT_MACRO()                                                               \
  }                                                                            \
  if (metric == "correlation") {                                               \
    using Distance = tdoann::QueryDistance<In, Out, It, tdoann::inner_product, \
                                           tdoann::normalize_center>;          \
    NEXT_MACRO()                                                               \
  }                                                                            \
  Rcpp::stop("Bad metric");

#endif // #RNN_MACROS_H

//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2021 James Melville
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

#include <Rcpp.h>

#include "rnn_util.h"

using namespace Rcpp;

// [[Rcpp::export]]
IntegerVector reverse_nbr_size_impl(IntegerMatrix nn_idx, std::size_t k,
                                    std::size_t len,
                                    bool include_self = false) {
  const std::size_t nr = nn_idx.nrow();

  auto data = as<std::vector<std::size_t>>(nn_idx);

  const constexpr std::size_t missing = static_cast<std::size_t>(-1);

  std::vector<std::size_t> n_reverse(len);

  for (std::size_t i = 0; i < nr; i++) {
    for (std::size_t j = 0; j < k; j++) {
      std::size_t inbr = data[nr * j + i];
      if (inbr == missing) {
        continue;
      }
      // zero index
      --inbr;
      if (inbr == i && !include_self) {
        continue;
      }
      ++n_reverse[inbr];
    }
  }
  return IntegerVector(n_reverse.begin(), n_reverse.end());
}

// [[Rcpp::export]]
IntegerVector shared_nbr_size_impl(IntegerMatrix nn_idx,
                                   std::size_t n_neighbors) {
  const std::size_t n_items = nn_idx.nrow();
  const constexpr std::size_t missing = static_cast<std::size_t>(-1);

  const std::size_t n_dim = nn_idx.ncol();
  auto data = r_to_idxt<std::size_t>(nn_idx);

  std::vector<std::size_t> n_shared(n_items);

  for (std::size_t i = 0; i < n_items; i++) {
    const auto ind = n_dim * i;
    for (std::size_t j = 0; j < n_neighbors; j++) {
      const auto inbr = data[ind + j];
      // skip missing and self neighbor
      if (inbr == missing || inbr == i) {
        continue;
      }
      const auto ij = n_dim * inbr;
      for (std::size_t k = 0; k < n_neighbors; k++) {
        if (data[ij + k] == i) {
          ++n_shared[i];
        }
      }
    }
  }
  return IntegerVector(n_shared.begin(), n_shared.end());
}

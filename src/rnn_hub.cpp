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

// NOLINTBEGIN(modernize-use-trailing-return-type)

#include <Rcpp.h>

#include "rnn_util.h"

using Rcpp::as;
using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;

// [[Rcpp::export]]
IntegerVector reverse_nbr_size_impl(const IntegerMatrix &nn_idx,
                                    std::size_t nnbrs, std::size_t len,
                                    bool include_self = false) {
  const std::size_t nobs = nn_idx.nrow();
  auto data = as<std::vector<std::size_t>>(nn_idx);
  constexpr auto missing = static_cast<std::size_t>(-1);
  std::vector<std::size_t> n_reverse(len);

  for (std::size_t i = 0; i < nnbrs; i++) {
    const auto inobs = nobs * i;
    for (std::size_t j = 0; j < nobs; j++) {
      auto jnbr = data[inobs + j];
      if (jnbr == missing) {
        continue;
      }
      // zero index
      --jnbr;
      if (jnbr == j && !include_self) {
        continue;
      }
      ++n_reverse[jnbr];
    }
  }

  return {n_reverse.begin(), n_reverse.end()};
}

// NOLINTEND(modernize-use-trailing-return-type)

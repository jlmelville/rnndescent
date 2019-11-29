//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2019 James Melville
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

// This code is a minor modification of that in dqrng, which is AGPL licensed.
// Anything in this file is AGPL:
//
// Copyright 2019 James Melville
//
// rnn_sample.h is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// rnn_sample.h is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

#ifndef RNN_SAMPLE_H
#define RNN_SAMPLE_H

#include <Rcpp.h>

// [[Rcpp::depends(dqrng, BH)]]
#include "dqrng_generator.h"
#include "minimal_int_set.h"

template <typename INT>
std::vector<INT> no_replacement_shuffle(dqrng::rng64_t &rng, INT m, INT n,
                                        int offset = 0) {
  std::vector<INT> tmp(m);
  std::iota(tmp.begin(), tmp.end(), static_cast<INT>(offset));
  for (INT i = 0; i < n; ++i)
    std::swap(tmp[i], tmp[i + (*rng)(m - i)]);
  if (m == n)
    return tmp;
  return std::vector<INT>(tmp.begin(), tmp.begin() + n);
}

template <typename INT, typename SET>
std::vector<INT> no_replacement_set(dqrng::rng64_t &rng, INT m, INT n,
                                    int offset) {
  std::vector<INT> result(n);

  SET elems(m, n);
  for (INT i = 0; i < n; ++i) {
    INT v = (*rng)(m);
    while (!elems.insert(v)) {
      v = (*rng)(m);
    }
    result[i] = offset + v;
  }
  return result;
}

template <typename INT>
inline std::vector<INT> sample(dqrng::rng64_t &rng, INT m, INT n,
                               int offset = 0) {
  if (!(m >= n))
    Rcpp::stop("Argument requirements not fulfilled: m >= n");
  if (m < 2 * n) {
    return no_replacement_shuffle<INT>(rng, m, n, offset);
  } else if (m < 1000 * n) {
    return no_replacement_set<INT, dqrng::minimal_bit_set>(rng, m, n, offset);
  } else {
    return no_replacement_set<INT, dqrng::minimal_hash_set<INT>>(rng, m, n,
                                                                 offset);
  }
}

#endif // RNN_SAMPLE_H

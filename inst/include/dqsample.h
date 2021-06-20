// This code is a minor modification of that in dqrng, which is AGPL licensed.
// Anything in this file is AGPL:
//
// Copyright 2019 James Melville
//
// dqsample.h is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// dqsample.h is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

#ifndef DQSAMPLE_H
#define DQSAMPLE_H

#include <numeric>
#include <utility>

#include "dqrng_generator.h"
#include "minimal_int_set.h"

namespace dqsample {

template <typename INT>
inline auto replacement(dqrng::rng64_t &rng, INT m, INT n, int offset)
    -> std::vector<INT> {
  std::vector<INT> result(n);
  std::generate(result.begin(), result.end(),
                [=, &rng]() { return static_cast<INT>(offset + (*rng)(m)); });
  return result;
}

template <typename INT>
auto no_replacement_shuffle(dqrng::rng64_t &rng, INT m, INT n, int offset = 0)
    -> std::vector<INT> {
  std::vector<INT> tmp(m);
  std::iota(tmp.begin(), tmp.end(), static_cast<INT>(offset));
  for (INT i = 0; i < n; ++i)
    std::swap(tmp[i], tmp[i + (*rng)(m - i)]);
  if (m == n)
    return tmp;
  return std::vector<INT>(tmp.begin(), tmp.begin() + n);
}

template <typename INT, typename SET>
auto no_replacement_set(dqrng::rng64_t &rng, INT m, INT n, int offset)
    -> std::vector<INT> {
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

// Sample size points from the range [0 + offset, n + offset) with or without
// replacement
template <typename INT>
inline auto sample(std::vector<INT> &result, dqrng::rng64_t &rng, INT n,
                   INT size, bool replace = false, int offset = 0) -> bool {
  if (replace || size <= 1) {
    result = replacement<INT>(rng, n, size, offset);
  } else {
    if (n < size)
      return false;
    if (n < 2 * size) {
      result = std::move(no_replacement_shuffle<INT>(rng, n, size, offset));
    } else if (n < 1000 * size) {
      result = std::move(no_replacement_set<INT, dqrng::minimal_bit_set>(
          rng, n, size, offset));
    } else {
      result = std::move(no_replacement_set<INT, dqrng::minimal_hash_set<INT>>(
          rng, n, size, offset));
    }
  }
  return true;
}

} // namespace dqsample

#endif // DQSAMPLE_H

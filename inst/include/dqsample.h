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
#include <vector>

#include "dqrng_generator.h"
#include "minimal_int_set.h"

namespace dqsample {

// sample size items in the range [0 + offset, end + offset)
template <typename INT>
inline auto replacement(std::shared_ptr<dqrng::random_64bit_generator> &rng, INT end, INT size, int offset)
    -> std::vector<INT> {
  std::vector<INT> result(size);
  auto generator = [=, &rng]() -> INT {
    return static_cast<INT>(offset + (*rng)(end));
  };
  std::generate(result.begin(), result.end(), generator);
  return result;
}

template <typename INT>
auto no_replacement_shuffle(std::shared_ptr<dqrng::random_64bit_generator> &rng, INT end, INT size,
                            int offset = 0) -> std::vector<INT> {
  std::vector<INT> tmp(end);
  std::iota(tmp.begin(), tmp.end(), static_cast<INT>(offset));
  for (INT i = 0; i < size; ++i) {
    std::swap(tmp[i], tmp[i + (*rng)(end - i)]);
  }
  if (end == size) {
    return tmp;
  }
  return std::vector<INT>(tmp.begin(), tmp.begin() + size);
}

template <typename INT, typename SET>
auto no_replacement_set(std::shared_ptr<dqrng::random_64bit_generator> &rng, INT end, INT size, int offset)
    -> std::vector<INT> {
  std::vector<INT> result(size);

  SET elems(end, size);
  for (INT i = 0; i < size; ++i) {
    INT val = (*rng)(end);
    while (!elems.insert(val)) {
      val = (*rng)(end);
    }
    result[i] = offset + val;
  }
  return result;
}

// Sample size points from the range [0 + offset, end + offset) with or without
// replacement
template <typename INT>
inline auto sample(std::vector<INT> &result, std::shared_ptr<dqrng::random_64bit_generator> &rng, INT end,
                   INT size, bool replace = false, int offset = 0) -> bool {
  if (replace || size <= 1) {
    result = replacement<INT>(rng, end, size, offset);
  } else {
    if (end < size) {
      return false;
    }
    if (end < 2 * size) {
      result = std::move(no_replacement_shuffle<INT>(rng, end, size, offset));
    } else if (end < 1000 * size) {
      result = std::move(no_replacement_set<INT, dqrng::minimal_bit_set>(
          rng, end, size, offset));
    } else {
      result = std::move(no_replacement_set<INT, dqrng::minimal_hash_set<INT>>(
          rng, end, size, offset));
    }
  }
  return true;
}

} // namespace dqsample

#endif // DQSAMPLE_H

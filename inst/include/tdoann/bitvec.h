// BSD 2-Clause License
//
// Copyright 2021 James Melville
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

#ifndef TDOANN_BITVEC_H
#define TDOANN_BITVEC_H

#include <bitset>
#include <cmath>
#include <vector>

namespace tdoann {
static constexpr uint32_t BITVEC_BIT_WIDTH = 64;
template <uint32_t n> using BitSet = std::bitset<n>;
using BitVec = std::vector<BitSet<BITVEC_BIT_WIDTH>>;

// Calculate the number of bitsets of size BITVEC_BIT_WIDTH required to account
// for a binary string of num_bits
inline auto num_blocks_needed(std::size_t num_bits) -> std::size_t {
  return std::ceil(static_cast<float>(num_bits) /
                   static_cast<float>(BITVEC_BIT_WIDTH));
}

// Instead of storing each bit as an element, we will pack them
// into a series of BITVEC_BIT_WIDTH-bit bitsets. Possibly compilers are smart
// enough to use built in integer popcount routines for the bitset count()
// method.
template <typename T> BitVec to_bitvec(T &&vec, std::size_t ndim) {
  const std::size_t n = vec.size() / ndim;
  const std::size_t num_blocks = num_blocks_needed(ndim);

  BitVec bitvec;
  bitvec.reserve(n * num_blocks);

  auto vec_copy = std::forward<T>(vec);

  // each binary string
  for (std::size_t i = 0; i < vec_copy.size(); i += ndim) {
    // each block of of binary data
    for (std::size_t j = 0; j < num_blocks; j++) {
      BitSet<BITVEC_BIT_WIDTH> bits;
      // each bit
      for (std::size_t k = 0;
           k < BITVEC_BIT_WIDTH && (j * BITVEC_BIT_WIDTH + k) < ndim; k++) {
        bits[k] = vec_copy[i + j * BITVEC_BIT_WIDTH + k];
      }
      bitvec.push_back(bits);
    }
  }

  return bitvec;
}

} // namespace tdoann

#endif // TDOANN_BITVEC_H

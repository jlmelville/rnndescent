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
static const unsigned int BITVEC_BIT_WIDTH = 64;
template <unsigned int n> using BitSet = std::bitset<n>;
using BitVec = std::vector<BitSet<BITVEC_BIT_WIDTH>>;

inline auto bitvec_size(std::size_t nbits) -> std::size_t {
  return std::ceil(nbits / float{BITVEC_BIT_WIDTH});
}

// Instead of storing each bit as an element, we will pack them
// into a series of BITVEC_BIT_WIDTH-bit bitsets. Possibly compilers are smart
// enough to use built in integer popcount routines for the bitset count()
// method. Relies on NRVO to avoid copying return value
template <typename T>
auto to_bitvec(const std::vector<T> &vec, std::size_t ndim) -> BitVec {
  BitSet<BITVEC_BIT_WIDTH> bits;
  std::size_t bit_count = 0;
  std::size_t vd_count = 0;

  BitVec bitvec;

  for (std::size_t i = 0; i < vec.size(); i++) {
    if (bit_count == bits.size() || vd_count == ndim) {
      // filled up current bitset
      bitvec.push_back(bits);
      bit_count = 0;
      bits.reset();

      if (vd_count == ndim) {
        // end of item
        vd_count = 0;
      }
    }
    bits[bit_count] = vec[i];

    ++vd_count;
    ++bit_count;
  }
  if (bit_count > 0) {
    bitvec.push_back(bits);
  }

  return bitvec;
}
} // namespace tdoann

#endif // TDOANN_BITVEC_H

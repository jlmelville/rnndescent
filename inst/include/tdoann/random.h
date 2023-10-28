// BSD 2-Clause License
//
// Copyright 2023 James Melville
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

#ifndef TDOANN_RANDOM_H
#define TDOANN_RANDOM_H

#include <cstdint>
#include <memory>

namespace tdoann {

// Needed for random sampling in various places, e.g. search graph pruning
// and nearest neighbor descent
class RandomGenerator {
public:
  virtual ~RandomGenerator() = default;

  // A random uniform value between 0 and 1
  virtual double unif() = 0;
};

// This wraps a RNG in such a way that it can be used in a parallel fashion
// Note that it is only deterministic for a given number of threads
// (because we use the end of each "window" into the range being processed as
// one of the seeds for each thread's generator)
class ParallelRandomProvider {
public:
  virtual ~ParallelRandomProvider() = default;

  // do internal generation and storage of a random number based on an external
  // seed to ensure reproducibility
  virtual void initialize() = 0;
  // create a RandomGenerator of uniform floats inside a thread
  virtual std::unique_ptr<RandomGenerator>
  get_parallel_instance(uint64_t seed2) = 0;
};

// Needed for random sampling of integers
template <typename Int = uint32_t> class RandomIntGenerator {
public:
  virtual ~RandomIntGenerator() = default;

  // Generates a random integer in range [0, n)
  virtual Int rand_int(Int n) = 0;

  // Generates n_ints random integers in range [0, max_val)
  virtual std::vector<Int> sample(int max_val, int n_ints) = 0;
};

template <typename Int> class ParallelRandomIntProvider {
public:
  virtual ~ParallelRandomIntProvider() = default;

  virtual void initialize() = 0;
  virtual std::unique_ptr<RandomIntGenerator<Int>>
  get_parallel_instance(uint64_t seed2) = 0;
};

} // namespace tdoann

#endif // TDOANN_RANDOM_H

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

#ifndef TDOANN_PROGRESSBAR_H
#define TDOANN_PROGRESSBAR_H

#include <ostream>

#ifndef TDOANN_PROGRESSBAR_OUTPUT_STREAM
#include <iostream>
#define TDOANN_PROGRESSBAR_OUTPUT_STREAM std::cout
#endif

namespace tdoann {

class ProgressBar {
private:
  static constexpr unsigned TOTAL_STEPS = 51;

  unsigned max;
  bool verbose;
  unsigned previous_value;
  std::ostream *pout; // Using a pointer to allow for moving

  unsigned calculate_stars(unsigned value) const {
    return static_cast<unsigned>((value * TOTAL_STEPS) / max + 0.5);
  }

public:
  ProgressBar(unsigned max, bool verbose,
              std::ostream &os = TDOANN_PROGRESSBAR_OUTPUT_STREAM)
      : max(max), verbose(verbose), previous_value(0), pout(&os) {
    initialize();
  }

  ProgressBar(ProgressBar &&other) noexcept
      : max(std::exchange(other.max, 0)),
        verbose(std::exchange(other.verbose, false)),
        previous_value(std::exchange(other.previous_value, 0)),
        pout(std::exchange(other.pout, nullptr)) {}

  ProgressBar &operator=(ProgressBar &&other) noexcept {
    if (this != &other) {
      max = other.max;
      verbose = other.verbose;
      previous_value = other.previous_value;
      pout = other.pout;
      other.pout = nullptr;
    }
    return *this;
  }

  ~ProgressBar() { cleanup(); }

  void initialize() {
    if (verbose) {
      (*pout) << "0%   10   20   30   40   50   60   70   80   90   100%"
              << std::endl;
      (*pout) << "[----|----|----|----|----|----|----|----|----|----]"
              << std::endl;
      pout->flush();
    }
  }

  void update(unsigned value) {
    if (!verbose) {
      return;
    }

    // Ensure value does not exceed max
    value = std::min(value, max);

    if (value <= previous_value)
      return;

    unsigned new_stars =
        calculate_stars(value) - calculate_stars(previous_value);

    for (unsigned i = 0; i < new_stars; ++i) {
      (*pout) << "*";
    }

    if (value == max) {
      (*pout) << "\n";
    }
    pout->flush();

    previous_value = value;
  }

  // Cleanup method
  void cleanup() {
    if (verbose) {
      update(max); // Display 100%
    }
  }
};

} // namespace tdoann

#endif // TDOANN_PROGRESSBAR_H

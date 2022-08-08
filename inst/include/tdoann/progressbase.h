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

#ifndef TDOANN_PROGRESS_BASE_H
#define TDOANN_PROGRESS_BASE_H

#include <string>

#define TDOANN_BREAKIFINTERRUPTED()                                            \
  if (progress.check_interrupt()) {                                            \
    break;                                                                     \
  }

#define TDOANN_ITERFINISHED()                                                  \
  TDOANN_BREAKIFINTERRUPTED()                                                  \
  progress.iter_finished();

#define TDOANN_BLOCKFINISHED()                                                 \
  TDOANN_BREAKIFINTERRUPTED()                                                  \
  progress.block_finished();

inline auto is_converged(std::size_t n_updates, double tol) -> bool {
  return static_cast<double>(n_updates) <= tol;
}

namespace tdoann {
// Defines the methods required, but does nothing. Safe to use from
// multi-threaded code if a dummy no-op version is needed.
struct NullProgress {
  NullProgress() = default;
  NullProgress(std::size_t /* niters */, bool /* verbose */) {}
  void set_n_blocks(std::size_t /* n */) {}
  void block_finished() {}
  void iter_finished() {}
  void stopping_early() {}
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  auto check_interrupt() -> bool { return false; }
  void converged(std::size_t /* n_updates */, double /* tol */) {}
  void log(const std::string & /* msg */) {}
};

} // namespace tdoann

#endif // TDOANN_PROGRESS_BASE_H

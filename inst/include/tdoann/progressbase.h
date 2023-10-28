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

namespace tdoann {

class ProgressBase {
public:
  // Constructors
  ProgressBase() = default;
  ProgressBase(std::size_t /* niters */, bool /* verbose */) {}
  explicit ProgressBase(bool /* verbose */) {}

  // Virtual Destructor
  virtual ~ProgressBase() = default;

  // Virtual Methods
  virtual void set_n_iters(uint32_t /* n */) {}
  virtual void set_n_batches(uint32_t /* n */) {}
  virtual void batch_finished() {}
  virtual void iter_finished() {}
  virtual void stopping_early() {}
  virtual void log(const std::string & /* msg */) const {}
  virtual auto check_interrupt() -> bool { return false; }
  virtual auto is_verbose() const -> bool { return false; }
};

// No-op implementation
class NullProgress : public ProgressBase {
public:
  using ProgressBase::ProgressBase;
};

} // namespace tdoann

#endif // TDOANN_PROGRESS_BASE_H

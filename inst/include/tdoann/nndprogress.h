// BSD 2-Clause License
//
// Copyright 2020 James Melville
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

#ifndef TDOANN_NNDPROGRESS_H
#define TDOANN_NNDPROGRESS_H

#include <memory>
#include <sstream>
#include <string>

#include "progressbase.h"

namespace tdoann {

// This enum exists because you can't have a virtual heap_report(NeighborHeap)
// method because that method would have to be templated.
enum class ReportingAction { HeapSum, DoNothing };

class NNDProgressBase {
public:
  virtual ~NNDProgressBase() = default;

  virtual ProgressBase &get_base_progress() = 0;

  virtual void set_n_blocks(std::size_t n) = 0;
  virtual void block_finished() = 0;
  virtual void iter_finished() = 0;
  virtual void stopping_early() = 0;
  virtual bool check_interrupt() = 0;

  virtual void log(const std::string &msg) = 0;
  virtual void converged(std::size_t n_updates, double tol) = 0;
  virtual ReportingAction get_reporting_action() const = 0;
};

class NNDProgress : public NNDProgressBase {
private:
  std::unique_ptr<ProgressBase> progress;

public:
  explicit NNDProgress(std::unique_ptr<ProgressBase> p)
      : progress(std::move(p)) {}

  NNDProgress(NNDProgress &&other) noexcept
      : progress(std::move(other.progress)) {}

  ProgressBase &get_base_progress() override { return *progress; }

  void set_n_blocks(std::size_t n) override { progress->set_n_blocks(n); }
  void block_finished() override { progress->block_finished(); }
  void iter_finished() override { progress->iter_finished(); }
  void stopping_early() override { progress->stopping_early(); }
  bool check_interrupt() override { return progress->check_interrupt(); }

  void log(const std::string &msg) override { progress->log(msg); }
  void converged(std::size_t n_updates, double tol) override {
    stopping_early();
    if (progress->is_verbose()) {
      std::ostringstream oss;
      oss << "Convergence: c = " << n_updates << " tol = " << tol;
      log(oss.str());
    }
  }
  ReportingAction get_reporting_action() const override {
    return ReportingAction::DoNothing;
  }
};

class HeapSumProgress : public NNDProgress {
public:
  using NNDProgress::NNDProgress; // Inherit constructors

  ReportingAction get_reporting_action() const override {
    return ReportingAction::HeapSum;
  }
};

template <typename Progress, typename NeighborHeap>
void pr(Progress &progress, const NeighborHeap &neighbor_heap) {
  pr(progress, neighbor_heap, "");
}

template <typename Progress, typename NeighborHeap>
void pr(Progress &progress, const NeighborHeap &neighbor_heap,
        const std::string &header) {
  typename NeighborHeap::Index n_points = neighbor_heap.n_points;
  std::size_t n_nbrs = neighbor_heap.n_nbrs;

  std::ostringstream os_out;
  os_out << header << std::endl;
  for (typename NeighborHeap::Index i = 0; i < n_points; i++) {
    std::size_t innbrs = i * n_nbrs;
    os_out << i << ": ";
    for (std::size_t j = 0; j < n_nbrs; j++) {
      auto idx = neighbor_heap.idx[innbrs + j];
      if (idx == neighbor_heap.npos()) {
        os_out << "-1 ";
      } else {
        os_out << neighbor_heap.idx[innbrs + j] << " ";
      }
    }
    os_out << std::endl;
  }
  for (typename NeighborHeap::Index i = 0; i < n_points; i++) {
    std::size_t innbrs = i * n_nbrs;
    os_out << i << ": ";
    for (std::size_t j = 0; j < n_nbrs; j++) {
      if (neighbor_heap.idx[innbrs + j] == neighbor_heap.npos()) {
        os_out << "NA ";
      } else {
        os_out << neighbor_heap.dist[innbrs + j] << " ";
      }
    }
    os_out << std::endl;
  }
  progress.log(os_out.str());
}

inline auto is_converged(std::size_t n_updates, double tol) -> bool {
  return static_cast<double>(n_updates) <= tol;
}

template <typename NbrHeap>
auto nnd_should_stop(NNDProgressBase &progress, const NbrHeap &nn_heap,
                     std::size_t num_updated, double tol) -> bool {
  if (progress.check_interrupt()) {
    return true;
  }
  progress.iter_finished();

  if (progress.get_reporting_action() == ReportingAction::HeapSum) {
    double heap_sum_value = heap_sum(nn_heap);
    std::ostringstream oss;
    oss << "heap sum = " << heap_sum_value;
    progress.log(oss.str());
  }

  if (is_converged(num_updated, tol)) {
    progress.converged(num_updated, tol);
    return true;
  }

  return false;
}

} // namespace tdoann

#endif // TDOANN_NNDPROGRESS_H

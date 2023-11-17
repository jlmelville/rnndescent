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

#include "heap.h"
#include "progressbase.h"

namespace tdoann {

// This enum exists because you can't have a virtual heap_report(NeighborHeap)
// method because that method would have to be templated.
enum class ReportingAction { HeapSum, DoNothing };

class NNDProgressBase {
public:
  virtual ~NNDProgressBase() = default;

  virtual ProgressBase &get_base_progress() = 0;

  virtual void set_n_batches(uint32_t n) = 0;
  virtual void batch_finished() = 0;
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

  void set_n_batches(uint32_t n) override { progress->set_n_batches(n); }
  void batch_finished() override { progress->batch_finished(); }
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
  constexpr auto npos = static_cast<typename NeighborHeap::Index>(-1);

  typename NeighborHeap::Index n_points = neighbor_heap.n_points;
  std::size_t n_nbrs = neighbor_heap.n_nbrs;

  std::ostringstream os_out;
  os_out << header << "\n";
  for (typename NeighborHeap::Index i = 0; i < n_points; i++) {
    std::size_t innbrs = i * n_nbrs;
    os_out << i << ": ";
    for (std::size_t j = 0; j < n_nbrs; j++) {
      auto idx = neighbor_heap.idx[innbrs + j];
      if (idx == npos) {
        os_out << "-1 ";
      } else {
        os_out << neighbor_heap.idx[innbrs + j] << " ";
      }
    }
    os_out << "\n";
  }
  for (typename NeighborHeap::Index i = 0; i < n_points; i++) {
    std::size_t innbrs = i * n_nbrs;
    os_out << i << ": ";
    for (std::size_t j = 0; j < n_nbrs; j++) {
      if (neighbor_heap.idx[innbrs + j] == npos) {
        os_out << "NA ";
      } else {
        os_out << neighbor_heap.dist[innbrs + j] << " ";
      }
    }
    os_out << "\n";
  }
  progress.log(os_out.str());
}

inline auto is_converged(unsigned long num_updates, double tol) -> bool {
  return static_cast<double>(num_updates) <= tol;
}

template <typename NbrHeap>
auto nnd_should_stop(NNDProgressBase &progress, const NbrHeap &nn_heap,
                     unsigned long num_updates, double delta) -> bool {
  if (progress.check_interrupt()) {
    return true;
  }
  progress.iter_finished();

  const double tol = delta * nn_heap.n_nbrs * nn_heap.n_points;
  if (progress.get_reporting_action() == ReportingAction::HeapSum) {
    double heap_sum_value = heap_sum(nn_heap);
    std::ostringstream oss;
    oss << "heap sum = " << heap_sum_value << " num_updates = " << num_updates
        << " tol = " << tol;
    progress.log(oss.str());
  }

  if (is_converged(num_updates, tol)) {
    progress.converged(num_updates, tol);
    return true;
  }

  return false;
}

// A cache of previously seen edges (potential neighbors) used in caching
// variants of the local join process
template <typename Idx> struct EdgeCache {
private:
  std::vector<std::unordered_set<Idx>> seen;

public:
  EdgeCache(std::size_t n_points, std::size_t n_nbrs,
            const std::vector<Idx> &idx_data)
      : seen(n_points) {
    for (Idx i = 0, innbrs = 0; i < n_points; i++, innbrs += n_nbrs) {
      for (std::size_t j = 0, idx_ij = innbrs; j < n_nbrs; j++, idx_ij++) {
        auto idx_p = idx_data[idx_ij];
        if (i > idx_p) {
          seen[idx_p].emplace(i);
        } else {
          seen[i].emplace(idx_p);
        }
      }
    }
  }

  // Static factory function
  template <typename Out>
  static EdgeCache<Idx> from_graph(const NNDHeap<Out, Idx> &heap) {
    return EdgeCache<Idx>(heap.n_points, heap.n_nbrs, heap.idx);
  }

  auto contains(const Idx &idx_p, const Idx &idx_q) const -> bool {
    return seen[idx_p].find(idx_q) != seen[idx_p].end();
  }

  auto insert(Idx idx_p, Idx idx_q) -> bool {
    return !seen[idx_p].emplace(idx_q).second;
  }
};

// mark any neighbor in the current graph that was retained in the new
// candidates as false
template <typename Out, typename Idx>
void flag_retained_new_candidates(NNDHeap<Out, Idx> &current_graph,
                                  const NNHeap<Out, Idx> &new_nbrs,
                                  std::size_t begin, std::size_t end) {
  constexpr auto npos = static_cast<Idx>(-1);
  const std::size_t n_nbrs = current_graph.n_nbrs;
  std::size_t ibegin = begin * n_nbrs;
  std::size_t ij = ibegin;
  for (auto i = begin; i < end; i++, ibegin += n_nbrs) {
    for (std::size_t j = 0; j < n_nbrs; j++, ij++) {
      const auto &nbr = current_graph.idx[ij];
      if (nbr == npos) {
        continue;
      }
      if (new_nbrs.contains(i, nbr)) {
        current_graph.flags[ij] = 0;
      }
    }
  }
}
} // namespace tdoann

#endif // TDOANN_NNDPROGRESS_H

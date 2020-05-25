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

#ifndef TDOANN_PROGRESS_H
#define TDOANN_PROGRESS_H

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

#define TDOANN_CHECKCONVERGENCE()                                              \
  if (is_converged(c, tol)) {                                                  \
    progress.converged(c, tol);                                                \
    break;                                                                     \
  }

namespace tdoann {
// Defines the methods required, but does nothing. Safe to use from
// multi-threaded code if a dummy no-op version is needed.
struct NullProgress {
  NullProgress() = default;
  NullProgress(std::size_t, bool) {}
  void set_n_blocks(std::size_t) {}
  void block_finished() {}
  void iter_finished() {}
  void stopping_early() {}
  auto check_interrupt() -> bool { return false; }
  void converged(std::size_t, double) {}
  void log(const std::string &) {}
};

template <typename Progress> struct NNDProgress {
  Progress progress;

  NNDProgress(Progress &progress) : progress(progress) {}
  void set_n_blocks(std::size_t n) { progress.set_n_blocks(n); }
  void block_finished() { progress.block_finished(); }
  void iter_finished() { progress.iter_finished(); }
  void stopping_early() { progress.stopping_early(); }
  auto check_interrupt() -> bool { return progress.check_interrupt(); }
  void converged(std::size_t n_updates, double tol) {
    progress.converged(n_updates, tol);
  }
  void log(const std::string &msg) { progress.log(msg); }
  template <typename NeighborHeap> void heap_report(const NeighborHeap &) {}
};

template <typename Progress> struct HeapSumProgress {
  Progress progress;

  HeapSumProgress(Progress &progress) : progress(progress) {}
  void set_n_blocks(std::size_t n) { progress.set_n_blocks(n); }
  void block_finished() { progress.block_finished(); }
  void iter_finished() { progress.iter_finished(); }
  void stopping_early() { progress.stopping_early(); }
  auto check_interrupt() -> bool { return progress.check_interrupt(); }
  void converged(std::size_t n_updates, double tol) {
    progress.converged(n_updates, tol);
  }
  void log(const std::string &msg) { progress.log(msg); }
  template <typename NeighborHeap>
  void heap_report(const NeighborHeap &neighbor_heap) {
    typename NeighborHeap::Index n_points = neighbor_heap.n_points;
    std::size_t n_nbrs = neighbor_heap.n_nbrs;
    typename NeighborHeap::DistanceOut hsum = 0.0;
    for (typename NeighborHeap::Index i = 0; i < n_points; i++) {
      std::size_t innbrs = i * n_nbrs;
      for (std::size_t j = 0; j < n_nbrs; j++) {
        hsum += neighbor_heap.dist[innbrs + j];
      }
    }
    std::ostringstream os;
    os << "heap sum = " << hsum;
    log(os.str());
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

  std::ostringstream os;
  os << header << std::endl;
  for (typename NeighborHeap::Index i = 0; i < n_points; i++) {
    std::size_t innbrs = i * n_nbrs;
    os << i << ": ";
    for (std::size_t j = 0; j < n_nbrs; j++) {
      auto idx = neighbor_heap.idx[innbrs + j];
      if (idx == neighbor_heap.npos()) {
        os << "-1 ";
      } else {
        os << neighbor_heap.idx[innbrs + j] << " ";
      }
    }
    os << std::endl;
  }
  for (typename NeighborHeap::Index i = 0; i < n_points; i++) {
    std::size_t innbrs = i * n_nbrs;
    os << i << ": ";
    for (std::size_t j = 0; j < n_nbrs; j++) {
      if (neighbor_heap.idx[innbrs + j] == neighbor_heap.npos()) {
        os << "NA ";
      } else {
        os << neighbor_heap.dist[innbrs + j] << " ";
      }
    }
    os << std::endl;
  }
  progress.log(os.str());
}
} // namespace tdoann

#endif // TDOANN_PROGRESS_H

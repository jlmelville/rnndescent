//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2019 James Melville
//
//  This file is part of rnndescent
//
//  rnndescent is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  rnndescent is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with rnndescent.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RNN_PROGRESS_H
#define RNN_PROGRESS_H

#include <Rcpp.h>
#include <progress.hpp>

#include "tdoann/heap.h"
#include "tdoann/progress.h"

#include "rnn_util.h"

struct RPProgress {
  std::size_t scale;
  Progress progress;
  std::size_t n_iters;
  std::size_t n_blocks_;
  bool verbose;

  std::size_t iter;
  std::size_t block;
  bool is_aborted;

  RPProgress(std::size_t n_iters, bool verbose)
      : scale(100), progress(scale, verbose), n_iters(n_iters), n_blocks_(0),
        verbose(verbose), iter(0), block(0), is_aborted(false) {}

  void set_n_blocks(std::size_t n_blocks) {
    n_blocks_ = n_blocks;
    block = 0;
  }
  void block_finished() {
    ++block;
    if (verbose) {
      progress.update(scaled(iter + (static_cast<double>(block) / n_blocks_)));
    }
  }
  void iter_finished() {
    if (verbose) {
      ++iter;
      progress.update(scaled(iter));
    }
  }
  void stopping_early() {
    progress.update(n_iters);
    progress.cleanup();
  }
  auto check_interrupt() -> bool {
    if (is_aborted) {
      return true;
    }
    if (Progress::check_abort()) {
      stopping_early();
      is_aborted = true;
      return true;
    }
    return false;
  }
  void converged(std::size_t n_updates, double tol) {
    stopping_early();
    if (verbose) {
      std::ostringstream os;
      os << "Convergence at iteration " << iter << ": c = " << n_updates
         << " tol = " << tol;
      log(os.str());
    }
  }
  auto scaled(double d) -> int {
    int res = std::nearbyint(scale * (d / n_iters));
    return res;
  }

  void log(const std::string &msg) {
    if (verbose) {
      Rcpp::Rcerr << msg << std::endl;
    }
  }
};

struct RInterruptableProgress {
  bool is_aborted{false};

  RInterruptableProgress();
  RInterruptableProgress(std::size_t, bool);
  void set_n_blocks(std::size_t) {}
  void block_finished() {}
  void iter_finished() {}
  void stopping_early() {}
  void converged(std::size_t, double) {}
  void log(const std::string &) {}
  auto check_interrupt() -> bool;
};

struct RIterProgress {
  std::size_t n_iters;
  bool verbose;

  std::size_t iter;
  bool is_aborted;

  RIterProgress(std::size_t n_iters, bool verbose)
      : n_iters(n_iters), verbose(verbose), iter(0), is_aborted(false) {
    iter_msg(0);
  }

  void iter_msg(std::size_t iter) const {
    if (verbose) {
      std::ostringstream os;
      os << iter << " / " << n_iters;
      log(os.str());
    }
  }
  void set_n_blocks(std::size_t) {}
  void block_finished() {}
  void iter_finished() {
    ++iter;
    iter_msg(iter);
  }
  void stopping_early() {}
  auto check_interrupt() -> bool {
    if (is_aborted) {
      return true;
    }
    try {
      Rcpp::checkUserInterrupt();
    } catch (Rcpp::internal::InterruptedException &) {
      is_aborted = true;
      stopping_early();
      return true;
    }
    return false;
  }
  void converged(std::size_t n_updates, double tol) {
    if (verbose) {
      std::ostringstream os;
      os << "Convergence: c = " << n_updates << " tol = " << tol;
      log(os.str());
    }
    stopping_early();
  }
  void log(const std::string &msg) const {
    if (verbose) {
      ts(msg);
    }
  }
};

#endif // RNN_PROGRESS_H

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
  static const constexpr std::size_t scale{100};
  Progress progress;
  std::size_t n_iters;
  std::size_t n_blocks_{0};
  bool verbose;

  std::size_t iter{0};
  std::size_t block{0};
  bool is_aborted{false};

  double iter_increment;  // Amount progress increases per iteration
  double block_increment; // Amount progress increases per block

  RPProgress(std::size_t n_iters, bool verbose)
      : progress(scale, verbose), n_iters(n_iters), verbose(verbose) {
    iter_increment = static_cast<double>(scale) / n_iters;
    // Default block_increment if no blocks are set
    block_increment = iter_increment;
  }

  void set_n_blocks(std::size_t n_blocks) {
    n_blocks_ = n_blocks;
    block = 0;
    block_increment = iter_increment / n_blocks_;
  }

  void block_finished() {
    ++block;
    if (verbose) {
      unsigned long progress_val =
          static_cast<unsigned long>(std::round(block * block_increment));
      progress.update(
          std::min(progress_val, static_cast<unsigned long>(scale)));
    }
  }

  void iter_finished() {
    if (verbose) {
      ++iter;
      unsigned long progress_val =
          static_cast<unsigned long>(std::round(iter * iter_increment));
      progress.update(
          std::min(progress_val, static_cast<unsigned long>(scale)));
    }
  }

  void stopping_early() {
    progress.update(scale); // Directly set progress to 100%
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
      std::ostringstream oss;
      oss << "Convergence at iteration " << iter << ": c = " << n_updates
          << " tol = " << tol;
      log(oss.str());
    }
  }

  void log(const std::string &msg) const {
    if (verbose) {
      Rcpp::Rcerr << msg << std::endl;
    }
  }
};

struct RInterruptableProgress {
  bool is_aborted{false};

  RInterruptableProgress();
  RInterruptableProgress(std::size_t /* n_iters */, bool /* verbose */);
  void set_n_blocks(std::size_t /* n_blocks */) {}
  void block_finished() {}
  void iter_finished() {}
  void stopping_early() {}
  void converged(std::size_t /* n_updates */, double /* tol */) {}
  void log(const std::string & /* msg */) {}
  auto check_interrupt() -> bool;
};

struct RIterProgress {
  std::size_t n_iters;
  bool verbose;

  std::size_t iter{0};
  bool is_aborted{false};

  RIterProgress(std::size_t n_iters, bool verbose)
      : n_iters(n_iters), verbose(verbose) {
    iter_msg(0);
  }

  void iter_msg(std::size_t iter) const {
    if (verbose) {
      std::ostringstream oss;
      oss << iter << " / " << n_iters;
      log(oss.str());
    }
  }
  void set_n_blocks(std::size_t /* n_blocks */) {}
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
      std::ostringstream oss;
      oss << "Convergence: c = " << n_updates << " tol = " << tol;
      log(oss.str());
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

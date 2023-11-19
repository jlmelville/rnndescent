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

#include "tdoann/heap.h"
#include "tdoann/nndcommon.h"
#define TDOANN_PROGRESSBAR_OUTPUT_STREAM Rcpp::Rcerr
#include "tdoann/progressbar.h"

#include "rnn_util.h"

struct RPProgress : public tdoann::ProgressBase {
  static constexpr std::size_t scale{100};
  tdoann::ProgressBar progress;
  bool verbose;

  uint32_t iter{0};
  uint32_t batch{0};
  bool is_aborted{false};

  double iter_increment{scale};  // Amount progress increases per iteration
  double batch_increment{scale}; // Amount progress increases per batch

  virtual ~RPProgress() = default;

  RPProgress(bool verbose) : progress(scale, verbose), verbose(verbose) {}

  RPProgress(uint32_t n_iters, bool verbose)
      : progress(scale, verbose), verbose(verbose),
        iter_increment(static_cast<double>(scale) / n_iters) {}

  RPProgress(RPProgress &&other) noexcept
      : progress(std::move(other.progress)), verbose(std::move(other.verbose)),
        iter(std::move(other.iter)), batch(std::move(other.batch)),
        is_aborted(std::move(other.is_aborted)),
        iter_increment(std::move(other.iter_increment)),
        batch_increment(std::move(other.batch_increment)) {}

  RPProgress &operator=(RPProgress &&other) noexcept {
    if (this != &other) {
      progress = std::move(other.progress);
      verbose = std::move(other.verbose);
      iter = std::move(other.iter);
      batch = std::move(other.batch);
      is_aborted = std::move(other.is_aborted);
      iter_increment = std::move(other.iter_increment);
      batch_increment = std::move(other.batch_increment);
    }
    return *this;
  }

  void set_n_iters(uint32_t n_iters) override {
    iter_increment = static_cast<double>(scale) / n_iters;
  }

  void set_n_batches(uint32_t n_batches) override {
    batch = 0;
    batch_increment = iter_increment / n_batches;
  }

  void batch_finished() override {
    ++batch;
    if (verbose) {
      unsigned long progress_val =
          static_cast<unsigned long>(std::round(batch * batch_increment));
      progress.update(
          std::min(progress_val, static_cast<unsigned long>(scale)));
    }
  }

  void iter_finished() override {
    if (verbose) {
      ++iter;
      unsigned long progress_val =
          static_cast<unsigned long>(std::round(iter * iter_increment));
      progress.update(
          std::min(progress_val, static_cast<unsigned long>(scale)));
    }
  }

  bool check_interrupt() override {
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

  void log(const std::string &msg) const override {
    if (verbose) {
      ts(msg);
    }
  }

  void stopping_early() override { progress.cleanup(); }

  bool is_verbose() const override { return verbose; }
};

struct RInterruptableProgress : public tdoann::ProgressBase {
  bool is_aborted{false};
  bool verbose;

  virtual ~RInterruptableProgress() = default;

  RInterruptableProgress() = default;
  RInterruptableProgress(uint32_t /* n_iters */, bool verbose)
      : verbose(verbose) {}

  RInterruptableProgress(RInterruptableProgress &&other) noexcept
      : is_aborted(std::move(other.is_aborted)),
        verbose(std::move(other.verbose)) {
    other.is_aborted = false;
    other.verbose = false;
  }

  RInterruptableProgress &operator=(RInterruptableProgress &&other) noexcept {
    if (this != &other) {
      is_aborted = std::move(other.is_aborted);
      verbose = std::move(other.verbose);

      other.is_aborted = false;
      other.verbose = false;
    }
    return *this;
  }

  void log(const std::string & /* msg */) const override {}

  bool check_interrupt() override {
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

  bool is_verbose() const override { return verbose; }
};

struct RIterProgress : public RInterruptableProgress {
  uint32_t n_iters;
  uint32_t iter{0};

  virtual ~RIterProgress() = default;

  RIterProgress(uint32_t n_iters, bool verbose)
      : RInterruptableProgress(n_iters, verbose), n_iters(n_iters) {}
  RIterProgress(RIterProgress &&other) noexcept
      : RInterruptableProgress(std::move(other)),
        n_iters(std::move(other.n_iters)), iter(std::move(other.iter)) {

    other.n_iters = 0;
    other.iter = 0;
  }
  RIterProgress &operator=(RIterProgress &&other) noexcept {
    if (this != &other) {
      RInterruptableProgress::operator=(std::move(other));

      n_iters = std::move(other.n_iters);
      iter = std::move(other.iter);

      other.n_iters = 0;
      other.iter = 0;
    }
    return *this;
  }

  void iter_msg(uint32_t iter) const {
    if (verbose) {
      std::ostringstream oss;
      oss << iter << " / " << n_iters;
      log(oss.str());
    }
  }

  void set_n_iters(uint32_t n_iters) override { this->n_iters = n_iters; }

  void iter_finished() override {
    ++iter;
    iter_msg(iter);
  }

  void log(const std::string &msg) const override {
    if (verbose) {
      ts(msg);
    }
  }
};

#endif // RNN_PROGRESS_H

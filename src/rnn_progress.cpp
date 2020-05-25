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

#include <Rcpp.h>

#include "rnn_progress.h"

RInterruptableProgress::RInterruptableProgress() = default;
RInterruptableProgress::RInterruptableProgress(std::size_t, bool) {}
auto RInterruptableProgress::check_interrupt() -> bool {
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

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

#ifndef RNN_DISTANCE_H
#define RNN_DISTANCE_H

#include <Rcpp.h>

#include "tdoann/distance.h"

template <typename T> auto r2vt(Rcpp::NumericMatrix data) -> std::vector<T> {
  return Rcpp::as<std::vector<T>>(Rcpp::transpose(data));
}

template <typename Distance>
auto r2dvt(Rcpp::NumericMatrix data) -> std::vector<typename Distance::Input> {
  return r2vt<typename Distance::Input>(data);
}

#endif // RNN_DISTANCE_H

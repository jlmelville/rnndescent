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

template <typename T>
auto r_to_vec(Rcpp::NumericMatrix data) -> std::vector<T> {
  return Rcpp::as<std::vector<T>>(data);
}

template <typename T>
auto r_to_vect(Rcpp::NumericMatrix data) -> std::vector<T> {
  return Rcpp::as<std::vector<T>>(Rcpp::transpose(data));
}

template <typename Distance>
auto r_to_dist_vect(Rcpp::NumericMatrix data)
    -> std::vector<typename Distance::Input> {
  return r_to_vect<typename Distance::Input>(data);
}

template <typename Distance>
auto r_to_dist(Rcpp::NumericMatrix reference, Rcpp::NumericMatrix query)
    -> Distance {
  auto ref_vec = r_to_dist_vect<Distance>(reference);
  auto query_vec = r_to_dist_vect<Distance>(query);
  return Distance(ref_vec, query_vec, reference.ncol());
}

template <typename Distance>
auto r_to_dist(Rcpp::NumericMatrix data) -> Distance {
  auto data_vec = r_to_dist_vect<Distance>(data);
  return Distance(data_vec, data.ncol());
}

#endif // RNN_DISTANCE_H

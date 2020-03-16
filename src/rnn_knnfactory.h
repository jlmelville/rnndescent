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

#ifndef RNN_KNNFACTORY_H
#define RNN_KNNFACTORY_H

#include <Rcpp.h>

#include "tdoann/distance.h"

template <typename Distance> struct KnnQueryFactory {
  using DataVec = std::vector<typename Distance::Input>;

  DataVec reference_vec;
  DataVec query_vec;
  int n_points;
  int ndim;

  KnnQueryFactory(Rcpp::NumericMatrix reference, Rcpp::NumericMatrix query)
      : reference_vec(Rcpp::as<DataVec>(Rcpp::transpose(reference))),
        query_vec(Rcpp::as<DataVec>(Rcpp::transpose(query))),
        n_points(query.nrow()), ndim(query.ncol()) {}

  Distance create_distance() const {
    return Distance(reference_vec, query_vec, ndim);
  }
};

template <typename Distance> struct KnnBuildFactory {
  using DataVec = std::vector<typename Distance::Input>;

  DataVec data_vec;
  int n_points;
  int ndim;

  KnnBuildFactory(Rcpp::NumericMatrix data)
      : data_vec(Rcpp::as<DataVec>(Rcpp::transpose(data))),
        n_points(data.nrow()), ndim(data.ncol()) {}

  Distance create_distance() const { return Distance(data_vec, ndim); }
};

#endif // RNN_KNNFACTORY_H

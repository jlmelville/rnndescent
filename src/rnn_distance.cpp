//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2022 James Melville
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

// NOLINTBEGIN(modernize-use-trailing-return-type,readability-identifier-length)

#include "tdoann/distance.h"

#include <Rcpp.h>

using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::stop;

template <typename Vec> inline void check_vecs(Vec x, Vec y) {
  if (x.length() != y.length()) {
    stop("x and y are not the same length");
  }
}

//' Find the Euclidean (L2) distance between two vectors
//'
//' @param x A numeric vector.
//' @param y A numeric vector of the same length as \code{x}.
//' @export
// [[Rcpp::export]]
double euclidean_distance(NumericVector x, NumericVector y) {
  check_vecs(x, y);
  return tdoann::euclidean<double>(x.begin(), x.end(), y.begin());
}

//' Find the squared Euclidean (squared L2) distance between two vectors
//'
//' @param x A numeric vector.
//' @param y A numeric vector of the same length as \code{x}.
//' @export
// [[Rcpp::export]]
double l2sqr_distance(NumericVector x, NumericVector y) {
  check_vecs(x, y);
  return tdoann::l2sqr<double>(x.begin(), x.end(), y.begin());
}

//' Find the cosine distance between two vectors
//'
//' @param x A numeric vector.
//' @param y A numeric vector of the same length as \code{x}.
//' @export
// [[Rcpp::export]]
double cosine_distance(NumericVector x, NumericVector y) {
  check_vecs(x, y);
  return tdoann::cosine<double>(x.begin(), x.end(), y.begin());
}

//' Find the Manhattan (L1) distance between two vectors
//'
//' @param x A numeric vector.
//' @param y A numeric vector of the same length as \code{x}.
//' @export
// [[Rcpp::export]]
double manhattan_distance(NumericVector x, NumericVector y) {
  check_vecs(x, y);
  return tdoann::manhattan<double>(x.begin(), x.end(), y.begin());
}

//' Find the Hamming distance between two vectors
//'
//' @param x An integer vector.
//' @param y An integer vector of the same length as \code{x}.
//' @export
// [[Rcpp::export]]
double hamming_distance(IntegerVector x, IntegerVector y) {
  check_vecs(x, y);
  return tdoann::hamming<double>(x.begin(), x.end(), y.begin());
}

//' Find the correlation distance between two vectors
//'
//' @param x A numeric vector.
//' @param y A numeric vector of the same length as \code{x}.
//' @export
// [[Rcpp::export]]
double correlation_distance(NumericVector x, NumericVector y) {
  check_vecs(x, y);
  return tdoann::correlation<double>(x.begin(), x.end(), y.begin());
}

// NOLINTEND(modernize-use-trailing-return-type, readability-identifier-length)

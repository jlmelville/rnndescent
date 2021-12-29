#include "tdoann/distance.h"

#include <Rcpp.h>
using namespace Rcpp;

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

// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// nn_descent
Rcpp::List nn_descent(Rcpp::NumericMatrix data, Rcpp::IntegerMatrix idx, Rcpp::NumericMatrix dist, const std::string metric, const std::size_t max_candidates, const std::size_t n_iters, const double delta, const double rho, bool use_set, bool verbose);
RcppExport SEXP _rnndescent_nn_descent(SEXP dataSEXP, SEXP idxSEXP, SEXP distSEXP, SEXP metricSEXP, SEXP max_candidatesSEXP, SEXP n_itersSEXP, SEXP deltaSEXP, SEXP rhoSEXP, SEXP use_setSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type data(dataSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type idx(idxSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type dist(distSEXP);
    Rcpp::traits::input_parameter< const std::string >::type metric(metricSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type max_candidates(max_candidatesSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type n_iters(n_itersSEXP);
    Rcpp::traits::input_parameter< const double >::type delta(deltaSEXP);
    Rcpp::traits::input_parameter< const double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< bool >::type use_set(use_setSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(nn_descent(data, idx, dist, metric, max_candidates, n_iters, delta, rho, use_set, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_rnndescent_nn_descent", (DL_FUNC) &_rnndescent_nn_descent, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_rnndescent(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

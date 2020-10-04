#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
IntegerVector reverse_nbr_size_impl(IntegerMatrix nn_idx,
                                    std::size_t k,
                                    std::size_t len,
                                    bool include_self = false) {
  const std::size_t nr = nn_idx.nrow();
  const std::size_t nc = nn_idx.ncol();

  if (nc < k) {
    stop("Not enough columns in index matrix");
  }

  auto data = as<std::vector<std::size_t>>(nn_idx);

  std::vector<std::size_t> n_reverse(len);

  for (std::size_t i = 0; i < nr; i++) {
    for (std::size_t j = 0; j < k; j++) {
      std::size_t inbr = data[nr * j + i] - 1;
      if (inbr == i && !include_self) {
        continue;
      }
      ++n_reverse[inbr];
    }
  }
  return IntegerVector(n_reverse.begin(), n_reverse.end());
}


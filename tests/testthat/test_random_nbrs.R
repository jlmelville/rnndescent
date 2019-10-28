library(rnndescent)
context("Random neighbors")

set.seed(1337)
rnbrs <- random_nbrs(ui10, k = 4, use_cpp = FALSE)
check_nbrs_idx(rnbrs$idx)
check_nbrs_dist(rnbrs, ui10_eucd)

set.seed(1337)
rnbrs <- random_nbrs(ui10, k = 4, use_cpp = TRUE, n_threads = 0)
check_nbrs_idx(rnbrs$idx)
check_nbrs_dist(rnbrs, ui10_eucd, tol = 1e-6)

set.seed(1337)
rnbrs <- random_nbrs(ui10, k = 4, use_cpp = TRUE, n_threads = 1)
check_nbrs_idx(rnbrs$idx)
check_nbrs_dist(rnbrs, ui10_eucd, tol = 1e-6)

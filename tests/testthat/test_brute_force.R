library(rnndescent)
context("Brute force")

set.seed(1337)
rnbrs <- brute_force_knn(ui10, k = 4, n_threads = 0)
check_nbrs_idx(rnbrs$idx)
check_nbrs_dist(rnbrs, ui10_eucd, tol = 1e-6)

set.seed(1337)
rnbrs <- brute_force_knn(ui10, k = 4, n_threads = 1)
check_nbrs_idx(rnbrs$idx)
check_nbrs_dist(rnbrs, ui10_eucd, tol = 1e-6)

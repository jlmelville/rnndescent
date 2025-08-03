library(rnndescent)
context("RP Tree")

# just one tree
expected_rpt_knn <- list(
  idx = matrix(
    c(
      1, 6, 10, 0,
      2, 7, 4, 9,
      3, 5, 8, 0,
      4, 9, 2, 7,
      5, 8, 3, 0,
      6, 1, 10, 0,
      7, 2, 4, 9,
      8, 5, 3, 0,
      9, 4, 2, 7,
      10, 6, 1, 0
    ),
    nrow = 10,
    byrow = TRUE
  ),
  dist = matrix(
    c(
      0, 0.3464102, 0.6782330, NA,
      0, 0.3000002, 0.5477225, 0.6164413,
      0, 0.3316626, 0.4690416, NA,
      0, 0.3464101, 0.5477225, 0.6708205,
      0, 0.1732050, 0.3316626, NA,
      0, 0.3464102, 0.5830952, NA,
      0, 0.3000002, 0.6708205, 0.8185353,
      0, 0.1732050, 0.4690416, NA,
      0, 0.3464101, 0.6164413, 0.8185353,
      0, 0.5830952, 0.6782330, NA
    ),
    nrow = 10,
    byrow = TRUE
  )
)

set.seed(1337)
res <- rpf_knn(ui10, k = 4, leaf_size = 4, n_trees = 1)
expect_equal(res, expected_rpt_knn, tol = 1e-7)

set.seed(1337)
res <- rpf_knn(ui10, k = 4)
expect_equal(res$idx, ui10_nn4$idx, check.attributes = FALSE)
expect_equal(res$dist, ui10_nn4$dist, check.attributes = FALSE, tol = 1e-4)

set.seed(1337)
res <- rpf_knn(ui10, k = 4, include_self = FALSE)
expect_equal(res$idx[, 1:3], ui10_nn4$idx[, 2:4], check.attributes = FALSE)
expect_equal(res$dist[, 1:3], ui10_nn4$dist[, 2:4], check.attributes = FALSE, tol = 1e-4)

# euclidean
set.seed(1337)
res <- rpf_knn(ui10, k = 4)
expect_equal(res$idx, ui10_nn4$idx, check.attributes = FALSE)
expect_equal(res$dist, ui10_nn4$dist, check.attributes = FALSE, tol = 1e-4)

# cosine
set.seed(1337)
uiris_rnn <- rpf_knn(uirism, 15, metric = "cosine", n_trees = 40)
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)


# multi-threading
set.seed(1337)
res <- rpf_knn(ui10, k = 4, leaf_size = 3, n_threads = 2, n_trees = 1)
expect_in(c(NA), res$dist)
expect_in(c(0), res$idx)

set.seed(1337)
res <- rpf_knn(ui10, k = 4, n_threads = 2)
expect_equal(res$idx, ui10_nn4$idx, check.attributes = FALSE)
expect_equal(res$dist, ui10_nn4$dist, check.attributes = FALSE, tol = 1e-4)

# euclidean converges
set.seed(1337)
uiris_rnn <- rpf_knn(uiris, 15, n_trees = 40, n_threads = 2)
expect_equal(sum(uiris_rnn$dist), ui_edsum, tol = 1e-3)

# cosine
set.seed(1337)
uiris_rnn <- rpf_knn(uirism, 15, metric = "cosine", n_trees = 40, n_threads = 2)
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)

# R index
expected_rpf_index <- list(
  trees = list(
    list(
      hyperplanes = matrix(c(
        -0.5000000, -0.8000002, -0.2, -0.3,
        0.3000002, -0.3000002, 0.1, -0.2,
        0.0000000, 0.0000000, 0.0, 0.0,
        0.0000000, 0.0000000, 0.0, 0.0,
        0.0000000, 0.0000000, 0.0, 0.0
      ), nrow = 5, byrow = TRUE),
      offsets = c(5.7700009, -0.5550003, NA, NA, NA),
      children = matrix(c(
        1, 4,
        2, 3,
        0, 3,
        3, 7,
        7, 10
      ), nrow = 5, byrow = TRUE),
      indices = c(2, 4, 7, 1, 3, 6, 8, 0, 5, 9),
      leaf_size = 4
    )
  ),
  margin = "explicit",
  actual_metric = "sqeuclidean",
  version = "0.0.12",
  use_alt_metric = TRUE,
  original_metric = "euclidean",
  sparse = FALSE,
  type = "rnndescent:rpforest"
)

set.seed(1337)
rpf_index <- rpf_build(ui10, metric = "euclidean", n_trees = 1, leaf_size = 4)
expect_equal(rpf_index, expected_rpf_index, tol = 1e-7)

# query data against itself to reproduce knn (just more slowly)
set.seed(1337)
rpf_query_res <-
  rpf_knn_query(
    ui10,
    ui10,
    rpf_index,
    k = 4,
    n_threads = 0,
    cache = TRUE
  )
expect_equal(rpf_query_res, expected_rpt_knn, tol = 1e-7)

set.seed(1337)
rpf_query_res <-
  rpf_knn_query(
    ui10,
    ui10,
    rpf_index,
    k = 4,
    n_threads = 0,
    cache = FALSE
  )
expect_equal(rpf_query_res, expected_rpt_knn, tol = 1e-7)

# return forest with knn
set.seed(1337)
rpf_knnf <-
  rpf_knn(
    ui10,
    k = 4,
    metric = "euclidean",
    n_trees = 1,
    ret_forest = TRUE,
    leaf_size = 4
  )
expect_equal(rpf_knnf$forest, expected_rpf_index, tol = 1e-7)
expect_equal(
  list(idx = rpf_knnf$idx, dist = rpf_knnf$dist),
  expected_rpt_knn,
  check.attributes = FALSE,
  tol = 1e-7
)

set.seed(1337)
nnd_with_tree <-
  nnd_knn(
    ui10,
    k = 4,
    ret_forest = TRUE,
    init = "tree",
    init_args = list(n_trees = 1, leaf_size = 4)
  )
expect_equal(nnd_with_tree$forest, expected_rpf_index, tol = 1e-7)

# handle alt metric
set.seed(1337)
nnd_with_tree <-
  nnd_knn(
    ui10,
    k = 4,
    ret_forest = TRUE,
    init = "tree",
    use_alt_metric = FALSE,
    init_args = list(n_trees = 1, leaf_size = 4)
  )
rpf_index_no_alt <- expected_rpf_index
rpf_index_no_alt$use_alt_metric <- FALSE
rpf_index_no_alt$actual_metric <- "euclidean"
expect_equal(nnd_with_tree$forest, rpf_index_no_alt, tol = 1e-7)


# filtering
set.seed(1337)
rpf_knnf3 <-
  rpf_knn(
    ui10,
    k = 4,
    metric = "euclidean",
    n_trees = 3,
    ret_forest = TRUE,
    leaf_size = 4
  )
expect_equal(length(rpf_knnf3$forest$trees), 3)
rpf_f3f <- rpf_filter(rpf_knnf3, n_trees = 1)
expect_equal(length(rpf_f3f$trees), 1)
expect_equal(rpf_f3f$trees[[1]], rpf_knnf3$forest$trees[[1]])


test_that("can't pass mismatched forest and nn to filter", {
  iris_knn_with_forest <-
    rpf_knn(iris[1:50, ], k = 15, ret_forest = TRUE)
  iris_query_virginica <-
    rpf_knn_query(
      query = iris[51:150, ],
      reference = iris[1:50, ],
      forest = iris_knn_with_forest$forest,
      k = 15
    )

  iris_forest <- rpf_build(iris, leaf_size = 15)
  expect_error(
    rpf_filter(
      nn = iris_query_virginica,
      forest = iris_forest,
      n_trees = 1
    ), "Mismatched"
  )
})

set.seed(1337)
expect_equal(
  rpf_build(ui10, metric = "euclidean", leaf_size = 4, n_threads = 0),
  rpf_index_ls4e,
  tol = 1e-7
)

# implicit margin
set.seed(1337)
rpi_knn <- rpf_knn(uirism[1:20, ], k = 4, verbose = FALSE, n_threads = 0, n_trees = 2, margin = "implicit")
set.seed(1337)
rpe_knn <- rpf_knn(uirism[1:20, ], k = 4, verbose = FALSE, n_threads = 0, n_trees = 2, margin = "explicit")
expect_equal(rpe_knn$dist, rpi_knn$dist)

expected_rpfi_index <- list(
  trees = list(list(
    normal_indices = matrix(c(
      4, 0,
      4, 1,
      -1, -1,
      -1, -1,
      -1, -1
    ), nrow = 5, byrow = TRUE),
    children = matrix(c(
      1, 4,
      2, 3,
      0, 3,
      3, 7,
      7, 10
    ), nrow = 5, byrow = TRUE),
    indices = c(2, 4, 7, 1, 3, 6, 8, 0, 5, 9),
    leaf_size = 4
  )),
  margin = "implicit",
  actual_metric = "sqeuclidean",
  version = "0.0.12",
  use_alt_metric = TRUE,
  original_metric = "euclidean",
  sparse = FALSE,
  type = "rnndescent:rpforest"
)
set.seed(1337)
rpf_knn2df <- rpf_knn(
  ui10,
  k = 4,
  metric = "euclidean",
  n_trees = 1,
  ret_forest = TRUE,
  leaf_size = 4,
  margin = "implicit"
)
expect_equal(list(idx = rpf_knn2df$idx, dist = rpf_knn2df$dist), expected_rpt_knn, tol = 1e-7)
expect_equal(rpf_knn2df$forest, expected_rpfi_index)

set.seed(1337)
rpfi_query_res <-
  rpf_knn_query(
    ui10,
    ui10,
    rpf_knn2df$forest,
    k = 4,
    n_threads = 0,
    cache = TRUE
  )
expect_equal(rpfi_query_res, expected_rpt_knn, tol = 1e-7)


set.seed(1337)
rpf_knnfi3 <-
  rpf_knn(
    ui10,
    k = 4,
    metric = "euclidean",
    n_trees = 3,
    ret_forest = TRUE,
    leaf_size = 4,
    margin = "implicit"
  )
expect_equal(length(rpf_knnfi3$forest$trees), 3)
rpf_fi3f <- rpf_filter(rpf_knnfi3, n_trees = 1)
expect_equal(length(rpf_fi3f$trees), 1)
expect_equal(rpf_fi3f$trees[[1]], rpf_knnfi3$forest$trees[[1]])
expect_equal(rpf_fi3f$margin, rpf_knnfi3$forest$margin)
expect_equal(rpf_fi3f$actual_metric, rpf_knnfi3$forest$actual_metric)
expect_equal(rpf_fi3f$version, rpf_knnfi3$forest$version)
expect_equal(rpf_fi3f$use_alt_metric, rpf_knnfi3$forest$use_alt_metric)
expect_equal(rpf_fi3f$original_metric, rpf_knnfi3$forest$original_metric)


set.seed(1337)
rpf_knnff3 <-
  rpf_knn(
    ui10,
    k = 4,
    metric = "euclidean",
    n_trees = 3,
    ret_forest = TRUE,
    leaf_size = 4,
    margin = "explicit"
  )
expect_equal(length(rpf_knnff3$forest$trees), 3)
rpf_ff3f <- rpf_filter(rpf_knnff3, n_trees = 1)
expect_equal(length(rpf_ff3f$trees), 1)
expect_equal(rpf_ff3f$trees[[1]], rpf_knnff3$forest$trees[[1]])
expect_equal(rpf_ff3f$margin, rpf_knnff3$forest$margin)
expect_equal(rpf_ff3f$actual_metric, rpf_knnff3$forest$actual_metric)
expect_equal(rpf_ff3f$version, rpf_knnff3$forest$version)
expect_equal(rpf_ff3f$use_alt_metric, rpf_knnff3$forest$use_alt_metric)
expect_equal(rpf_ff3f$original_metric, rpf_knnff3$forest$original_metric)

set.seed(1337)
expect_equal(
  rpf_build(ui10, metric = "euclidean", leaf_size = 4, margin = "implicit", n_threads = 0),
  rpf_index_ls4i,
  tol = 1e-7
)

set.seed(1337)
rpf_index_ls4i_no_alt <- rpf_index_ls4i
rpf_index_ls4i_no_alt$use_alt_metric <- FALSE
rpf_index_ls4i_no_alt$actual_metric <- "euclidean"

expect_equal(
  rpf_build(ui10, metric = "euclidean", use_alt_metric = FALSE, leaf_size = 4, margin = "implicit", n_threads = 0),
  rpf_index_ls4i_no_alt,
  tol = 1e-7
)

# cosine test
set.seed(1337)
uiriscos <-
  rpf_knn(
    uirism,
    k = 15,
    metric = "cosine",
    n_threads = 0,
    ret_forest = TRUE,
    n_trees = 1
  )
set.seed(1337)
uiriscosq <-
  rpf_knn_query(
    uirism,
    uirism,
    uiriscos$forest,
    k = 15,
    n_threads = 0,
  )
# handle ties where indices swap places
expect_equal(sum(uiriscos$idx - uiriscosq$idx), 0)

# test uncached
uiriscosq_nocache <- rpf_knn_query(
  uirism,
  uirism,
  uiriscos$forest,
  k = 15,
  n_threads = 2,
  cache = FALSE
)
expect_equal(sum(uiriscos$idx - uiriscosq_nocache$idx), 0)

set.seed(1337)
uiriscosi <-
  rpf_knn(
    uirism,
    k = 15,
    metric = "cosine",
    n_threads = 0,
    ret_forest = TRUE,
    margin = "implicit",
    n_trees = 1
  )
set.seed(1337)
uiriscosiq <-
  rpf_knn_query(
    uirism,
    uirism,
    uiriscosi$forest,
    k = 15,
    n_threads = 0,
  )
expect_equal(sum(uiriscosi$idx - uiriscosq$idx), 0)
expect_equal(sum(uiriscosi$idx - uiriscosiq$idx), 0)

set.seed(1337)
ui6f <- rpf_knn(
  ui6,
  k = 4,
  leaf_size = 3,
  ret_forest = TRUE
)
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6f, query = ui4, init = ui6f$forest, k = 4)
expect_equal(sum(qnbrs4$dist), ui4q_edsum, tol = 1e-6)

test_that("binary data", {
  # euclidean forces conversion to float data
  set.seed(1337)
  bin_euc_imp <- rpf_knn(lbitdata, k = 4, margin = "implicit")
  set.seed(1337)
  bin_euc_exp <- rpf_knn(lbitdata, k = 4, margin = "explicit")
  expect_equal(bin_euc_imp, bin_euc_exp)

  set.seed(1337)
  bin_jac_imp <- rpf_knn(lbitdata, k = 4, margin = "implicit", metric = "jaccard")
  set.seed(1337)
  bin_jac_exp <- rpf_knn(lbitdata, k = 4, margin = "explicit", metric = "jaccard")
  expect_equal(bin_jac_imp, bin_jac_exp)
  set.seed(1337)
  bin_jac_aut <- rpf_knn(lbitdata, k = 4, margin = "auto", metric = "jaccard")
  expect_equal(bin_jac_aut, bin_jac_imp)

  set.seed(1337)
  euc_forest_i <- rpf_build(lbitdata, leaf_size = 10, margin = "implicit")
  bin_euc_impq <-
    rpf_knn_query(
      query = lbitdata,
      reference = lbitdata,
      forest = euc_forest_i,
      k = 4
    )
  expect_equal(bin_euc_impq, bin_euc_imp)

  set.seed(1337)
  euc_forest_e <- rpf_build(lbitdata, leaf_size = 10, margin = "explicit")
  bin_euc_expq <-
    rpf_knn_query(
      query = lbitdata,
      reference = lbitdata,
      forest = euc_forest_e,
      k = 4
    )
  expect_equal(bin_euc_expq, bin_euc_exp)

  set.seed(1337)
  euc_forest_a <- rpf_build(lbitdata, leaf_size = 10, margin = "auto")
  bin_euc_autq <-
    rpf_knn_query(
      query = lbitdata,
      reference = lbitdata,
      forest = euc_forest_a,
      k = 4
    )
  expect_equal(bin_euc_autq, bin_euc_exp)
})

test_that("sparse implicit margin", {
  set.seed(1337)
  dknn <- rpf_knn(ui10z, k = 4, leaf_size = 3, n_trees = 2, margin = "implicit")
  set.seed(1337)
  sknn <- rpf_knn(ui10sp, k = 4, leaf_size = 3, n_trees = 2, margin = "implicit")
  expect_equal(sknn, dknn, tol = 1e-6)

  set.seed(1337)
  dknn <- rpf_knn(ui10z, k = 4, leaf_size = 3, n_trees = 2, margin = "implicit", ret_forest = TRUE)
  set.seed(1337)
  sknn <- rpf_knn(ui10sp, k = 4, leaf_size = 3, n_trees = 2, margin = "implicit", ret_forest = TRUE)
  expect_equal(list(idx = sknn$idx, dist = sknn$dist), list(idx = dknn$idx, dist = dknn$dist), tol = 1e-6)
  dknn$forest$sparse <- TRUE
  expect_equal(sknn$forest, dknn$forest)

  set.seed(1337)
  dforest <- rpf_build(ui10z, leaf_size = 3, n_trees = 2, margin = "implicit", metric = "cosine")
  set.seed(1337)
  sforest <- rpf_build(ui10sp, leaf_size = 3, n_trees = 2, margin = "implicit", metric = "cosine")
  expect_equal(sforest$actual_metric, "alternative-cosine")
  expect_true(sforest$sparse)
  sforest$sparse <- FALSE
  expect_equal(sforest, dforest, tol = 1e-6)

  set.seed(1337)
  dforest6 <- rpf_build(ui10z6, leaf_size = 3, n_trees = 2, margin = "implicit", metric = "cosine")
  set.seed(1337)
  dquery4 <- rpf_knn_query(query = ui10z4, reference = ui10z6, forest = dforest6, k = 4)
  expect_error(squery4 <- rpf_knn_query(query = ui10sp4, reference = ui10sp6, forest = dforest6, k = 4), "sparse forest")
  # hack the forest to force it to work with sparse
  dforest6$sparse <- TRUE
  set.seed(1337)
  squery4 <- rpf_knn_query(query = ui10sp4, reference = ui10sp6, forest = dforest6, k = 4)
  expect_equal(squery4, dquery4, tol = 1e-4)

  set.seed(1337)
  sforest6 <- rpf_build(ui10sp6, leaf_size = 3, n_trees = 2, margin = "implicit", metric = "cosine")
  set.seed(1337)
  squery4b <- rpf_knn_query(query = ui10sp4, reference = ui10sp6, forest = sforest6, k = 4)
  expect_equal(squery4b, squery4, tol = 1e-5)

  set.seed(1337)
  squery4b <- rpf_knn_query(query = ui10sp4, reference = ui10sp6, forest = sforest6, k = 4, cache = FALSE)
  expect_equal(squery4b, squery4, tol = 1e-5)
})


test_that("sparse explicit margin", {
  set.seed(1337)
  dknn <- rpf_knn(ui10z, k = 4, leaf_size = 3, n_trees = 2, margin = "explicit")
  set.seed(1337)
  sknn <- rpf_knn(ui10sp, k = 4, leaf_size = 3, n_trees = 2, margin = "explicit")
  expect_equal(sknn, dknn)

  # implict and explicit should give the same results for euclidean
  set.seed(1337)
  siknn <- rpf_knn(ui10sp, k = 4, leaf_size = 3, n_trees = 2, margin = "implicit")
  expect_equal(siknn, sknn)

  set.seed(1337)
  sknn6 <- rpf_knn(ui10sp6, k = 4, leaf_size = 2, n_trees = 2, ret_forest = TRUE)
  set.seed(1337)
  sforest6 <- rpf_build(ui10sp6, leaf_size = 2, n_trees = 2)
  expect_equal(sforest6$margin, "explicit")
  expect_equal(sknn6$forest$margin, "explicit")
  expect_equal(sforest6, sknn6$forest)

  s6_ff <- rpf_filter(sknn6)
  expect_equal(length(s6_ff$trees), 1)
  expect_equal(s6_ff$trees[[1]], sknn6$forest$trees[[1]])

  set.seed(1337)
  res_forest <- rpf_knn_query(ui10sp4, ui10sp6, forest = sforest6, k = 4)
  set.seed(1337)
  res_knnforest <- rpf_knn_query(ui10sp4, ui10sp6, forest = sknn6$forest, k = 4)
  expect_equal(res_forest, res_knnforest)

  set.seed(1337)
  dknn6 <- rpf_knn(ui10z6, k = 4, leaf_size = 2, n_trees = 2, margin = "explicit", ret_forest = TRUE)
  set.seed(1337)
  res_dknn <- rpf_knn_query(ui10z4, ui10z6, forest = dknn6$forest, k = 4)
  expect_equal(res_forest, res_dknn)

  # uncached
  set.seed(1337)
  res_dknn_nocache <- rpf_knn_query(ui10z4, ui10z6, forest = dknn6$forest, k = 4, cache = FALSE)
  expect_equal(res_forest, res_dknn_nocache)

  # implict and explicit should give the same results for cosine also
  set.seed(1337)
  secknn <- rpf_knn(ui10sp, k = 4, leaf_size = 3, n_trees = 2, margin = "explicit", metric = "cosine")
  set.seed(1337)
  sicknn <- rpf_knn(ui10sp, k = 4, leaf_size = 3, n_trees = 2, margin = "implicit", metric = "cosine")
  expect_equal(secknn, sicknn)

  set.seed(1337)
  sacknn <- rpf_knn(ui10sp,
    k = 4, leaf_size = 3, n_trees = 2,
    margin = "explicit", metric = "cosine", use_alt_metric = FALSE
  )
  expect_equal(sacknn, secknn, tol = 1e-4)
})

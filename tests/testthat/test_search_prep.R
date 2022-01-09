library(rnndescent)
context("Search graph preparation")

ui10_bf <- brute_force_knn(ui10, k = 4)

test_that("single thread prepare", {
  sg_full <-
    prepare_search_graph(
      data = ui10,
      graph = ui10_bf,
      diversify_prob = NULL,
      pruning_degree_multiplier = NULL
    )

  expect_s4_class(sg_full, "sparseMatrix")
  expect_equal(
    sg_full@x,
    c(
      0.7, 0.3464, 0.6782, 0.4243, 0.5477, 0.4796, 0.3, 0.6164, 0.7, 0.4243,
      0.3317, 0.5, 0.2236, 1.044, 0.5477, 0.4243, 0.3464, 0.4796, 0.3317,
      0.3464, 0.1732, 0.3464, 0.5, 0.5831, 0.3, 0.2236, 0.3464, 0.4583, 0.4243,
      0.1732, 0.4583, 0.5831, 0.6164, 0.3464, 0.5831, 0.6782, 1.044, 0.5831
    ),
    tolerance = 1e-4
  )
  expect_equal(
    sg_full@i,
    c(
      2, 5, 9, 2, 3, 4, 6, 8, 0, 1, 4, 5, 6, 9, 1, 7, 8, 1, 2, 6, 7, 0, 2, 9, 1,
      2, 4, 7, 3, 4, 6, 8, 1, 3, 7, 0, 2, 5
    )
  )
  expect_equal(sg_full@p, c(0, 3, 8, 14, 17, 21, 24, 28, 32, 35, 38))

  sg_occ <-
    prepare_search_graph(
      data = ui10,
      graph = ui10_bf,
      diversify_prob = 1,
      pruning_degree_multiplier = NULL
    )
  expect_equal(
    sg_occ@x,
    c(
      0.3464, 0.3, 0.3317, 0.5, 0.2236, 0.4243, 0.3464, 0.3317, 0.1732, 0.3464,
      0.5, 0.5831, 0.3, 0.2236, 0.4243, 0.1732, 0.3464, 0.5831
    ),
    tolerance = 1e-4
  )
  expect_equal(sg_occ@i, c(5, 6, 4, 5, 6, 7, 8, 2, 7, 0, 2, 9, 1, 2, 3, 4, 3, 5))
  expect_equal(sg_occ@p, c(0, 1, 2, 5, 7, 9, 12, 14, 16, 17, 18))

  set.seed(1337)
  sg_occp <-
    prepare_search_graph(
      data = ui10,
      graph = ui10_bf,
      diversify_prob = 0.5,
      pruning_degree_multiplier = NULL
    )
  expect_equal(
    sg_occp@x,
    c(
      0.7, 0.3464, 0.6782, 0.4243, 0.5477, 0.3, 0.7, 0.3317, 0.5, 0.2236, 1.044,
      0.5477, 0.4243, 0.3464, 0.3317, 0.1732, 0.3464, 0.5, 0.5831, 0.3, 0.2236,
      0.4583, 0.4243, 0.1732, 0.4583, 0.3464, 0.6782, 0.5831
    ),
    tolerance = 1e-4
  )
  expect_equal(sg_occp@i, c(
    2, 5, 9, 2, 3, 6, 0, 4, 5, 6, 9, 1, 7, 8, 2, 7, 0, 2, 9, 1, 2, 7, 3, 4, 6,
    3, 0, 5
  ))
  expect_equal(sg_occp@p, c(0, 3, 6, 11, 14, 16, 19, 22, 25, 26, 28))


  sg_trunc <-
    prepare_search_graph(
      data = ui10,
      graph = ui10_bf,
      diversify_prob = NULL,
      pruning_degree_multiplier = 1.5
    )
  expect_equal(
    sg_trunc@x,
    c(
      0.7, 0.3464, 0.6782, 0.4243, 0.5477, 0.4796, 0.3, 0.6164, 0.7, 0.4243,
      0.3317, 0.5, 0.2236, 1.044, 0.5477, 0.4243, 0.3464, 0.4796, 0.3317, 0.3464,
      0.1732, 0.3464, 0.5, 0.5831, 0.3, 0.2236, 0.3464, 0.4583, 0.4243, 0.1732,
      0.4583, 0.5831, 0.6164, 0.3464, 0.5831, 0.6782, 1.044, 0.5831
    ),
    tolerance = 1e-4
  )
  expect_equal(
    sg_trunc@i,
    c(
      2, 5, 9, 2, 3, 4, 6, 8, 0, 1, 4, 5, 6, 9, 1, 7, 8, 1, 2, 6, 7, 0, 2, 9, 1,
      2, 4, 7, 3, 4, 6, 8, 1, 3, 7, 0, 2, 5
    )
  )
  expect_equal(sg_trunc@p, c(0, 3, 8, 14, 17, 21, 24, 28, 32, 35, 38))

  sg_occ_trunc <-
    prepare_search_graph(
      data = ui10,
      graph = ui10_bf,
      diversify_prob = 1,
      pruning_degree_multiplier = 0.5
    )
  expect_equal(
    sg_occ_trunc@x,
    c(
      0.3464, 0.3, 0.3317, 0.5, 0.2236, 0.4243, 0.3464, 0.3317, 0.1732, 0.3464,
      0.5831, 0.3, 0.2236, 0.4243, 0.1732, 0.3464
    ),
    tolerance = 1e-4
  )
  expect_equal(
    sg_occ_trunc@i,
    c(
      5, 6, 4, 5, 6, 7, 8, 2, 7, 0, 9, 1, 2, 3, 4, 3
    )
  )
  expect_equal(sg_occ_trunc@p, c(0, 1, 2, 5, 7, 9, 11, 13, 15, 16, 16))
})

test_that("parallel prepare", {
  sg_full <-
    prepare_search_graph(
      data = ui10,
      graph = ui10_bf,
      diversify_prob = NULL,
      pruning_degree_multiplier = NULL,
      n_threads = 1
    )

  expect_s4_class(sg_full, "sparseMatrix")
  expect_equal(
    sg_full@x,
    c(
      0.7, 0.3464, 0.6782, 0.4243, 0.5477, 0.4796, 0.3, 0.6164, 0.7, 0.4243,
      0.3317, 0.5, 0.2236, 1.044, 0.5477, 0.4243, 0.3464, 0.4796, 0.3317,
      0.3464, 0.1732, 0.3464, 0.5, 0.5831, 0.3, 0.2236, 0.3464, 0.4583, 0.4243,
      0.1732, 0.4583, 0.5831, 0.6164, 0.3464, 0.5831, 0.6782, 1.044, 0.5831
    ),
    tolerance = 1e-4
  )
  expect_equal(
    sg_full@i,
    c(
      2, 5, 9, 2, 3, 4, 6, 8, 0, 1, 4, 5, 6, 9, 1, 7, 8, 1, 2, 6, 7, 0, 2, 9, 1,
      2, 4, 7, 3, 4, 6, 8, 1, 3, 7, 0, 2, 5
    )
  )
  expect_equal(sg_full@p, c(0, 3, 8, 14, 17, 21, 24, 28, 32, 35, 38))

  sg_occ <-
    prepare_search_graph(
      data = ui10,
      graph = ui10_bf,
      diversify_prob = 1,
      pruning_degree_multiplier = NULL,
      n_threads = 1
    )
  expect_equal(
    sg_occ@x,
    c(
      0.3464, 0.3, 0.3317, 0.5, 0.2236, 0.4243, 0.3464, 0.3317, 0.1732, 0.3464,
      0.5, 0.5831, 0.3, 0.2236, 0.4243, 0.1732, 0.3464, 0.5831
    ),
    tolerance = 1e-4
  )
  expect_equal(sg_occ@i, c(5, 6, 4, 5, 6, 7, 8, 2, 7, 0, 2, 9, 1, 2, 3, 4, 3, 5))
  expect_equal(sg_occ@p, c(0, 1, 2, 5, 7, 9, 12, 14, 16, 17, 18))

  set.seed(1337)
  sg_occp <-
    prepare_search_graph(
      data = ui10,
      graph = ui10_bf,
      diversify_prob = 0.5,
      pruning_degree_multiplier = NULL,
      n_threads = 1
    )
  # check we have more edges kept than when diversify_prob = 1
  expect_gt(length(sg_occp@x), length(sg_occ@x))
  # and that the returned edges are a subset of the undiversified graph
  # floating point values so convert to strings
  expect_true(all(formatC(sg_occ@x) %in% formatC(sg_occp@x)))
  expect_true(all(formatC(sg_occp@x) %in% formatC(sg_full@x)))


  sg_trunc <-
    prepare_search_graph(
      data = ui10,
      graph = ui10_bf,
      diversify_prob = NULL,
      pruning_degree_multiplier = 1.5,
      n_threads = 1
    )
  expect_equal(
    sg_trunc@x,
    c(
      0.7, 0.3464, 0.6782, 0.4243, 0.5477, 0.4796, 0.3, 0.6164, 0.7, 0.4243,
      0.3317, 0.5, 0.2236, 1.044, 0.5477, 0.4243, 0.3464, 0.4796, 0.3317, 0.3464,
      0.1732, 0.3464, 0.5, 0.5831, 0.3, 0.2236, 0.3464, 0.4583, 0.4243, 0.1732,
      0.4583, 0.5831, 0.6164, 0.3464, 0.5831, 0.6782, 1.044, 0.5831
    ),
    tolerance = 1e-4
  )
  expect_equal(
    sg_trunc@i,
    c(
      2, 5, 9, 2, 3, 4, 6, 8, 0, 1, 4, 5, 6, 9, 1, 7, 8, 1, 2, 6, 7, 0, 2, 9, 1,
      2, 4, 7, 3, 4, 6, 8, 1, 3, 7, 0, 2, 5
    )
  )
  expect_equal(sg_trunc@p, c(0, 3, 8, 14, 17, 21, 24, 28, 32, 35, 38))

  sg_occ_trunc <-
    prepare_search_graph(
      data = ui10,
      graph = ui10_bf,
      diversify_prob = 1,
      pruning_degree_multiplier = 0.5,
      n_threads = 1
    )
  expect_equal(
    sg_occ_trunc@x,
    c(
      0.3464, 0.3, 0.3317, 0.5, 0.2236, 0.4243, 0.3464, 0.3317, 0.1732, 0.3464,
      0.5831, 0.3, 0.2236, 0.4243, 0.1732, 0.3464
    ),
    tolerance = 1e-4
  )
  expect_equal(
    sg_occ_trunc@i,
    c(
      5, 6, 4, 5, 6, 7, 8, 2, 7, 0, 9, 1, 2, 3, 4, 3
    )
  )
  expect_equal(sg_occ_trunc@p, c(0, 1, 2, 5, 7, 9, 11, 13, 15, 16, 16))
})

test_that("explicit zeros are preserved", {
  ui10_bf0 <- list(idx = ui10_bf$idx, dist = ui10_bf$dist)
  ui10_bf0$dist[10, 4] <- 0
  sg_0 <-
    prepare_search_graph(
      data = ui10,
      graph = ui10_bf0
    )
  expect_true(sg_0[10, 3] > 0)
})

test_that("column orientation", {
  sg_occ_trunc <-
    prepare_search_graph(
      data = t(ui10),
      graph = ui10_bf,
      diversify_prob = 1,
      pruning_degree_multiplier = 0.5,
      n_threads = 2,
      obs = "C"
    )
  expect_equal(
    sg_occ_trunc@x,
    c(
      0.3464, 0.3, 0.3317, 0.5, 0.2236, 0.4243, 0.3464, 0.3317, 0.1732, 0.3464,
      0.5831, 0.3, 0.2236, 0.4243, 0.1732, 0.3464
    ),
    tolerance = 1e-4
  )
  expect_equal(
    sg_occ_trunc@i,
    c(
      5, 6, 4, 5, 6, 7, 8, 2, 7, 0, 9, 1, 2, 3, 4, 3
    )
  )
  expect_equal(sg_occ_trunc@p, c(0, 1, 2, 5, 7, 9, 11, 13, 15, 16, 16))
})

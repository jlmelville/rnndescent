library(rnndescent)
context("Corrections")

# 14: bad uncorrection broke cosine, jaccard, hellinger default search

float32_max <- 3.4028235e+38

test_that("cosine", {
  expect_equal(correct_alternative_cosine(0), 0)
  expect_equal(correct_alternative_cosine(1), 0.5)
  expect_equal(correct_alternative_cosine(float32_max), 1.0)

  expect_equal(uncorrect_alternative_cosine(0), 0)
  expect_equal(uncorrect_alternative_cosine(0.5), 1.0)
  expect_equal(uncorrect_alternative_cosine(1), float32_max)
})

test_that("jaccard", {
  expect_equal(correct_alternative_jaccard(0), 0)
  expect_equal(correct_alternative_jaccard(1), 0.5)
  expect_equal(correct_alternative_jaccard(float32_max), 1.0)

  expect_equal(uncorrect_alternative_jaccard(0), 0)
  expect_equal(uncorrect_alternative_jaccard(0.5), 1.0)
  expect_equal(uncorrect_alternative_jaccard(1), float32_max)
})

test_that("hellinger", {
  expect_equal(correct_alternative_hellinger(0), 0)
  expect_equal(correct_alternative_hellinger(1), 1 / sqrt(2))
  expect_equal(correct_alternative_hellinger(float32_max), 1.0)

  expect_equal(uncorrect_alternative_hellinger(0), 0)
  expect_equal(uncorrect_alternative_hellinger(1 / sqrt(2)), 1.0)
  expect_equal(uncorrect_alternative_hellinger(1), float32_max)
})

test_that("dot", {
  expect_equal(correct_alternative_dot(0), 0)
  expect_equal(correct_alternative_dot(1), 0.5)
  expect_equal(correct_alternative_dot(float32_max), 1.0)
})

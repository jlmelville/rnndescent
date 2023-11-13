find_sparse_alt_metric <- function(metric) {
  switch(metric,
         euclidean = "sqeuclidean",
         cosine = "alternative-cosine",
         dot = "alternative-dot",
         hellinger = "alternative-hellinger",
         jaccard = "alternative-jaccard",
         trueangular = "alternative-cosine",
         metric
  )
}

find_dense_alt_metric <- function(metric) {
  switch(metric,
         euclidean = "sqeuclidean",
         cosine = "alternative-cosine",
         dot = "alternative-dot",
         hellinger = "alternative-hellinger",
         jaccard = "alternative-jaccard",
         trueangular = "alternative-cosine",
         metric
  )
}

find_alt_metric <- function(metric, is_sparse = FALSE) {
  if (is_sparse) {
    find_sparse_alt_metric(metric)
  }
  else {
    find_dense_alt_metric(metric)
  }
}

# needed for any method which can take a pre-calculate `init` parameter *and*
# also `use_alt_metric = TRUE`, e.g. if we are actually going to be working on
# squared Euclidean distances, we need to transform initial Euclidean distances
# accordingly
apply_alt_metric_uncorrection <- function(metric, dist, is_sparse = FALSE) {
  if (is_sparse) {
    apply_sparse_alt_metric_uncorrection(metric, dist)
  }
  else {
    apply_dense_alt_metric_uncorrection(metric, dist)
  }
}

apply_dense_alt_metric_uncorrection <- function(metric, dist) {
  switch(metric,
         cosine = apply(dist, c(1, 2), uncorrect_alternative_cosine),
         dot = apply(dist, c(1, 2), uncorrect_alternative_cosine),
         euclidean = dist * dist,
         hellinger = apply(dist, c(1, 2), uncorrect_alternative_hellinger),
         jaccard = apply(dist, c(1, 2), uncorrect_alternative_jaccard),
         trueangular = apply(dist, c(1, 2), uncorrect_true_angular),
         dist
  )
}

apply_sparse_alt_metric_uncorrection <- function(metric, dist) {
  switch(
    metric,
    cosine = apply(dist, c(1, 2), uncorrect_alternative_cosine),
    euclidean = dist * dist,
    dot = apply(dist, c(1, 2), uncorrect_alternative_cosine),
    hellinger = apply(dist, c(1, 2), uncorrect_alternative_hellinger),
    jaccard = apply(dist, c(1, 2), uncorrect_alternative_jaccard),
    trueangular = apply(dist, c(1, 2), uncorrect_true_angular),
    dist
  )
}

apply_dense_alt_metric_correction <- function(metric, dist) {
  switch(metric,
         cosine = apply(dist, c(1, 2), correct_alternative_cosine),
         dot = apply(dist, c(1, 2), correct_alternative_dot),
         euclidean = sqrt(dist),
         hellinger = apply(dist, c(1, 2), correct_alternative_hellinger),
         jaccard = apply(dist, c(1, 2), correct_alternative_jaccard),
         trueangular = apply(dist, c(1, 2), true_angular_from_alt_cosine),
         dist
  )
}

apply_sparse_alt_metric_correction <- function(metric, dist) {
  switch(metric,
         cosine = apply(dist, c(1, 2), correct_alternative_cosine),
         dot = apply(dist, c(1, 2), correct_alternative_dot),
         euclidean = sqrt(dist),
         hellinger = apply(dist, c(1, 2), correct_alternative_hellinger),
         jaccard = apply(dist, c(1, 2), correct_alternative_jaccard),
         trueangular = apply(dist, c(1, 2), true_angular_from_alt_cosine),
         dist
  )
}

apply_alt_metric_correction <- function(metric, dist, is_sparse = FALSE) {
  if (is_sparse) {
    apply_sparse_alt_metric_correction(metric, dist)
  }
  else {
    apply_dense_alt_metric_correction(metric, dist)
  }
}

get_actual_metric <- function(use_alt_metric, metric, data, verbose) {
  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric, is_sparse(data))
    if (actual_metric != metric) {
      tsmessage("Using alt metric '", actual_metric, "' for '", metric, "'")
    }
  } else {
    actual_metric <- metric
  }
  actual_metric
}

isclose <- function(a, b, rtol = 1.0e-5, atol = 1.0e-8) {
  diff <- abs(a - b)
  diff <= (atol + rtol * abs(b))
}

correct_alternative_cosine <- function(dist) {
  # -ve distance is fine for dot, but not cosine
  max(correct_alternative_dot(dist), 0.0)
}

correct_alternative_dot <- function(dist) {
  if (is.na(dist)) {
    return(NA)
  }
  # -ve distance is ok for dot
  1.0 - (2.0 ^ -dist)
}

correct_alternative_jaccard <- function(dist) {
  if (is.na(dist)) {
    return(NA)
  }
  if (isclose(0.0, abs(dist), atol = 1e-7) || dist < 0.0) {
    0.0
  }
  else {
    1.0 - (2.0 ^ -dist)
  }
}

correct_alternative_hellinger <- function(dist) {
  sqrt(correct_alternative_jaccard(dist))
}

true_angular_from_alt_cosine <- function(dist) {
  if (is.na(dist)) {
    return(NA)
  }
  res <- 2 ^ -dist
  res <- max(min(res, 1.0), -1.0)
  1.0 - (acos(res) / pi)
}

uncorrect_true_angular <- function(dist) {
  if (is.na(dist)) {
    return(NA)
  }
  res <- max(min(1 - dist, 0.5), -0.5)
  -log2(cos(pi * res))
}

uncorrect_alternative_jaccard <- function(dist) {
  ifelse(dist >= (1.0 - 1.e-10), 0.0, -log2(1.0 - dist))
}

uncorrect_alternative_hellinger <- function(dist) {
  ifelse(dist >= (1.0 - 1.e-10), 0.0, -log2(1.0 - (dist * dist)))
}

uncorrect_alternative_cosine <- function(dist) {
  ifelse(dist >= (1.0 - 1.e-10), 0.0, -log2(1.0 - dist))
}

# Cosine and trueangular retain negative similarities with plain calculations.
find_sparse_alt_metric <- function(metric) {
  switch(
    metric,
    euclidean = "sqeuclidean",
    dot = "alternative-dot",
    hellinger = "alternative-hellinger",
    jaccard = "alternative-jaccard",
    metric
  )
}

find_dense_alt_metric <- function(metric) {
  switch(
    metric,
    euclidean = "sqeuclidean",
    dot = "alternative-dot",
    hellinger = "alternative-hellinger",
    jaccard = "alternative-jaccard",
    metric
  )
}

find_alt_metric <- function(metric, is_sparse = FALSE) {
  if (is_sparse) {
    find_sparse_alt_metric(metric)
  } else {
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
  } else {
    apply_dense_alt_metric_uncorrection(metric, dist)
  }
}

apply_dense_alt_metric_uncorrection <- function(metric, dist) {
  if (length(dist) == 0L && metric %in% c("dot", "hellinger", "jaccard")) {
    # Keep apply()/sapply() simplification for empty matrices and vectors.
    return(if (is.matrix(dist)) dist + 0.0 else as.list(dist))
  }
  switch(
    metric,
    dot = uncorrect_alternative_cosine(dist),
    euclidean = dist * dist,
    hellinger = uncorrect_alternative_hellinger(dist),
    jaccard = uncorrect_alternative_jaccard(dist),
    dist
  )
}

apply_sparse_alt_metric_uncorrection <- function(metric, dist) {
  apply_dense_alt_metric_uncorrection(metric, dist)
}

apply_dense_alt_metric_correction <- function(metric, dist) {
  if (length(dist) == 0L && metric %in% c("dot", "hellinger", "jaccard")) {
    return(if (is.matrix(dist)) dist + 0.0 else as.list(dist))
  }
  switch(
    metric,
    dot = correct_alternative_dot(dist),
    euclidean = sqrt(dist),
    hellinger = correct_alternative_hellinger(dist),
    jaccard = correct_alternative_jaccard(dist),
    dist
  )
}

apply_sparse_alt_metric_correction <- function(metric, dist) {
  apply_dense_alt_metric_correction(metric, dist)
}

apply_alt_metric_correction <- function(metric, dist, is_sparse = FALSE) {
  if (is_sparse) {
    apply_sparse_alt_metric_correction(metric, dist)
  } else {
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

correct_alternative_dot <- function(dist) {
  missing <- is.na(dist)
  if (all(missing)) {
    # Scalar missing values simplified to logical NA in apply()/sapply().
    missing[] <- NA
    return(missing)
  }
  # -ve distance is ok for dot
  result <- 1.0 - (2.0^-dist)
  result[missing] <- NA
  result
}

correct_alternative_jaccard <- function(dist) {
  result <- correct_alternative_dot(dist)
  clamped <- !is.na(dist) & (isclose(0.0, abs(dist), atol = 1e-7) | dist < 0.0)
  if (any(clamped)) {
    result[clamped] <- 0.0
  }
  result
}

correct_alternative_hellinger <- function(dist) {
  sqrt(correct_alternative_jaccard(dist))
}

float32_max <- 3.4028235e+38

uncorrect_alternative_jaccard <- function(dist) {
  saturated <- dist >= (1.0 - 1.e-10)
  result <- ifelse(saturated, float32_max, NA_real_)
  unsaturated <- which(!saturated)
  if (length(unsaturated)) {
    result[unsaturated] <- -log2(1.0 - dist[unsaturated])
  }
  result
}

uncorrect_alternative_hellinger <- function(dist) {
  saturated <- dist >= (1.0 - 1.e-10)
  result <- ifelse(saturated, float32_max, NA_real_)
  unsaturated <- which(!saturated)
  if (length(unsaturated)) {
    values <- dist[unsaturated]
    result[unsaturated] <- -log2(1.0 - (values * values))
  }
  result
}

uncorrect_alternative_cosine <- function(dist) {
  uncorrect_alternative_jaccard(dist)
}

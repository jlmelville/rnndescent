stime <- function() {
  format(Sys.time(), "%T")
}

# message with a time stamp
# appears only if called from an environment where a logical verbose = TRUE
# OR force = TRUE
tsmessage <-
  function(...,
           domain = NULL,
           appendLF = TRUE,
           force = FALSE,
           time_stamp = TRUE) {
    verbose <- get0("verbose", envir = sys.parent())

    if (force || (!is.null(verbose) && verbose)) {
      msg <- ""
      if (time_stamp) {
        msg <- paste0(stime(), " ")
      }
      message(msg, ..., domain = domain, appendLF = appendLF)
      utils::flush.console()
    }
  }

# convert data frame to matrix using numeric columns
x2m <- function(X) {
  if (is_sparse(X)) {
    return(X)
  }
  if (!methods::is(X, "matrix")) {
    m <- as.matrix(X[, which(vapply(X, is.numeric, logical(1)))])
  } else {
    m <- X
  }
  m
}

set_thread_options <- function(n_threads) {
  Sys.setenv(RCPP_PERPENDICULAR_NUM_THREADS = n_threads)
}

thread_msg <- function(..., n_threads) {
  msg <- paste0(...)
  if (n_threads > 0) {
    msg <- paste0(msg, " using ", n_threads, " threads")
  }
  msg
}

# Add the (named) values in l2 to l1.
# Use to override default values in l1 with user-supplied values in l2
lmerge <- function(l1, l2) {
  for (name in names(l2)) {
    l1[[name]] <- l2[[name]]
  }
  l1
}

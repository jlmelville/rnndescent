# Prepared handles must be invalidated before their DLL is unloaded.
data <- matrix(as.double(1:40), ncol = 2)
index <- rnndescent::rnnd_build(data, k = 4L, n_iters = 0L)
sparse <- Matrix::Matrix(data, sparse = TRUE)
sparse_index <- rnndescent::rnnd_build(sparse, k = 4L, n_iters = 0L)
transient <- rnndescent::rnnd_prepare(index)
prepared <- rnndescent::rnnd_prepare(index)
prepared_sparse <- rnndescent::rnnd_prepare(sparse_index)
rm(transient)
invisible(gc())
unloadNamespace("rnndescent")
stopifnot(!"rnndescent" %in% names(getLoadedDLLs()))
invisible(gc())
error <- tryCatch(
  rnndescent::rnnd_query(prepared, data, k = 4L),
  error = identity
)
stopifnot(
  inherits(error, "error"),
  grepl("Prepared index is invalid", conditionMessage(error))
)
error <- tryCatch(
  rnndescent::rnnd_query(prepared_sparse, sparse, k = 4L),
  error = identity
)
stopifnot(
  inherits(error, "error"),
  grepl("Prepared index is invalid", conditionMessage(error))
)
rebuilt <- rnndescent::rnnd_prepare(index)
stopifnot(nrow(rnndescent::rnnd_query(rebuilt, data, k = 4L)$idx) == 20L)
rebuilt_sparse <- rnndescent::rnnd_prepare(sparse_index)
stopifnot(
  nrow(rnndescent::rnnd_query(rebuilt_sparse, sparse, k = 4L)$idx) == 20L
)

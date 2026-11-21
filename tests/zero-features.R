# A character-only data frame previously terminated the R process.
library(rnndescent)

error <- tryCatch(
  brute_force_knn(data.frame(label = letters[1:5]), k = 2),
  error = identity
)
stopifnot(
  inherits(error, "error"),
  grepl("data must contain at least one feature", conditionMessage(error))
)

valid <- brute_force_knn(matrix(1:10, nrow = 5), k = 2)
stopifnot(identical(dim(valid$idx), c(5L, 2L)))

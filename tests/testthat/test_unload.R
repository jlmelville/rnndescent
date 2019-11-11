context("Unloading")

dyn_lib_names <- function() {
  sapply(.dynLibs(), `[[`, "name")
}

expect_true("rnndescent" %in% dyn_lib_names())
unloadNamespace("rnndescent")
expect_false("rnndescent" %in% dyn_lib_names())

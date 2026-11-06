build_sites <- function() {
  root <- normalizePath(".")
  destination <- file.path(root, "docs")
  scratch <- tempfile("pkgdown-release-")
  dir.create(scratch)
  on.exit(unlink(scratch, recursive = TRUE), add = TRUE)

  packages <- available.packages(
    repos = c(CRAN = "https://cloud.r-project.org"),
    type = "source"
  )
  version <- packages["rnndescent", "Version"]
  archive <- file.path(scratch, "release.tar.gz")
  tag <- paste0("v", version)
  download.file(
    paste0(
      "https://github.com/jlmelville/rnndescent/archive/refs/tags/",
      tag,
      ".tar.gz"
    ),
    archive,
    mode = "wb"
  )
  untar(archive, exdir = scratch)
  release <- file.path(scratch, paste0("rnndescent-", version))
  stopifnot(
    read.dcf(file.path(release, "DESCRIPTION"))[1, "Version"] == version
  )

  config <- yaml::read_yaml(file.path(root, "_pkgdown.yml"))
  topics <- list.files(file.path(release, "man"), pattern = "[.]Rd$")
  internal <- vapply(
    topics,
    function(topic) {
      any(grepl(
        "\\keyword{internal}",
        readLines(file.path(release, "man", topic)),
        fixed = TRUE
      ))
    },
    logical(1)
  )
  topics <- sub("[.]Rd$", "", topics[!internal])
  for (i in seq_along(config$reference)) {
    if (!is.null(config$reference[[i]]$contents)) {
      config$reference[[i]]$contents <- intersect(
        config$reference[[i]]$contents,
        topics
      )
    }
  }
  extra <- setdiff(topics, unlist(lapply(config$reference, `[[`, "contents")))
  if (length(extra) > 0L) {
    config$reference <- c(
      config$reference,
      list(
        list(title = "Additional release topics"),
        list(contents = extra)
      )
    )
  }
  articles <- list.files(
    file.path(release, "vignettes"),
    pattern = "[.]Rmd$",
    recursive = TRUE
  )
  articles <- sub("[.]Rmd$", "", articles)
  for (i in seq_along(config$articles)) {
    config$articles[[i]]$contents <- intersect(
      config$articles[[i]]$contents,
      articles
    )
  }
  config$development$mode <- "release"
  config$repo$branch <- tag
  yaml::write_yaml(config, file.path(release, "_pkgdown.yml"))
  file.copy(file.path(root, "pkgdown"), release, recursive = TRUE)

  links <- list(
    c("querying-data.Rmd", "(hubness.Rmd)", "(hubness.html)"),
    c(
      "articles/fmnist-example.Rmd",
      "(jlmelville/snedata)",
      "(https://github.com/jlmelville/snedata)"
    ),
    c(
      "articles/whatsnew.Rmd",
      "(NEWS.md)",
      "(https://jlmelville.github.io/rnndescent/news/index.html)"
    )
  )
  for (link in links) {
    article <- file.path(release, "vignettes", link[1])
    if (file.exists(article)) {
      writeLines(
        gsub(link[2], link[3], readLines(article), fixed = TRUE),
        article
      )
    }
  }

  articles <- list.files(
    file.path(release, "vignettes"),
    pattern = "[.]Rmd$",
    recursive = TRUE,
    full.names = TRUE
  )
  for (article in articles) {
    lines <- readLines(article)
    headers <- grep("^```\\{r[[:space:]]+[^,}]+", lines)
    for (i in headers) {
      label <- sub("^```\\{r[[:space:]]+([^,}]+).*", "\\1", lines[i])
      safe_label <- gsub("[^[:alnum:]_.-]+", "-", trimws(label))
      lines[i] <- sub(label, safe_label, lines[i], fixed = TRUE)
    }
    writeLines(lines, article)
  }

  pkgdown::build_site_github_pages(
    release,
    dest_dir = destination,
    install = TRUE,
    new_process = TRUE
  )
  pkgdown::build_site_github_pages(
    root,
    dest_dir = destination,
    install = FALSE,
    new_process = TRUE
  )
}

build_sites()

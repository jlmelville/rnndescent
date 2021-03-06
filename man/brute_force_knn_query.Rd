% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rnndescent.R
\name{brute_force_knn_query}
\alias{brute_force_knn_query}
\title{Query Exact Nearest Neighbors by Brute Force}
\usage{
brute_force_knn_query(
  query,
  reference,
  k,
  metric = "euclidean",
  use_alt_metric = TRUE,
  n_threads = 0,
  verbose = FALSE
)
}
\arguments{
\item{query}{Matrix of \code{n} query items.}

\item{reference}{Matrix of \code{m} reference items. The nearest neighbors to the
queries are calculated from this data.}

\item{k}{Number of nearest neighbors to return.}

\item{metric}{Type of distance calculation to use. One of \code{"euclidean"},
\code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"},
\code{"correlation"} (1 minus the Pearson correlation), or
\code{"hamming"}.}

\item{use_alt_metric}{If \code{TRUE}, use faster metrics that maintain the
ordering of distances internally (e.g. squared Euclidean distances if using
\code{metric = "euclidean"}), then apply a correction at the end. Probably
the only reason to set this to \code{FALSE} is if you suspect that some
sort of numeric issue is occurring with your data in the alternative code
path.}

\item{n_threads}{Number of threads to use.}

\item{verbose}{If \code{TRUE}, log information to the console.}
}
\value{
the nearest neighbor graph as a list containing:
\itemize{
\item \code{idx} an n by k matrix containing the nearest neighbor indices in
\code{reference}.
\item \code{dist} an n by k matrix containing the nearest neighbor distances to the
items in \code{reference}.
}
}
\description{
Query Exact Nearest Neighbors by Brute Force
}
\examples{
# 100 reference iris items
iris_ref <- iris[iris$Species \%in\% c("setosa", "versicolor"), ]

# 50 query items
iris_query <- iris[iris$Species == "versicolor", ]

# For each item in iris_query find the 4 nearest neighbors in iris_ref
# If you pass a data frame, non-numeric columns are removed
# set verbose = TRUE to get details on the progress being made
iris_query_nn <- brute_force_knn_query(iris_query,
  reference = iris_ref,
  k = 4, metric = "euclidean", verbose = TRUE
)

# Manhattan (l1) distance
iris_query_nn <- brute_force_knn_query(iris_query,
  reference = iris_ref,
  k = 4, metric = "manhattan"
)
}

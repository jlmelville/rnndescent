% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rptree.R
\name{rpf_build}
\alias{rpf_build}
\title{Create a random projection forest nearest neighbor index}
\usage{
rpf_build(
  data,
  metric = "euclidean",
  use_alt_metric = TRUE,
  n_trees = NULL,
  leaf_size = 10,
  max_tree_depth = 200,
  margin = "auto",
  n_threads = 0,
  verbose = FALSE,
  obs = "R"
)
}
\arguments{
\item{data}{Matrix of \code{n} items to generate the index for, with observations
in the rows and features in the columns. Optionally, input can be passed
with observations in the columns, by setting \code{obs = "C"}, which should be
more efficient. Possible formats are \code{\link[base:data.frame]{base::data.frame()}}, \code{\link[base:matrix]{base::matrix()}}
or \code{\link[Matrix:sparseMatrix]{Matrix::sparseMatrix()}}. Sparse matrices should be in \code{dgCMatrix}
format. Dataframes will be converted to \code{numerical} matrix format
internally, so if your data columns are \code{logical} and intended to be used
with the specialized binary \code{metric}s, you should convert it to a logical
matrix first (otherwise you will get the slower dense numerical version).}

\item{metric}{Type of distance calculation to use. One of:
\itemize{
\item \code{"braycurtis"}
\item \code{"canberra"}
\item \code{"chebyshev"}
\item \code{"correlation"} (1 minus the Pearson correlation)
\item \code{"cosine"}
\item \code{"dice"}
\item \code{"euclidean"}
\item \code{"hamming"}
\item \code{"hellinger"}
\item \code{"jaccard"}
\item \code{"jensenshannon"}
\item \code{"kulsinski"}
\item \code{"sqeuclidean"} (squared Euclidean)
\item \code{"manhattan"}
\item \code{"rogerstanimoto"}
\item \code{"russellrao"}
\item \code{"sokalmichener"}
\item \code{"sokalsneath"}
\item \code{"spearmanr"} (1 minus the Spearman rank correlation)
\item \code{"symmetrickl"} (symmetric Kullback-Leibler divergence)
\item \code{"tsss"} (Triangle Area Similarity-Sector Area Similarity or TS-SS
metric)
\item \code{"yule"}
}

For non-sparse data, the following variants are available with
preprocessing: this trades memory for a potential speed up during the
distance calculation. Some minor numerical differences should be expected
compared to the non-preprocessed versions:
\itemize{
\item \code{"cosine-preprocess"}: \code{cosine} with preprocessing.
\item \code{"correlation-preprocess"}: \code{correlation} with preprocessing.
}

For non-sparse binary data passed as a \code{logical} matrix, the following
metrics have specialized variants which should be substantially faster than
the non-binary variants (in other cases the logical data will be treated as
a dense numeric vector of 0s and 1s):
\itemize{
\item \code{"dice"}
\item \code{"hamming"}
\item \code{"jaccard"}
\item \code{"kulsinski"}
\item \code{"matching"}
\item \code{"rogerstanimoto"}
\item \code{"russellrao"}
\item \code{"sokalmichener"}
\item \code{"sokalsneath"}
\item \code{"yule"}
}

Note that if \code{margin = "explicit"}, the metric is only used to determine
whether an "angular" or "Euclidean" distance is used to measure the
distance between split points in the tree.}

\item{use_alt_metric}{If \code{TRUE}, use faster metrics that maintain the
ordering of distances internally (e.g. squared Euclidean distances if using
\code{metric = "euclidean"}). Probably the only reason to set this to \code{FALSE} is
if you suspect that some sort of numeric issue is occurring with your data
in the alternative code path. Only applies if the implicit \code{margin} method
is used.}

\item{n_trees}{The number of trees to use in the RP forest. A larger number
will give more accurate results at the cost of a longer computation time.
The default of \code{NULL} means that the number is chosen based on the number
of observations in \code{data}.}

\item{leaf_size}{The maximum number of items that can appear in a leaf. This
value should be chosen to match the expected number of neighbors you will
want to retrieve when running queries (e.g. if you want find 50 nearest
neighbors set \code{leaf_size = 50}) and should not be set to a value smaller
than \code{10}.}

\item{max_tree_depth}{The maximum depth of the tree to build (default = 200).
If the maximum tree depth is exceeded then the leaf size of a tree may
exceed \code{leaf_size} which can result in a large number of neighbor distances
being calculated. If \code{verbose = TRUE} a message will be logged to indicate
that the leaf size is large. However, increasing the \code{max_tree_depth} may
not help: it may be that there is something unusual about the distribution
of your data set under your chose \code{metric} that makes a tree-based
initialization inappropriate.}

\item{margin}{A character string specifying the method used to  assign points
to one side of the hyperplane or the other. Possible values are:
\itemize{
\item \code{"explicit"} categorizes all distance metrics as either Euclidean or
Angular (Euclidean after normalization), explicitly calculates a hyperplane
and offset, and then calculates the margin based on the dot product with
the hyperplane.
\item \code{"implicit"} calculates the distance from a point to each of the
points defining the normal vector. The margin is calculated by comparing the
two distances: the point is assigned to the side of the hyperplane that
the normal vector point with the closest distance belongs to.
\item \code{"auto"} (the default) picks the margin method depending on whether a
binary-specific \code{metric} such as \code{"bhammming"} is chosen, in which case
\code{"implicit"} is used, and \code{"explicit"} otherwise: binary-specific metrics
involve storing the data in a way that isn't very efficient for the
\code{"explicit"} method and the binary-specific metric is usually a lot faster
than the generic equivalent such that the cost of two distance calculations
for the margin method is still faster.
}}

\item{n_threads}{Number of threads to use.}

\item{verbose}{If \code{TRUE}, log information to the console.}

\item{obs}{set to \code{"C"} to indicate that the input \code{data} orientation stores
each observation as a column. The default \code{"R"} means that observations are
stored in each row. Storing the data by row is usually more convenient, but
internally your data will be converted to column storage. Passing it
already column-oriented will save some memory and (a small amount of) CPU
usage.}
}
\value{
a forest of random projection trees as a list. Each tree in the
forest is a further list, but is not intended to be examined or manipulated
by the user. As a normal R data type, it can be safely serialized and
deserialized with \code{\link[base:readRDS]{base::saveRDS()}} and \code{\link[base:readRDS]{base::readRDS()}}. To use it for
querying pass it as the \code{forest} parameter of \code{\link[=rpf_knn_query]{rpf_knn_query()}}. The forest
does not store any of the \code{data} passed into build the tree, so if you
are going to search the forest, you will also need to store the \code{data} used
to build it and provide it during the search.
}
\description{
Builds a "forest" of Random Projection Trees (Dasgupta and Freund, 2008),
which can later be searched to find approximate nearest neighbors.
}
\examples{
# Build a forest of 10 trees from the odd rows
iris_odd <- iris[seq_len(nrow(iris)) \%\% 2 == 1, ]
iris_odd_forest <- rpf_build(iris_odd, n_trees = 10)

iris_even <- iris[seq_len(nrow(iris)) \%\% 2 == 0, ]
iris_even_nn <- rpf_knn_query(
  query = iris_even, reference = iris_odd,
  forest = iris_odd_forest, k = 15
)
}
\references{
Dasgupta, S., & Freund, Y. (2008, May).
Random projection trees and low dimensional manifolds.
In \emph{Proceedings of the fortieth annual ACM symposium on Theory of computing}
(pp. 537-546).
\doi{10.1145/1374376.1374452}.
}
\seealso{
\code{\link[=rpf_knn_query]{rpf_knn_query()}}
}

% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/train.R
\name{train}
\alias{train}
\title{Train forest using ranger for multiple imputation algorithms.}
\usage{
train(
  data = NULL,
  response_name = character(),
  predictor_names = character(),
  x = NULL,
  y = NULL,
  case_weights = numeric(),
  classification = NULL,
  n_tree = 10,
  replace = TRUE,
  sample_fraction = ifelse(replace, 1, 0.632),
  n_try = NULL,
  draw_predictor_weights = numeric(),
  names_of_always_draw = character(),
  split_rule = NULL,
  max_depth = 0,
  min_split_n_sample = 0,
  min_leaf_n_sample = 0,
  unordered_predictors = NULL,
  response_weights = numeric(),
  n_random_split = 1,
  alpha = 0.5,
  min_prop = 0.1,
  seed = 1L + sample.int(n = .Machine$integer.max - 1L, size = 1),
  save_memory = FALSE,
  n_thread = 0,
  verbose = FALSE
)
}
\arguments{
\item{data}{Training data of class \code{data.frame}, \code{matrix}, or \code{dgCMatrix}
(Matrix), for the latter two; must have column names.}

\item{response_name}{Name of response (dependent) variable if \code{data} was
provided.}

\item{predictor_names}{Names of predictor (independent) variables if \code{data}
was provided; default is all variables that are not the response.}

\item{x}{Predictor data (independent variables), alternative interface to
\code{data} and \code{response_name}.}

\item{y}{Response vector (dependent variable), alternative interface to
\code{data} and \code{response_name}.}

\item{case_weights}{Weights for sampling of training observations.
Observations with larger weights will be selected with higher probability
in the bootstrap (or sub-sampled) samples for each tree.}

\item{classification}{Set to \code{TRUE} to grow a classification forest if the
response is numeric (including if data is a matrix), else, a regression
forest is grown.}

\item{n_tree}{Number of trees (default 10).}

\item{replace}{Sample with replacement to train each tree.}

\item{sample_fraction}{Fraction of observations to sample to train each tree.
Default is 1 for sampling with replacement and 0.632 for sampling without
replacement. For classification, this can be a vector of class-specific
values.}

\item{n_try}{Number of variables (predictors) to draw that are candidates for
splitting each node by. Default is the (rounded down) square root of the
number of predictors. Alternatively, a single argument function returning an
integer, given the number of predictors.}

\item{draw_predictor_weights}{For predictor-drawing weights shared by all
trees; a numeric vector of \emph{non-negative} weights for each predictor. For
tree-specific predictor-drawing weights; a list of size \code{n_tree} containing
(non-negative) vectors with length equal to the number of predictors.}

\item{names_of_always_draw}{Character vector with predictor (variable) names
to be selected \emph{in addition} to the \code{n_try} predictors drawn as candidates to
split by.}

\item{split_rule}{Splitting rule. For classification estimation "gini",
"extratrees" or "hellinger" with default "gini". For regression "variance",
"extratrees", "maxstat" or "beta" with default "variance".}

\item{max_depth}{Maximal tree depth. A value of NULL or 0 (the default)
corresponds to unlimited depth, 1 to tree stumps (1 split per tree).}

\item{min_split_n_sample}{Minimal number of in-bag samples a node must have
in order to be split. Default 1 for classification and 5 for regression.}

\item{min_leaf_n_sample}{Minimum number of in-bag samples in a leaf node.}

\item{unordered_predictors}{Handling of unordered factor predictors. One of
"ignore", "order" and "partition". For the "extratrees" splitting rule the
default is "partition" for all other splitting rules "ignore".}

\item{response_weights}{Classification only: Weights for the response classes
(in order of the factor levels) in the splitting rule e.g. cost-sensitive
learning. Weights are also used by each tree to determine majority vote.}

\item{n_random_split}{"extratrees" split metric only: Number of random splits
to consider for each candidate splitting variable, default is 1.}

\item{alpha}{"maxstat" splitting rule only: Significance threshold to allow
splitting, default is 0.5, must be in the interval \verb{(0,1]}.}

\item{min_prop}{"maxstat" splitting rule only: Lower quantile of covariate
distribution to be considered for splitting, default is 0.1, must be in the
interval \verb{[0,0.5]}.}

\item{seed}{Random seed, an integer between 1 and \code{.Machine$integer.max}.
Default generates the seed from \code{R}, set to \code{0} to ignore the \code{R} seed and
use a C++ \code{std::random_device}.}

\item{save_memory}{Use memory saving (but slower) splitting mode. Warning:
This option slows down the tree growing, use only if you encounter memory
problems.}

\item{n_thread}{Number of threads. Default is determined by system, typically
the number of cores.}

\item{verbose}{Show computation status and estimated runtime.}
}
\value{
Object of class \code{literanger} with elements:
\describe{
\item{\code{tree_type}}{The type of tree in the forest, either classification
or regression.}
\item{\code{n_tree}}{The number of trees in the forest.}
\item{\code{training}}{The parameters for training that were passed at the time
the forest was trained.}
\item{\code{predictors}}{A list with the names of the predictors, the names of
the unordered predictors, and the levels of any factors.}
\item{\code{response}}{The levels and type indicator (e.g. logical, factor, etc)
of the response.}
\item{\code{oob_error}}{The misclassification rate or the mean square error
using out-of-bag samples.}
\item{\code{cpp11_ptr}}{An external pointer to the trained forest. DO NOT
MODIFY.}
}
}
\description{
'literanger' trains random forests for use in multiple imputation problems
via an adaptation of the 'ranger' R package. ranger is a fast
implementation of random forests (Breiman, 2001) or recursive partitioning,
particularly suited for high dimensional data (Wright et al, 2017a).
literanger supports prediction used in algorithms such as "Multiple
Imputation via Chained Equations" (Van Buuren, 2007).
}
\details{
literanger trains classification and regression forests using the original
Random Forest (Breiman, 2001) or extremely randomized trees (Geurts et
al, 2006) algorithms. The trained forest retains information about the in-bag
responses in each terminal node, thus facilitating a variation on the
algorithm for multiple imputation with random forests proposed by Doove et
al (2014). This algorithm should match the predictive distribution more
closely than using predictive mean matching.

The default split metric for classification trees is the Gini impurity, which
can be extended to use the extra-randomized trees rule (Geurts et al, 2006).
For binary responses, the Hellinger distance metric may be used instead
(Cieslak et al, 2012).

The default split metric for regression trees is the estimated variance,
which can be extended to include the extra-randomized trees rule, too.
Alternatively, the beta log-likelihood (Wright et al, 2017b) or maximally
selected rank statistics (Wright et al, 2019) are available.

When the \code{data} and \code{response_name} arguments are supplied the response
variable is identified by its corresponding column name. The type of response
may be used to determine the type of tree. If the response is a factor then
classification trees are used. If the response is numeric then regression
trees are used. The \code{classification} argument can be used to override the
default tree type when the response is numeric. Alternatively, use \code{x} and
\code{y} arguments to specify response and predictor; this can avoid conversions
and save memory. If memory usage issues persist, consider setting
\code{save_memory=TRUE} but be aware that this option slows down the tree growing.

The \code{min_split_n_sample} rule can be used to control the minimum number of
in-bag samples required to split a node; thus, as in the original algorithm,
nodes with fewer samples than \code{min_split_n_sample} are possible. To put a
floor under the number of samples per node, the \code{min_leaf_n_sample}
argument is used.

When drawing candidate predictors for splitting a node on, the predictors
identified by \code{names_of_always_draw} are included \emph{in addition} to the
\code{n_try} predictors that are randomly drawn. Another way to modify the way
predictors are selected is via the \code{draw_predictor_weights} argument, which
are normalised and interpreted as probabilities when drawing candidates. The
weights are assigned \emph{in the order they appear in the data}. Weights assigned
by \code{draw_predictor_weights} to variables in \code{names_of_always_draw} are
ignored. The usage of \code{draw_predictor_weights} can increase the computation
times for large forests.

Unordered-factor predictors can be handled in 3 different ways by using
\code{unordered_predictors}:
\itemize{
\item For "ignore" all factors are regarded ordered;
\item For "partition" all possible 2-partitions are considered for splitting.
\item For "order" and 2-class classification the factor levels are ordered by
their proportion falling in the second class, for regression by their
mean response, as described in Hastie et al. (2009), chapter 9.2.4. For
multi-class classification the factor levels are ordered by the first
principal component of the weighted covariance matrix of the contingency
table (Coppersmith et al, 1999).
}

The use of "order" is recommended, as it computationally fast and
can handle an unlimited number of factor levels. Note that the factors are
only reordered once and not again in each split.

Compared to the original package ranger, literanger excludes certain
features:
\itemize{
\item Formula interface.
\item Probability, survival, and quantile regression forests.
\item Support for class gwaa.data.
\item Measures of variable importance.
\item Regularisation of importance.
\item Access to in-bag data via R.
\item Support for user-specified hold-out data.
}
}
\examples{
## Classification forest with default settings
train(data=iris, response_name="Species")

## Prediction
train_idx <- sample(nrow(iris), 2/3 * nrow(iris))
iris_train <- iris[train_idx, ]
iris_test <- iris[-train_idx, ]
lr_iris <- train(data=iris_train, response_name="Species")
pred_iris <- predict(lr_iris, newdata=iris_test)
table(iris_test$Species, pred_iris$values)

}
\references{
\itemize{
\item Breiman, L. (2001). Random forests. \emph{Machine Learning}, 45, 5-32.
\doi{10.1023/A:1010933404324}.
\item Cieslak, D. A., Hoens, T. R., Chawla, N. V., & Kegelmeyer, W. P. (2012).
Hellinger distance decision trees are robust and skew-insensitive. \emph{Data
Mining and Knowledge Discovery}, 24, 136-158.
\doi{10.1007/s10618-011-0222-1}.
\item Coppersmith, D., Hong, S. J., & Hosking, J. R. (1999). Partitioning
nominal attributes in decision trees. \emph{Data Mining and Knowledge
Discovery}, 3, 197-217. \doi{10.1023/A:1009869804967}.
\item Doove, L. L., Van Buuren, S., & Dusseldorp, E. (2014). Recursive
partitioning for missing data imputation in the presence of interaction
effects. \emph{Computational Statistics & Data Analysis}, 72, 92-104.
\doi{10.1016/j.csda.2013.10.025}.
\item Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees.
\emph{Machine Learning}, 63, 3-42. \doi{10.1007/s10994-006-6226-1}.
\item Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009).
The elements of statistical learning: data mining, inference, and
prediction (Vol. 2). New York: Springer.
\doi{10.1007/978-0-387-21606-5}.
\item Van Buuren, S. (2007). Multiple imputation of discrete and continuous
data by fully conditional specification. \emph{Statistical Methods in Medical
Research}, 16(3), 219-242. \doi{10.1177/0962280206074463}.
\item Weinhold, L., Schmid, M., Wright, M. N., & Berger, M. (2019). A random
forest approach for modeling bounded outcomes. \emph{arXiv preprint},
arXiv:1901.06211. \doi{10.48550/arXiv.1901.06211}.
\item Wright, M. N., & Ziegler, A. (2017a). ranger: A Fast Implementation of
Random Forests for High Dimensional Data in C++ and R. \emph{Journal of
Statistical Software}, 77, 1-17. \doi{10.18637/jss.v077.i01}.
\item Wright, M. N., Dankowski, T., & Ziegler, A. (2017b). Unbiased split
variable selection for random survival forests using maximally selected
rank statistics. \emph{Statistics in medicine}, 36(8), 1272-1284.
\doi{10.1002/sim.7212}.
}
}
\seealso{
\code{\link{predict.literanger}}
}
\author{
stephematician \href{mailto:stephematician@gmail.com}{stephematician@gmail.com}, Marvin N Wright (original
ranger package)
}

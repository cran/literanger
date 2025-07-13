#' Merge two random forests
#'
#' Copy the trees from two forests to construct a new random forest object.
#'
#' This is a naive implementation of a random-forest merge procedure. The trees
#' from each forest are copied and then used to construct a new random forest
#' object.
#'
#' Classification and regression forests cannot be mixed together. The response
#' type and levels (if a factor) must match.
#'
#' The predictor names, type, and levels (if a factor) must match, although they
#' can be provided in a different order.
#'
#' There is no requirement that the forests were trained on the same data; just
#' the same data types.
#'
#' Internally, literanger will 'map' any differences in the order of the
#' predictors (or its internal representation of response values) between `x`
#' and `y` so that the result has the same ordering as `x`.
#'
#' The out-of-bag error is discarded, along with the training information, as
#' the result is a _merged_ forest (not a _trained_ one). It is up to you,
#' the user, to keep track of the training parameters of `x` and `y` if they are
#' still of use to you.
#'
#' @param x A trained random forest `literanger` object.
#' @param y A trained random forest `literanger` object.
#' @param save_memory Ignored, only used in training (perhaps future use).
#' @param verbose Print additional debug output from merging procedure.
#' @param ... Ignored.
#'
#' @return Object of class `literanger` with a _copy_ of the trees from `x` and
#' `y` held in the `cpp11_ptr` item, and the following items:
#' \describe{
#'   \item{`tree_type`}{The type of tree in the forest, either 'classification'
#'     or 'regression'.}
#'   \item{`n_tree`}{The sum of the number of trees in `x` and `y`.}
#'   \item{`training`}{An empty list; as the result is due to merging, not
#'   training.}
#'   \item{`predictors`}{A list with the names of the predictors, the names of
#'     the unordered predictors, and the levels of any factors.}
#'   \item{`response`}{The levels and type indicator (e.g. logical, factor, etc)
#'     of the response.}
#'   \item{`oob_error`}{NULL, as there is no consensus on how to merge OOB
#'    estimates}
#'   \item{`cpp11_ptr`}{An external pointer to the _merged_ forest. DO NOT
#'     MODIFY.}
#' }
#'
#' @examples
#' ## Train two classification forests
#' train_idx <- sample(nrow(iris), 2/3 * nrow(iris))
#' iris_train <- iris[train_idx, ]
#' iris_test <- iris[-train_idx, ]
#' lr_x <- train(data=iris_train, response_name="Species", n_tree=32)
#' lr_y <- train(data=iris_train, response_name="Species", n_tree=32)
#'
#' ## Merge
#' lr_iris <- merge(lr_x, lr_y)
#' pred_iris <- predict(lr_iris, newdata=iris_test)
#' table(iris_test$Species, pred_iris$values)
#'
#' @author stephematician <stephematician@gmail.com>.
#'
#' @export
#' @md
merge.literanger <- function(x, y, save_memory=FALSE, verbose=FALSE, ...) {

  # check that objects are compatible
    if (x$tree_type != y$tree_type)
        stop(paste0("Tree type of forests does not match ['", x$tree_type,
                    "' vs '", y$tree_type, "']"))

    if (!setequal(x$predictors$names, y$predictors$names))
        stop("Set of predictors is not equal")

    if (!setequal(x$predictors$names_of_ordered, y$predictors$names_of_ordered))
        stop("Set of ordered predictors is not equal")

    nm_order <- x$predictors$names
    equal_levels <- mapply(setequal,
                           x$predictors[nm_order]$levels,
                           y$predictors[nm_order]$levels)
    if (!all(equal_levels))
        stop("Predictor levels are not identical")

    if (!identical(x$response, y$response))
        stop("Response must be identical")

    result <- cpp11_merge(x=x, y=y,
                          x_predictors=x$predictors$names,
                          y_predictors=y$predictors$names,
                          save_memory=save_memory, verbose=verbose)

    result$n_tree <- x$n_tree + y$n_tree
    result$training <- list() # TODO: merge training information? oob-error?
    result$predictors <- x$predictors
    result$response <- x$response

    class(result) <- "literanger"

    invisible(result)

}
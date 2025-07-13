# ------------------------------------------------------------------------------
# This file is part of 'literanger'. literanger was adapted from the 'ranger'
# package for R statistical software. ranger was authored by Marvin N Wright
# with the GNU General Public License version 3. The adaptation was performed by
# stephematician in 2023. literanger carries the same license, terms, and
# permissions as ranger.
#
# literanger is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# literanger is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with literanger. If not, see <https://www.gnu.org/licenses/>.
#
# Written by:
#
#   stephematician
#   stephematician@gmail.com
#   Australia
# ------------------------------------------------------------------------------

#' Serialize random forest
#'
#' Write a random forest to a file or connection using light-weight
#' serialization for C++ objects.
#'
#' This function uses ['cereal'][cereal_url] light-weight serialization to
#' write a literanger object (random forest) to a file or connection. The
#' file can be read in via [read_literanger()] and used for prediction
#' with no requirement for the original training data.
#'
#' [cereal_url]: https://uscilab.github.io/cereal/
#'
#' @param object A trained random forest `literanger` object.
#' @param file A connection or the name of the file where the `literanger`
#' object will be saved.
#' @param verbose Show additional serialization information (not implemented).
#' @param ... Further arguments passed to [saveRDS()].
#'
#' @examples
#' ## Classification forest
#' train_idx <- sample(nrow(iris), 2/3 * nrow(iris))
#' iris_train <- iris[ train_idx, ]
#' iris_test  <- iris[-train_idx, ]
#' lr_iris <- train(data=iris_train, response_name="Species")
#' file <- tempfile()
#' write_literanger(lr_iris, file)
#' lr_copy <- read_literanger(file)
#' pred_bagged <- predict(lr_copy, newdata=iris_test, prediction_type="bagged")
#'
#' @author stephematician <stephematician@gmail.com>
#'
#' @seealso [read_literanger()] [saveRDS]
#'
#' @export
#' @md
write_literanger <- function(object, file, verbose=FALSE, ...) {

    stopifnot(inherits(object, "literanger"))

    serialized <- list(cpp11=cpp11_serialize(object, verbose=verbose))

    serialized$tree_type <- object$tree_type
    serialized$n_tree <- object$n_tree
    serialized$training <- object$training
    serialized$predictors <- object$predictors
    serialized$response <- object$response

    saveRDS(object=serialized, file=file, ...)

    invisible(NULL)

}

#' De-serialize random forest
#'
#' Read the random forest from a file or connection using light-weight
#' serialization for C++ objects.
#'
#' This function uses ['cereal'][cereal_url] light-weight serialization to
#' read a literanger object (random forest) from a file or connection. The
#' file is usually the result of a call to [write_literanger()]. The random
#' forest returned can be used for prediction immediately upon return, and does
#' not require the original training data or training environment.
#'
#' [cereal_url]: https://uscilab.github.io/cereal/
#'
#' @param file A connection or the name of a file containing a serialized
#' `literanger` object.
#' @param verbose Show additional serialization information (not implemented).
#' @param ... Further arguments passed to [readRDS()].
#' @return A `literanger` random forest object
#'
#' @author stephematician <stephematician@gmail.com
#'
#' @seealso [write_literanger()] [readRDS]
#'
#' @export
#' @md
read_literanger <- function(file, verbose=FALSE, ...) {

    serialized <- readRDS(file=file, ...)

    object <- cpp11_deserialize(serialized$cpp11, verbose=verbose)

    object$tree_type <- serialized$tree_type
    object$n_tree <- serialized$n_tree
    object$training <- serialized$training
    object$predictors <- serialized$predictors
    object$response <- serialized$response

    class(object) <- "literanger"

    invisible(object)

}


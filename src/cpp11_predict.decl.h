/*-------------------------------------------------------------------------------
 * This file is part of 'literanger'. literanger was adapted from the 'ranger'
 * package for R Statistical Software <https://www.r-project.org>. ranger was
 * authored by Marvin N Wright with the GNU General Public License version 3.
 * The adaptation was performed by stephematician in 2023. literanger carries the
 * same license, terms, and permissions as ranger.
 *
 * literanger is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * literanger is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with literanger. If not, see <https://www.gnu.org/licenses/>.
 *
 * Written by:
 *
 *   stephematician
 *   stephematician@gmail.com
 *   Australia
 *-------------------------------------------------------------------------------
 */
#ifndef LITERANGER_CPP11_PREDICT_DECL_H
#define LITERANGER_CPP11_PREDICT_DECL_H

/* standard library headers */
#include <cstddef>

/* cpp11 and R headers */
#include "cpp11.hpp"

/** Predict values using a trained random forest
 *
 * See file R/predict.R in the R package for further details.
 *
 * @param[in] object A random forest object returned by `literanger::train` to
 * generate predictions with.
 * @param[in] x The predictor data as a numeric matrix; each column is a
 * predictor and each row is an observation (or case).
 * @param[in] sparse_x Optional (set to NULL in R to disable); the predictor
 * data represented using a sparse matrix structure, same layout as in \p x, but
 * the underlying data structure more compactly represents matrices with lots of
 * zeros.
 * @param[in] prediction_type Name of the prediction algorithm; "bagged" is the
 * most-frequent value among in-bag samples for classification, or the mean of
 * in-bag responses for regression; "inbag" predicts by drawing one in-bag
 * response from a random tree for each row in @p x or @p sparse_x; "nodes"
 * returns the node keys (ids) of the terminal node from every tree for each
 * row.
 * @param[in] seed Seed used for (seeding) pseudo-random number generators for
 * each tree.
 * @param[in] n_thread The number of threads to use for training, if zero then
 * the number of threads will be the value returned by
 * `std::thread::hardware_concurrency`.
 * @param[in] verbose Indicator for additional printed output while growing and
 * predicting.
 *
 * @returns A list (in R) with:
 * -   `values`: the predicted value(s) for each node, depending on prediction
 *     type and tree type. For "bagged" and "inbag" @p prediction_type, a
 *     numeric value for each row.
 */
cpp11::list cpp11_predict(
    cpp11::list object,
    cpp11::doubles_matrix<> x, cpp11::sexp sparse_x,
    std::string prediction_type, const size_t seed,
    const size_t n_thread, const bool verbose
);


#endif /* LITERANGER_CPP11_PREDICT_DECL_H */


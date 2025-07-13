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
#ifndef LITERANGER_CPP11_MERGE_DECL_H
#define LITERANGER_CPP11_MERGE_DECL_H

#include <cstddef>
#include <unordered_map>

/* cpp11 and R headers */
#include "cpp11.hpp"

/** Merge random forests
 *
 * See file R/merge.R in the R package for further details.
 *
 * @param[in] x A random forest object returned by `literanger::train`.
 * @param[in] y A random forest object returned by `literanger::train`.
 * @param[in] x_predictors The recorded predictor names from training @p x.
 * @param[in] y_predictors The recorded predictor names from training @p y.
 * @param[in] verbose Indicator for additional printed output while merging.
 *
 * @returns A list (in R) with:
 * -   `values`: the predicted value(s) for each node, depending on prediction
 *     type and tree type. For "bagged" and "inbag" @p prediction_type, a
 *     numeric value for each row.
 */
cpp11::list cpp11_merge(cpp11::list x,
                        cpp11::list y,
                        cpp11::strings x_predictors,
                        cpp11::strings y_predictors,
                        const bool save_memory,
                        const bool verbose);

template <typename T>
std::unordered_map<size_t,size_t> make_key_map(const T from_values,
                                               const T to_values);

#endif /* LITERANGER_CPP11_MERGE_DECL_H */


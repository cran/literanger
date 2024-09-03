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
#ifndef LITERANGER_CPP11_IO_DECL_H
#define LITERANGER_CPP11_IO_DECL_H

/* standard library headers */
#include <cstddef>
#include <string>

/* cpp11 and R headers */
#include "cpp11.hpp"


/** Serialize a random forest
 *
 * @param[in] object A random forest object returned by `literanger::train`.
 * @param[in] verbose Indicator for additional printed output while growing and
 * predicting.
 *
 * @returns A list (in R) with:
 * -   `values`: the predicted value(s) for each node, depending on prediction
 *     type and tree type. For "bagged" and "inbag" @p prediction_type, a
 *     numeric value for each row.
 */
cpp11::raws cpp11_serialize(cpp11::list object, const bool verbose);

cpp11::list cpp11_deserialize(cpp11::raws object, const bool verbose);


#endif /* LITERANGER_CPP11_PREDICT_DECL_H */


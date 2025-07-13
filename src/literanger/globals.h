/* This file is part of the C++ core of 'literanger'.
 *
 * literanger's C++ core was adapted from the C++ core of the 'ranger' package
 * for R Statistical Software <https://www.r-project.org>. The ranger C++ core
 * is Copyright (c) [2014-2018] [Marvin N. Wright] and distributed with MIT
 * license. literanger's C++ core is distributed with the same license, terms,
 * and permissions as ranger's C++ core.
 *
 * Copyright [2023] [stephematician]
 *
 * This software may be modified and distributed under the terms of the MIT
 * license. You should have received a copy of the MIT license along with
 * literanger. If not, see <https://opensource.org/license/mit/>.
 */
#ifndef LITERANGER_GLOBALS_H
#define LITERANGER_GLOBALS_H

/* standard library headers */
#include <bitset>
#include <cstddef>
#include <limits>
#include <vector>


namespace literanger {

using count_vector = std::vector<size_t>;
using key_vector = std::vector<size_t>;
using bool_vector = std::vector<bool>;
using dbl_vector = std::vector<double>;
using ull_bitenc = std::bitset<std::numeric_limits<unsigned long long>::digits>;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr size_t DEFAULT_N_THREAD = 0;

constexpr size_t DEFAULT_MIN_SPLIT_N_SAMPLE_CLASSIFICATION = 2;
constexpr size_t DEFAULT_MIN_LEAF_N_SAMPLE_CLASSIFICATION = 1;
constexpr size_t DEFAULT_MIN_SPLIT_N_SAMPLE_REGRESSION = 5;
constexpr size_t DEFAULT_MIN_LEAF_N_SAMPLE_REGRESSION = 1;

constexpr double STATUS_INTERVAL = 30.0;

// Threshold for q value split method switch
constexpr double Q_THRESHOLD = 0.02;


} /* namespace literanger */


#endif /* LITERANGER_GLOBALS_H */


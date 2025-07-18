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
#ifndef LITERANGER_DATA_DECL_H
#define LITERANGER_DATA_DECL_H

/* standard library headers */
#include <cstddef>
#include <string>
#include <vector>

/* general literanger headers */
#include "literanger/globals.h"


namespace literanger {

/** Data for random forests (interface). */
struct Data {

    public:

        /** Construct data with column names.
         * @param[in] n_row The number of observations.
         * @param[in] n_col The number of predictors. */
        Data(const size_t n_row, const size_t n_col) noexcept;

        /** Non-copyable. @param[in] rhs right-hand side of copy. */
        Data(const Data & rhs) = delete;
        /** Non-assignable. @param[in] rhs right-hand side of assignment. */
        Data & operator=(const Data & rhs) = delete;
        /** Virtual destructor for pure-abstract class. */
        virtual ~Data();

        /** Get the recorded value of a predictor.
         *
         * Given a predictor key, get either the recorded value from the row
         * (i.e. sample key) in the original data or, optionally, from a row
         * in a permutation of the dataset - if a permutation has been
         * constructed via new_permutation().
         *
         * @param[in] sample_key The observation key, i.e. the row offset in the
         * non-permuted or permuted data set.
         * @param[in] predictor_key The predictor key, a.k.a the column offset
         * in the original dataset.
         * @param[in] permute Indicator whether to use non-permuted (false) or
         * permuted (true) predictor data.
         * @returns The numeric value of the predictor as recorded in the
         * dataset (or permutation thereof if requested). */
        virtual double get_x(const size_t sample_key,
                             const size_t predictor_key,
                             const bool permute = false) const noexcept = 0;

        /** Get the recorded value of the response.
         * @param[in] sample_key The row offset, a.k.a. the observation key.
         * @param[in] column The column offset, a.k.a. the index into the vector
         * of responses for each observation. */
        virtual double get_y(const size_t sample_key,
                             const size_t column) const noexcept = 0;

        /** Get the number of predictors a.k.a. number of columns in data. */
        size_t get_n_col() const noexcept;

        /** Get the number of observations i.e. number of rows in the data. */
        size_t get_n_row() const noexcept;

        /** Get all values for predictor (_sorted_, no duplicates) for a given
         * subset of observations.
         * @param[out] all_values Container to hold values.
         * @param[in] sample_keys The keys (row index) corresponding to each
         * observation.
         * @param[in] predictor_key The predictor from which values will be
         * extracted (a.k.a. column offset).
         * @param[in] start The start of the subset of the observation keys.
         * @param[in] end The (past-the-)end of the subset of the observation
         * keys to extract values from. */
        void get_all_values(dbl_vector & all_values,
                            const key_vector & sample_keys,
                            const size_t predictor_key,
                            const size_t start, const size_t end,
                            const bool permute = false) const;

        /** Get the least and greatest value of a predictor for a given subset
         * of observations.
         * @param[out] all_values Container to store values.
         * @param[in] sample_keys The keys (row index) corresponding to each
         * observation.
         * @param[in] predictor_key The predictor from which values will be
         * extracted (a.k.a. column offset).
         * @param[in] start The start of the subset of the observation keys.
         * @param[in] end The (past-the-)end of the subset of the observation
         * keys to extract values from. */
        void get_minmax_values(double & min, double & max,
                               const key_vector & sample_keys,
                               const size_t predictor_key,
                               const size_t start, const size_t end,
                               const bool permute = false) const;

        /** Initialise an index for each predictor using its unique values.
         *
         * For each predictor; make a container with the (ordered/sorted) unique
         * values, and; make a index for the observations which specifies the
         * offset into the container (of unique values) for each observation. */
        void new_predictor_index() const;

        /** Clear the index for each predictor */
        void finalise_predictor_index() const noexcept;

        /** Indicator that index for each predictor is available */
        bool has_predictor_index() const noexcept;

        /** Get the offset into the (_sorted_) unique values of a predictor for
         * a given observation.
         * @param[in] sample_key The sample (row offset) in the non-permuted
         *  or permuted data set.
         * @param[in] predictor_key The predictor (column offset) in the
         * original data set.
         * @param[in] permute Indicator to use non-permuted (false) or permuted
         * (true) predictors.
         * @returns The index (offset) into the unique values of the predictor.
         */
        size_t get_unique_key(const size_t sample_key,
                              const size_t predictor_key,
                              const bool permute = false) const;

        /** @copydoc Data::get_unique_key */
        size_t rawget_unique_key(const size_t sample_key,
                                 const size_t predictor_key,
                                 const bool permute = false) const noexcept;

        /** Get the recorded value of a predictor given the offset into the
         * unique-value (_sorted_) container.
         * @param[in] predictor_key The column offset a.k.a. predictor key.
         * @param[in] offset The offset into the unique-value container for the
         * predictor. */
        double get_unique_value(const size_t predictor_key,
                                const size_t offset) const;

        /** Get the number of unique values observed for a predictor.
         * @param[in] predictor_key The column offset a.k.a. predictor key. */
        size_t get_n_unique_value(const size_t predictor_key) const;

        /** Maximum number of unique values observed across all predictors. */
        size_t get_max_n_unique_value() const noexcept;

        /** Get the unique values of the response in order of appearance. */
        const dbl_vector & get_response_values() const;

        /** Initialise a new container of response keys, for each observation,
         * for a given ordering (index) of the response values.
         * @param[in] response_values A container of response values sorted in
         * the proposed order. */
        void new_response_index(const dbl_vector & response_values) const;

        /** Clear the response-key container. */
        void finalise_response_index() const noexcept;

        /** Get the response-key container. */
        const key_vector & get_response_index() const noexcept;

        /** Initialise a container of observation keys categorised by the
         * response key. */
        void new_sample_keys_by_response() const;

        /** Clear the categorised observation-keys container. */
        void finalise_sample_keys_by_response() const noexcept;

        /** Get the categorised observation-keys container. */
        const std::vector<key_vector> &
        get_sample_keys_by_response() const noexcept;

        /** Initialise a permutation of the observation keys.
         * @param[in] seed The seed used for pseudo-random permutation. */
        void new_permutation(const size_t seed) const;

        /** Clear the permutation of the observation keys. */
        void finalise_permutation() const noexcept;


    protected:

        /** Convert an observation key into a row offset in the data set.
         * @param[in] sample_key The observation key, i.e. the row offset in the
         * non-permuted or permuted data set.
         * @param[in] permute Indicator whether @p sample_key is a row offset
         * in the original dataset (false) or the permuted dataset (true). */
        size_t as_row_offset(const size_t sample_key,
                             const bool permute = false) const noexcept;

        /** Number of rows, a.k.a. observations, in dataset. */
        const size_t n_row = 0;

        /** Number of columns, a.k.a. predictors, in dataset. */
        const size_t n_col = 0;

        /** The (sorted) unique values for each predictor. */
        mutable std::vector<dbl_vector> unique_predictor_values;

        /** The maximum number of unique values for any predictor. */
        mutable size_t max_n_unique_value = 0ull;

        /** The key (offset) into the unique-value index for each observed
         * value of a predictor, stored as a vector. */
        mutable key_vector predictor_index;

        /** A container of sorted values of the response. */
        mutable dbl_vector response_values;

        /** A container of the key (offset) for each observation into the sorted
         * values of the response. */
        mutable key_vector response_index;

        /** A container of the observation key (row offset) stored by the key
         * for each response-value. */
        mutable std::vector<key_vector> sample_keys_by_response;

        /** Permutation of the rows of the original (predictor) dataset. */
        mutable key_vector permuted_sample_keys;


};


} /* namespace literanger */


#endif /* LITERANGER_DATA_DECL_H */


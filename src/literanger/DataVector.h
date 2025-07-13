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
#ifndef LITERANGER_DATA_VECTOR_H
#define LITERANGER_DATA_VECTOR_H

/* base class definition */
#include "literanger/Data.h"

/* standard library headers */
#include <cstddef>
#include <stdexcept>

#include "globals.h"

/* cereal types */
#include "cereal/types/vector.hpp"


namespace literanger {

/** Data for random forests using matrix predictor and response. */
struct DataVector : public Data {

    public:

        /** Move-construct data from vector containers.
         * @param[in] X Predictor data in column-major order with one predictor
         * per column and one observation (case) per row.
         * @param[in] y Response data in column-major order with one observation
         * (or case) per row and one response component per column. */
        DataVector(const size_t n_row, const size_t n_col,
                   dbl_vector && X, dbl_vector && y);

        /** Copy-construct data from vector containers.
         * @param[in] X Predictor data in column-major order with one predictor
         * per column and one observation (case) per row.
         * @param[in] y Response data in column-major order with one observation
         * (or case) per row and one response component per column. */
        DataVector(const size_t n_row, const size_t n_col,
                   const dbl_vector & X, const dbl_vector & y);

        /** @copydoc Data::~Data */
        virtual ~DataVector() override = default;

        /** @copydoc Data::get_x */
        double get_x(const size_t sample_key,
                     const size_t predictor_key,
                     const bool permute = false) const noexcept override;

        /** @copydoc Data::get_y */
        double get_y(const size_t sample_key,
                     const size_t column) const noexcept override;

        /** @name Enable cereal for DataVector. */
        /**@{*/
        template <typename archive_type>
        void serialize(archive_type & archive);

        template <typename archive_type>
        void load_and_construct(archive_type & archive,
                                cereal::construct<DataVector> & construct);
        /**@}*/


    private:

        /** Vector containing the column-major values of the predictors */
        dbl_vector X;
        /** Vector containing the response */
        dbl_vector y;


};


/* Member definitions */

inline DataVector::DataVector(const size_t n_row,
                              const size_t n_obs,
                              dbl_vector && X,
                              dbl_vector && y) :
    Data(n_row, n_col), X(std::move(X)), y(std::move(y)) {

    if (X.size() % n_row != 0)
        throw std::invalid_argument("Mismatch between number of observations "
            "and size of 'X'");
    if (X.size() % n_col != 0)
        throw std::invalid_argument("Mismatch between number of predictors "
            "and size of 'X'");
    if (y.size() % n_row != 0)
        throw std::invalid_argument("Mismatch between number of observations "
            "and size of 'y'");

}


inline DataVector::DataVector(const size_t n_row, const size_t n_col,
                              const dbl_vector & X, const dbl_vector & y) :
    Data(n_row, n_col), X(X), y(y) {

    if (X.size() % n_row != 0)
        throw std::invalid_argument("Mismatch between number of observations "
            "and size of 'X'");
    if (X.size() % n_col != 0)
        throw std::invalid_argument("Mismatch between number of predictors "
            "and size of 'X'");
    if (y.size() % n_row != 0)
        throw std::invalid_argument("Mismatch between number of observations "
            "and size of 'y'");

}


template <typename archive_type>
void DataVector::serialize(archive_type & archive) {
    archive(n_row, n_col, X, y);
}


template <typename archive_type>
void DataVector::load_and_construct(
    archive_type & archive,
    cereal::construct<DataVector> & construct
) {
    size_t n_obs, n_col;
    dbl_vector X, y;

    archive(n_row, n_col, X, y);

    construct(n_row, n_col, std::move(X), std::move(y));

}


inline double DataVector::get_x(const size_t sample_key,
                                const size_t predictor_key,
                                const bool permute) const noexcept {
    return X[as_row_offset(sample_key, permute) + n_row * predictor_key];
}


inline double DataVector::get_y(const size_t sample_key,
                                const size_t column) const noexcept {
    return y[sample_key + n_row * column];
}


} /* namespace literanger */


#endif /* LITERANGER_DATA_VECTOR_H */


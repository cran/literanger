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
#ifndef LITERANGER_TRAINING_PARAMETERS_H
#define LITERANGER_TRAINING_PARAMETERS_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/* cereal types */
#include "cereal/types/memory.hpp"
#include "cereal/types/string.hpp" /* for SplitRule? */
#include "cereal/types/vector.hpp"

/* general literanger headers */
#include "literanger/enum_types.h"
#include "literanger/globals.h"


namespace literanger {

/** Generic parameters for a tree in a random forest.
 *
 * Parameters that describe the sampling, drawing, and splitting of a tree from
 * a random forest. A vector of these parameters is passed to the forest
 * constructor which dictates how many trees and what the values of the
 * parameters for each tree are. */
struct TrainingParameters {

    using key_vector_ptr = std::shared_ptr<key_vector>;
    using dbl_vector_ptr = std::shared_ptr<dbl_vector>;

    TrainingParameters() = default;

    /** Generic tree parameter constructor
     * @param[in] replace Whether to sample with replacement when training.
     * @param[in] sample_fraction The fraction of observations to use to train
     * each tree.
     * @param[in] n_try The number of candidate predictors for each split.
     * @param[in] draw_always_predictor_keys The key of each predictor that will
     * always be a candidate for splitting (sorted by key).
     * @param[in] draw_predictor_weights Weights for each predictor when drawing
     * candidates.
     * @param[in] split_rule The rule for identifying the best split.
     * @param[in] min_metric_decrease The minimum change in the metric for an
     * acceptable split.
     * @param[in] max_depth The maximum depth of any tree.
     * @param[in] min_split_n_sample The minimum number of in-bag samples in a
     * node that may be split in the growth phase.
     * @param[in] min_leaf_n_sample The minimum number of in-bag samples in a
     * leaf node in the growth phase.
     * @param[in] n_random_split The number of values to draw when splitting
     * via the extratrees rule.
     * @param[in] min_prop The smallest proportion for max-stat splitting
     * @param[in] response_weights Weights for each class of the response in a
     * classification forest. */
    TrainingParameters(
        const bool replace, const dbl_vector_ptr sample_fraction,
        const size_t n_try, const key_vector_ptr draw_always_predictor_keys,
        const dbl_vector_ptr draw_predictor_weights,
        const dbl_vector_ptr response_weights,
        const SplitRule split_rule, const double min_metric_decrease,
        const size_t max_depth,
        const size_t min_split_n_sample,
        const size_t min_leaf_n_sample, const size_t n_random_split,
        const double min_prop
    );

    /** @name Resampling training data for growing (training) a tree. */
    /**@{*/
    /** Indicator for sampling with replacement when when training. */
    bool replace;

    /** The fraction of observations to use when training each tree (scalar) or,
     * when when a vector is supplied, the response-specific fractions. */
    dbl_vector_ptr sample_fraction;
    /**@}*/

    /** @name Drawing candidate predictors for node splitting. */
    /**@{*/
    /** Number of randomly-drawn predictors amongst the candidates at each
      * node split */
    size_t n_try;
    /** Predictors that are always candidates for splitting. */
    key_vector_ptr draw_always_predictor_keys;
    /** Weights for each predictor that determine probability of selection as a
     * candidate for splitting (see std::discrete_distribution). */
    dbl_vector_ptr draw_predictor_weights;
    /**@}*/

    /** @name Response parameters (currently in classification only) */
    /**@{*/
    /** Weights for each class of response in a classificaiton forest */
    dbl_vector_ptr response_weights;
    /**@}*/

    /** @name Node-splitting rules. */
    /**@{*/
    /** Rule for selecting the predictor and value to split on. */
    SplitRule split_rule;
    /** Minimum decrease in metric that will be accepted when splitting. */
    double min_metric_decrease;
    /** Maximum depth of the trees in the forest. */
    size_t max_depth;
    /** Minimum number of in-bag samples a node must have to consider for
     * splitting. */
    size_t min_split_n_sample;
    /** Minimum number of in-bag samples in a leaf node. */
    size_t min_leaf_n_sample;
    /** Number of random splits to draw when using extra-random trees
      * algorithm. */
    size_t n_random_split;
    /* The smallest proportion for a child-node (compared to parent) when using
     * max-stat splitting rule */
    double min_prop;
    /**@}*/


};


inline TrainingParameters::TrainingParameters(
    const bool replace, const dbl_vector_ptr sample_fraction,
    const size_t n_try, const key_vector_ptr draw_always_predictor_keys,
    const dbl_vector_ptr draw_predictor_weights,
    const dbl_vector_ptr response_weights,
    const SplitRule split_rule, const double min_metric_decrease,
    const size_t max_depth,
    const size_t min_split_n_sample,
    const size_t min_leaf_n_sample, const size_t n_random_split,
    const double min_prop
) :
    replace(replace), sample_fraction(sample_fraction),
    n_try(n_try), draw_always_predictor_keys(draw_always_predictor_keys),
    draw_predictor_weights(draw_predictor_weights),
    response_weights(response_weights),
    split_rule(split_rule), min_metric_decrease(min_metric_decrease),
    max_depth(max_depth), min_split_n_sample(min_split_n_sample),
    min_leaf_n_sample(min_leaf_n_sample), n_random_split(n_random_split),
    min_prop(min_prop)
{
    if (this->n_try == 0) throw std::domain_error("'n_try' must be positive.");
    if (this->split_rule == EXTRATREES && this->n_random_split == 0)
        throw std::domain_error("'n_random_split' must be positive.");
}


} /* namespace literanger */


#endif /* LITERANGER_TRAINING_PARAMETERS_H */


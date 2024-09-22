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
#ifndef LITERANGER_ENUM_TYPES_H
#define LITERANGER_ENUM_TYPES_H

/* standard library headers */
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

/* cereal types */
#include "cereal/types/string.hpp"

namespace literanger {

/** Enumerated tree types. */
enum TreeType { TREE_CLASSIFICATION, TREE_REGRESSION };


/** Enumerated rules for selecting a predictor to split on. */
enum SplitRule { LOGRANK, MAXSTAT, EXTRATREES, BETA, HELLINGER };


/** Enumerated types of prediction. */
enum PredictionType {
    BAGGED, /**< Each predicted value is bootstrap-aggregated over all trees */
    INBAG,  /**< Each predicted value comes from one randomly-sampled tree */
    NODES   /**< Return terminal node-id for every tree */
};


template <PredictionType prediction_type>
using enable_if_bagged =
    typename std::enable_if<prediction_type == BAGGED, std::nullptr_t>::type;

template <PredictionType prediction_type>
using enable_if_inbag =
    typename std::enable_if<prediction_type == INBAG, std::nullptr_t>::type;

template <PredictionType prediction_type>
using enable_if_nodes =
    typename std::enable_if<prediction_type == NODES, std::nullptr_t>::type;


/* Declarations */

/** Convert a string to enumerated tree type.
 * @param x "classification" or "regression" only supported. */
TreeType as_tree_type(std::string x);
/** Convert a string to enumerated splitting rule.
 * @param x e.g. "gini", "variance", etc. */
SplitRule as_split_rule(std::string x);
/** Convert a string to enumerated prediction type.
 * @param x "bagged", "inbag" or "nodes". */
PredictionType as_prediction_type(std::string x);

std::string as_string(TreeType x);
std::string as_string(SplitRule x);


/* Definitions */

inline TreeType as_tree_type(std::string x) {

    static std::unordered_map<std::string,TreeType> table = {
        { "classification", TreeType::TREE_CLASSIFICATION },
        { "regression", TreeType::TREE_REGRESSION }
    };

    const auto it = table.find(x);

    if (it == std::end(table))
        throw std::invalid_argument("Invalid tree type.");

    return it->second;

}


inline SplitRule as_split_rule(std::string x) {

    static std::unordered_map<std::string,SplitRule> table = {
        { "gini", SplitRule::LOGRANK },
        { "variance", SplitRule::LOGRANK },
        { "maxstat", SplitRule::MAXSTAT },
        { "extratrees", SplitRule::EXTRATREES },
        { "beta", SplitRule::BETA },
        { "hellinger", SplitRule::HELLINGER }
    };

    auto it = table.find(x);

    if (it == std::end(table))
        throw std::invalid_argument("Invalid split metric.");

    return it->second;

}


inline PredictionType as_prediction_type(std::string x) {

    static std::unordered_map<std::string,PredictionType> table = {
        {"bagged", PredictionType::BAGGED},
        {"inbag", PredictionType::INBAG},
        {"nodes", PredictionType::NODES}
    };

    auto it = table.find(x);

    if (it == std::end(table))
        throw std::invalid_argument("Invalid prediction type.");

    return it->second;

}


inline std::string as_string(TreeType x) {

    static std::unordered_map<TreeType, std::string> table = {
        { TreeType::TREE_CLASSIFICATION, "classification" },
        { TreeType::TREE_REGRESSION, "regression" }
    };

    return table.find(x)->second;

};


inline std::string as_string(SplitRule x) {

    static std::unordered_map<SplitRule,std::string> table = {
        {  SplitRule::LOGRANK, "gini" }, /* also 'variance' */
        {  SplitRule::MAXSTAT, "maxstat" },
        {  SplitRule::EXTRATREES, "extratrees" },
        {  SplitRule::BETA, "beta" },
        {  SplitRule::HELLINGER, "hellinger" }
    };

    return table.find(x)->second;

}


} /* namespace literanger */


namespace cereal {

    template <typename archive_type>
    std::string save_minimal(const archive_type & archive,
                             const literanger::TreeType & x) {
        return literanger::as_string(x);
    }

    template <typename archive_type>
    std::string save_minimal(const archive_type & archive,
                             const literanger::SplitRule & x) {
        return literanger::as_string(x);
    }

    template <typename archive_type>
    void load_minimal(const archive_type & archive,
                    literanger::TreeType & y, const std::string & x) {
        y = literanger::as_tree_type(x);
    }

    template <typename archive_type>
    void load_minimal(const archive_type & archive,
                      literanger::SplitRule & y, const std::string & x) {
        y = literanger::as_split_rule(x);
    }

} /* namespace cereal */


#endif /* LITERANGER_ENUM_TYPES_H */


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

/* call declaration */
#include "cpp11_merge.decl.h"

/* standard library headers */
#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/* cpp11 and R headers */
#include "cpp11.hpp"

/* general literanger headers */
#include "literanger/enum_types.h"
#include "literanger/globals.h"
#include "literanger/utility.h"
/* required literanger class headers */
#include "literanger/ForestClassification.h"
#include "literanger/ForestRegression.h"
#include "literanger/TreeClassification.h"
#include "literanger/TreeRegression.h"


[[cpp11::register]]
cpp11::list cpp11_merge(
    cpp11::list x, cpp11::list y,
    cpp11::strings x_predictors, cpp11::strings y_predictors,
    const bool save_memory, const bool verbose
) {

    using namespace literanger;
    using namespace cpp11::literals;

  /* Return value */
    cpp11::writable::list result;
    std::unique_ptr<ForestBase> forest{ };

  /* Logging */
    toggle_print print_out { verbose, Rprintf };

  /* Check compatibility */
    std::string tree_type = (
        cpp11::as_cpp<std::string>(x["tree_type"])
    );
    if (tree_type != cpp11::as_cpp<std::string>(y["tree_type"]))
        throw std::invalid_argument("Forest type must match");

    cpp11::external_pointer<ForestBase> x_forest = x["cpp11_ptr"];
    cpp11::external_pointer<ForestBase> y_forest = y["cpp11_ptr"];

  /* copy n_predictor and is_ordered for the merged forest */
    const size_t n_predictor = x_forest->get_n_predictor();
    const std::shared_ptr<bool_vector> is_ordered {
        new bool_vector(*x_forest->get_is_ordered())
    };

    if (n_predictor != y_forest->get_n_predictor())
        throw std::invalid_argument("Forest predictor count must match");

  /* Make a map from second set of predictors to first set */
    auto predictor_map = make_key_map(y_predictors, x_predictors);

    for (auto map : predictor_map) {
      /* Check ordering property of predictors in the second forest */
        const std::shared_ptr<const bool_vector> y_is_ordered = (
            y_forest->get_is_ordered()
        );
        if ((*y_is_ordered)[map.first] != (*is_ordered)[map.second])
            throw std::invalid_argument("Predictors must have same ordered "
                "property.");
    }

  /* Merge forests */
    std::vector<std::unique_ptr<TreeBase>> trees;

  /* FIXME: maybe tidy this up? */
    switch (as_tree_type(tree_type)) {

    case TREE_CLASSIFICATION: {
        auto & x_impl = dynamic_cast<ForestClassification &>(*x_forest);
        auto & y_impl = dynamic_cast<ForestClassification &>(*y_forest);
        print_out("Merging classification forests");
        dbl_vector response_values = x_impl.get_response_values();

        std::unordered_map<size_t,size_t> key_map = make_key_map(
            y_impl.get_response_values(), response_values
        );

        print_out("Copying %i trees from \'x\'", x_impl.peek_trees().size());
        for (auto & tree_ptr: x_impl.peek_trees()) {
            const auto & tree_impl = (
                dynamic_cast<TreeClassification &>(*tree_ptr)
            );
            auto result_tree = make_tree<TreeClassification>(
                save_memory, n_predictor, is_ordered, tree_impl
            );
            trees.push_back(std::move(result_tree));
        }
        print_out("Copying %i trees from \'y\'", y_impl.peek_trees().size());
        for (auto & tree_ptr: y_impl.peek_trees()) {
            auto & tree_impl = dynamic_cast<TreeClassification &>(*tree_ptr);
            auto result_tree = make_tree<TreeClassification>(
                save_memory, n_predictor, is_ordered, tree_impl
            );
            result_tree->transform_split_keys(predictor_map);
            static_cast<TreeClassification &>(
                *result_tree
            ).transform_response_keys(key_map);
            trees.push_back(std::move(result_tree));
        }
        print_out("Constructing classification forest");
        forest = make_forest<ForestClassification>(
             save_memory, n_predictor, is_ordered,
             std::move(trees), std::move(response_values)
        );
    } break;

    case TREE_REGRESSION: {
        auto & x_impl = dynamic_cast<ForestRegression &>(*x_forest);
        auto & y_impl = dynamic_cast<ForestRegression &>(*y_forest);
        print_out("Merging regression forests");
        print_out("Copying %i trees from \'x\'", x_impl.peek_trees().size());
        for (auto & tree_ptr: x_impl.peek_trees()) {
            auto & tree_impl = dynamic_cast<TreeRegression &>(*tree_ptr);
            auto result_tree = make_tree<TreeRegression>(
                save_memory, n_predictor, is_ordered, tree_impl
            );
            trees.push_back(std::move(result_tree));
        }
        print_out("Copying %i trees from \'y\'", y_impl.peek_trees().size());
        for (auto & tree_ptr: y_impl.peek_trees()) {
            auto & tree_impl = dynamic_cast<TreeRegression &>(*tree_ptr);
            auto result_tree = make_tree<TreeRegression>(
                save_memory, n_predictor, is_ordered, tree_impl
            );
            result_tree->transform_split_keys(predictor_map);
            trees.push_back(std::move(result_tree));
        }
        print_out("Constructing regression forest");
        forest = make_forest<ForestRegression>(
            save_memory, n_predictor, is_ordered, std::move(trees)
        );
    } break;

    default: throw std::invalid_argument("Unsupported tree type.");
    }

  /* Return tree type, null oob-error, and external pointer to Forest */
    result.push_back({ "tree_type"_nm = x["tree_type"] });
    result.push_back({ "oob_error"_nm = R_NilValue });
    result.push_back({
        "cpp11_ptr"_nm = cpp11::external_pointer<ForestBase>(forest.release())
    });

    return result;

}


template <typename T>
std::unordered_map<size_t,size_t> make_key_map(
    const T from_values,
    const T to_values
) {
    const size_t n_value = from_values.size();
    std::unordered_map<size_t,size_t> key_map;

    for (size_t j_to = 0; j_to != n_value; ++j_to) {
        const size_t j_from = (
            std::find(from_values.cbegin(),
                      from_values.cend(),
                      to_values[j_to]) - from_values.cbegin()
        );

        if (j_from >= n_value)
            throw std::domain_error("Mapped value must be from same domain");
        if (key_map.count(j_from) > 0)
            throw std::domain_error("Mapping must be injective");

        key_map[j_from] = j_to;
    }

    return key_map;
}


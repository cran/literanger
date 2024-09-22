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
#include "cpp11_io.decl.h"

/* standard library headers */
#include <memory>
#include <sstream>

/* cpp11 and R headers */
#include "cpp11.hpp"

/* cereal headers - must be included before literanger definitions */
#include "cereal/archives/binary.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"

/* general literanger headers */
#include "literanger/enum_types.h"
#include "literanger/globals.h"
/* required literanger class headers */
#include "literanger/ForestClassification.h"
#include "literanger/ForestRegression.h"

/* literanger R package headers */
#include "cpp11_utility.h"



[[cpp11::register]]
cpp11::raws cpp11_serialize(
    cpp11::list object, const bool verbose
) {

    using namespace literanger;
    using dbl_vector_ptr = std::shared_ptr<dbl_vector>;

    // toggle_print print_out { verbose, Rprintf };

    const std::string tree_type = (
        cpp11::as_cpp<std::string>(object["tree_type"])
    );
    const auto predictor_names = as_vector<std::string>(
        object["predictor_names"]
    );
    const auto names_of_unordered = as_vector<std::string>(
        object["names_of_unordered"]
    );
    const size_t n_tree = cpp11::as_cpp<size_t>(object["n_tree"]),
                 n_try = cpp11::as_cpp<size_t>(object["n_try"]);
    const std::string split_rule = cpp11::as_cpp<std::string>(
        object["split_rule"]
    );
    const size_t max_depth = cpp11::as_cpp<size_t>(object["max_depth"]);
    const size_t min_metric_decrease = cpp11::as_cpp<size_t>(
        object["min_metric_decrease"]
    );
    const size_t min_split_n_sample = cpp11::as_cpp<size_t>(
        object["min_split_n_sample"]
    );
    const size_t min_leaf_n_sample = cpp11::as_cpp<size_t>(
        object["min_leaf_n_sample"]
    );
    const size_t seed = cpp11::as_cpp<size_t>(object["seed"]);
    const double oob_error = cpp11::as_cpp<double>(object["oob_error"]);
    const size_t n_random_split = (
        as_split_rule(split_rule) == EXTRATREES ?
            cpp11::as_cpp<size_t>(object["n_random_split"]) : 0ul
    );

    const dbl_vector_ptr response_values = (
        as_tree_type(tree_type) == TREE_CLASSIFICATION ?
            as_vector_ptr<double>(object["response_values"]) : dbl_vector_ptr()
    );

    std::stringstream ss;
    {
        std::unique_ptr<ForestBase> forest_ptr {
            cpp11::as_cpp<cpp11::external_pointer<ForestBase>>(
                object["cpp11_ptr"]
            ).get(),
        };
        cereal::BinaryOutputArchive oarchive(ss); // Create an output archive
        oarchive(tree_type, predictor_names, names_of_unordered,
                 n_tree, n_try, split_rule,
                 max_depth, min_metric_decrease,
                 min_split_n_sample, min_leaf_n_sample,
                 seed, oob_error,
                 n_random_split, response_values,
                 forest_ptr);
        forest_ptr.release();
    }

    ss.seekg(0, ss.end);
    cpp11::writable::raws result(ss.tellg());
    ss.seekg(0, ss.beg);
    std::copy(std::istreambuf_iterator<char>{ss},
              std::istreambuf_iterator<char>(),
              result.begin());

    return result;

}


[[cpp11::register]]
cpp11::list cpp11_deserialize(cpp11::raws object, const bool verbose) {

    using namespace literanger;
    using namespace cpp11::literals;
    using dbl_vector_ptr = std::shared_ptr<dbl_vector>;

    // toggle_print print_out { verbose, Rprintf };

    std::stringstream ss;
    std::copy(object.cbegin(), object.cend(), std::ostream_iterator<char>(ss));

    std::string tree_type;
    std::vector<std::string> predictor_names, names_of_unordered;
    size_t n_tree, n_try;
    std::string split_rule;
    size_t max_depth, min_metric_decrease,
           min_split_n_sample, min_leaf_n_sample,
           seed;
    double oob_error;
    size_t n_random_split;
    dbl_vector_ptr response_values;
    std::unique_ptr<ForestBase> forest_ptr;

    {
        cereal::BinaryInputArchive iarchive(ss); // Read from input archive
        iarchive(tree_type, predictor_names, names_of_unordered,
                 n_tree, n_try, split_rule,
                 max_depth, min_metric_decrease,
                 min_split_n_sample, min_leaf_n_sample,
                 seed, oob_error,
                 n_random_split, response_values,
                 forest_ptr);
    }

    const TreeType enum_tree_type(as_tree_type(tree_type));
    const SplitRule enum_split_rule(as_split_rule(split_rule));

    cpp11::writable::list result;

   /* Store the results (selected arguments) */
    result.push_back({ "predictor_names"_nm = predictor_names });
    result.push_back({ "names_of_unordered"_nm = names_of_unordered });
    result.push_back({ "tree_type"_nm = tree_type });
    result.push_back({ "n_tree"_nm = n_tree });
    result.push_back({ "n_try"_nm = n_try });
    result.push_back({ "split_rule"_nm = split_rule });
    result.push_back({ "max_depth"_nm = max_depth });
    result.push_back({ "min_metric_decrease"_nm = min_metric_decrease });
    result.push_back({ "min_split_n_sample"_nm = min_split_n_sample });
    result.push_back({ "min_leaf_n_sample"_nm = min_leaf_n_sample });
    // TODO:  min_prop ?
    result.push_back({ "seed"_nm = seed });
    result.push_back({ "oob_error"_nm = oob_error });

    if (enum_split_rule == EXTRATREES)
        result.push_back({ "n_random_split"_nm = n_random_split });

    if (enum_tree_type == TREE_CLASSIFICATION) {
        result.push_back({ "response_values"_nm = *response_values });
    }

    result.push_back({
        "cpp11_ptr"_nm = cpp11::external_pointer<ForestBase>(
            forest_ptr.release()
        )
    });

    return result;

}


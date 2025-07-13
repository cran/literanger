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
#include "cpp11_train.decl.h"

/* standard library headers */
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_map>

/* general literanger headers */
#include "literanger/enum_types.h"
#include "literanger/globals.h"
#include "literanger/utility.h"
/* literanger class headers */
#include "literanger/ForestClassification.h"
#include "literanger/ForestRegression.h"
#include "literanger/TrainingParameters.h"

/* literanger R package headers */
#include "cpp11_utility.h"
#include "DataR.h"
#include "DataSparse.h"


[[cpp11::register]]
cpp11::list cpp11_train(
    cpp11::doubles_matrix<> x, cpp11::doubles_matrix<> y, cpp11::sexp sparse_x,
    cpp11::doubles case_weights,
    std::string tree_type, const size_t n_tree,
    cpp11::strings predictor_names, cpp11::strings names_of_unordered,
    const bool replace, cpp11::doubles sample_fraction,
    size_t n_try,
    cpp11::list draw_predictor_weights, cpp11::strings names_of_always_draw,
    std::string split_rule, const size_t max_depth, size_t min_split_n_sample,
    size_t min_leaf_n_sample,
    cpp11::doubles response_weights,
    const size_t n_random_split, const double alpha, const double min_prop,
    const size_t seed, const bool save_memory, const size_t n_thread,
    const bool verbose
) {

    using namespace literanger;
    using namespace cpp11::literals;

    cpp11::writable::list result;

    std::unique_ptr<ForestBase> forest{ };
    std::shared_ptr<Data> data { };

    toggle_print print_out { verbose, Rprintf };
    R_user_interruptor user_interrupt { };

    const TreeType enum_tree_type(as_tree_type(tree_type));

  /* Convert the parameters for the forest to standard library types and set
   * default values. */
    const auto predictor_names_std = as_vector<std::string>(predictor_names);
    const auto names_of_unordered_std = as_vector<std::string>(
        names_of_unordered
    );
    const auto n_predictor = predictor_names_std.size();

    const auto sample_fraction_std = as_vector_ptr<double>(
        sample_fraction
    );

    set_n_try(n_try, predictor_names);
    const auto names_of_always_draw_std = as_vector<std::string>(
        names_of_always_draw
    );
    const auto draw_predictor_weights_std =
        as_nested_ptr<double,cpp11::doubles>(draw_predictor_weights);

    const SplitRule enum_split_rule(as_split_rule(split_rule));
    double min_metric_decrease;
    set_min_metric_decrease(min_metric_decrease, enum_split_rule, alpha);

    set_min_split_n_sample(min_split_n_sample, enum_tree_type);
    set_min_leaf_n_sample(min_leaf_n_sample, enum_tree_type);

    const auto response_weights_std = as_vector_ptr<double>(response_weights);

  /* Construct the container for the parameters of each tree in the forest. */
    std::vector<TrainingParameters> forest_parameters;
    const auto is_ordered = make_is_ordered(predictor_names_std,
                                            names_of_unordered_std);
    const auto draw_always_predictor_keys = make_draw_always_predictor_keys(
        predictor_names_std, names_of_always_draw_std, n_try
    );

    {
        const auto empty = std::shared_ptr<dbl_vector>(new dbl_vector());

        for (size_t j = 0; j != n_tree; ++j) {
            std::shared_ptr<dbl_vector> draw_predictor_weights_j;
            switch (draw_predictor_weights_std.size()) {
            case 0: { draw_predictor_weights_j = empty;
            } break;
            case 1: { draw_predictor_weights_j = draw_predictor_weights_std[0];
            } break;
            default: { draw_predictor_weights_j = draw_predictor_weights_std[j];
            }
            }
            set_draw_predictor_weights(
                draw_predictor_weights_j, n_predictor, n_try,
                *draw_always_predictor_keys
            );
            forest_parameters.emplace_back(
                replace, sample_fraction_std,
                n_try, draw_always_predictor_keys, draw_predictor_weights_j,
                response_weights_std,
                enum_split_rule, min_metric_decrease, max_depth,
                min_split_n_sample, min_leaf_n_sample, n_random_split,
                min_prop
            );
        }
    }


  /* Construct the data used for training */
    const bool use_sparse = sparse_x != R_NilValue;

    if (use_sparse) {

        cpp11::integers sp_Dim  = { sparse_x.attr("Dim") };

        if ((size_t)sp_Dim[1] != (size_t)predictor_names.size())
            throw std::domain_error("Mismatch between length of "
                "'predictor_names' and 'x'.");
    } else {
        if ((size_t)x.ncol() != (size_t)predictor_names.size())
            throw std::domain_error("Mismatch between length of "
                "'predictor_names' and 'x'.");
    }

    if (use_sparse) {
        data = std::shared_ptr<Data>(
            new DataSparse(cpp11::as_integers({ sparse_x.attr("Dim")}),
                           cpp11::as_integers({ sparse_x.attr("i")}),
                           cpp11::as_integers({ sparse_x.attr("p")}),
                           cpp11::as_doubles({ sparse_x.attr("x")}),
                           y)
        );
    } else {
        data = std::shared_ptr<Data>(new DataR(x, y));
    }


  /* Create the random forest object. */
    switch (enum_tree_type) {
    case TREE_CLASSIFICATION: {
        forest = make_forest<ForestClassification>(save_memory);
    } break;
    case TREE_REGRESSION: {
        forest = make_forest<ForestRegression>(save_memory);
    } break;
    default: throw std::invalid_argument("Unsupported tree type.");
    }


  /* Now train the forest */
    const size_t plant_n_thread = n_thread == DEFAULT_N_THREAD ?
        std::thread::hardware_concurrency() : n_thread;
    if (plant_n_thread == 0)
        throw std::domain_error("'n_thread' must be positive.");

    double oob_error;
    forest->plant(
        n_predictor, is_ordered, forest_parameters, data,
        as_vector_ptr<double>(case_weights), seed,
        plant_n_thread, true, user_interrupt, oob_error, print_out
    );
    // TODO: per-tree case weights?


  /* Store selected arguments or parameters not related to observations */
    result.push_back({ "tree_type"_nm = tree_type });
    result.push_back({ "n_try"_nm = n_try });
    result.push_back({ "min_split_n_sample"_nm = min_split_n_sample });
    result.push_back({ "min_leaf_n_sample"_nm = min_leaf_n_sample });

  /* Out-of-bag error estimate */
    result.push_back({ "oob_error"_nm = oob_error });

    // TODO: ??? confusion matrix

    result.push_back({
        "cpp11_ptr"_nm = cpp11::external_pointer<ForestBase>(forest.release())
    });

    return result;

}


/* Helpers to set default values. */

void set_n_try(size_t & n_try, cpp11::strings predictor_names) {
    if (n_try != 0) return;
    n_try = (size_t)std::max(1., std::sqrt((double)(predictor_names.size())));
}


void set_min_split_n_sample(size_t & min_split_n_sample,
                            const literanger::TreeType tree_type) {
    using namespace literanger;
    #if !defined(__GNUC__) || __GNUC__ >= 5
      using umap_key_t = TreeType;
    #else
      using umap_key_t = size_t;
    #endif

    if (min_split_n_sample != 0) return;

    static std::unordered_map<umap_key_t,size_t> table = {
        { TreeType::TREE_CLASSIFICATION,
          DEFAULT_MIN_SPLIT_N_SAMPLE_CLASSIFICATION },
        { TreeType::TREE_REGRESSION, DEFAULT_MIN_SPLIT_N_SAMPLE_REGRESSION }
    };

    min_split_n_sample = table[tree_type];
}


void set_min_leaf_n_sample(size_t & min_leaf_n_sample,
                           const literanger::TreeType tree_type) {
    using namespace literanger;
    #if !defined(__GNUC__) || __GNUC__ >= 5
      using umap_key_t = TreeType;
    #else
      using umap_key_t = size_t;
    #endif

    if (min_leaf_n_sample != 0) return;

    static std::unordered_map<umap_key_t,size_t> table = {
        { TreeType::TREE_CLASSIFICATION,
          DEFAULT_MIN_LEAF_N_SAMPLE_CLASSIFICATION },
        { TreeType::TREE_REGRESSION, DEFAULT_MIN_LEAF_N_SAMPLE_REGRESSION }
    };

    min_leaf_n_sample = table[tree_type];
}


void set_min_metric_decrease(double & min_metric_decrease,
                             const literanger::SplitRule split_rule,
                             const double alpha) {
    using namespace literanger;

    switch (split_rule) {
    case EXTRATREES: case LOGRANK: case HELLINGER: {
        min_metric_decrease = 0;
    } break;
    case BETA: {
        min_metric_decrease = -std::numeric_limits<double>::max();
    } break;
    case MAXSTAT: {
        min_metric_decrease = -alpha;
    } break;
    default: throw std::runtime_error("Unexpected value of split metric.");
    }

}

void set_draw_predictor_weights(
    std::shared_ptr<std::vector<double>> draw_predictor_weights,
    const size_t n_predictor, const size_t n_try,
    const std::vector<size_t> & draw_always_predictor_keys
) {

    if (draw_predictor_weights->empty()) return;

    if (draw_predictor_weights->size() != n_predictor)
        throw std::invalid_argument("Number of draw-predictor weights not "
            "equal to number of predictors.");

  /* indicator variable for belonging to always-draw predictor */
    std::vector<bool> is_always(n_predictor, false);
    for (auto key : draw_always_predictor_keys) is_always[key] = true;

    size_t n_zero_weight = 0;

    for (size_t j = 0; j != n_predictor; ++j) {
        double & w = (*draw_predictor_weights)[j];
        if (w < 0)
            throw std::domain_error("One or more draw-predictor weights not "
                "in range [0,Inf).");
        w = w != 0 && !is_always[j] ? w : (++n_zero_weight, 0);
    }

    if (n_predictor - n_zero_weight < n_try)
        throw std::invalid_argument("Too many zeros in draw-predictor weights. "
            "Need at least n_try variables to split at.");

}


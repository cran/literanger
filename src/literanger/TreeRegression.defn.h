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
#ifndef LITERANGER_TREE_REGRESSION_DEFN_H
#define LITERANGER_TREE_REGRESSION_DEFN_H

/* class declaration */
#include "literanger/TreeRegression.decl.h"

/* standard library headers */
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>

/* cereal types */
#include "cereal/types/polymorphic.hpp"
#include "cereal/types/unordered_map.hpp"
#include "cereal/types/utility.hpp"
#include "cereal/types/vector.hpp"

/* general literanger headers */
#include "literanger/utility_math.h"
/* required literanger class definitions */
#include "literanger/Data.defn.h"
#include "literanger/Tree.defn.h"
#include "literanger/TrainingParameters.h"


namespace literanger {

inline TreeRegression::TreeRegression(
    const bool save_memory, const size_t n_predictor,
    const cbool_vector_ptr is_ordered
):
    Tree(save_memory, n_predictor, is_ordered)
{ }


inline TreeRegression::TreeRegression(
    const bool save_memory, const size_t n_predictor,
    const cbool_vector_ptr is_ordered,
    key_vector && split_keys, dbl_vector && split_values,
    std::pair<key_vector,key_vector> && child_node_keys,
    std::unordered_map<size_t,dbl_vector> && leaf_values,
    std::unordered_map<size_t,double> && leaf_mean
) :
    Tree(save_memory, n_predictor, is_ordered,
         std::move(split_keys), std::move(split_values),
         std::move(child_node_keys)),
    leaf_values(std::move(leaf_values)),
    leaf_mean(std::move(leaf_mean))
{ }


inline TreeRegression::TreeRegression(
    const bool save_memory, const size_t n_predictor,
    const cbool_vector_ptr is_ordered,
    const TreeRegression & tree
) :
    Tree(save_memory, n_predictor, is_ordered, tree),
    leaf_values(tree.leaf_values),
    leaf_mean(tree.leaf_mean)
{ }


inline const std::unordered_map<size_t,dbl_vector> &
TreeRegression::get_leaf_values() const noexcept { return leaf_values; }


template <PredictionType prediction_type, typename result_type,
          enable_if_bagged<prediction_type>>
void TreeRegression::predict_from_inbag(
    const size_t node_key,
    result_type & result
) {

    using const_iterator = decltype(leaf_mean)::const_iterator;
    const const_iterator mean_it = leaf_mean.find(node_key);
    const bool have_prediction = mean_it != leaf_mean.cend();

    if (!have_prediction) {

        double leaf_sum = 0;
        for (const double & response : leaf_values.at(node_key))
            leaf_sum += response;
        if (leaf_values.at(node_key).empty()) return;
        leaf_mean[node_key] = leaf_sum / leaf_values.at(node_key).size();
        result = leaf_mean[node_key];

    } else result = mean_it->second;

}


template <PredictionType prediction_type, typename result_type,
          enable_if_inbag<prediction_type>>
void TreeRegression::predict_from_inbag(
    const size_t node_key,
    result_type & result
) {

    std::uniform_int_distribution<> U_rng(0,
                                          leaf_values.at(node_key).size() - 1);
    const size_t key = U_rng(gen);
    result = leaf_values.at(node_key)[key];

}


template <PredictionType prediction_type, typename result_type,
          enable_if_nodes<prediction_type>>
void TreeRegression::predict_from_inbag(
    const size_t node_key, result_type & result
) {
    result = node_key;
}


template <typename archive_type>
void TreeRegression::serialize(archive_type & archive) {
    archive(cereal::base_class<TreeBase>(this), leaf_values, leaf_mean);
}


template <typename archive_type>
void TreeRegression::load_and_construct(
    archive_type & archive, cereal::construct<TreeRegression> & construct
) {
    /* base-class constructor arguments */
    bool save_memory;
    size_t n_predictor;
    bool_vector_ptr is_ordered;
    key_vector split_keys;
    dbl_vector split_values;
    std::pair<key_vector,key_vector> child_node_keys;
    /* regression-specific constructor arguments */
    std::unordered_map<size_t,dbl_vector> leaf_values;
    std::unordered_map<size_t,double> leaf_mean;

    archive(save_memory, n_predictor, is_ordered,
            split_keys, split_values, child_node_keys);
    archive(leaf_values, leaf_mean);

    construct(
        save_memory, n_predictor, is_ordered,
        std::move(split_keys), std::move(split_values),
        std::move(child_node_keys),
        std::move(leaf_values), std::move(leaf_mean)
    );
}


inline void TreeRegression::new_growth(
    const TrainingParameters & parameters,
    const std::shared_ptr<const Data> data
) {
    const size_t n_sample = data->get_n_row();

    switch (parameters.split_rule) {
    case BETA: case EXTRATREES: case LOGRANK: case MAXSTAT: {
    } break;
    case HELLINGER: {
        throw std::invalid_argument("Unsupported split metric for regression.");
    } break;
    default: {
        throw std::invalid_argument("Invalid split metric.");
    } break; }

    leaf_values.clear();
    leaf_mean.clear();
  /* Guess for maximum number of leaf nodes */
    leaf_values.reserve(
        std::ceil(n_sample / (double)parameters.min_split_n_sample)
    );
    leaf_mean.reserve(
        std::ceil(n_sample / (double)parameters.min_split_n_sample)
    );
}


inline void TreeRegression::add_terminal_node(
    const size_t node_key, const std::shared_ptr<const Data> data,
    const key_vector & sample_keys
) {
    const size_t start = start_pos[node_key];
    const size_t end = end_pos[node_key];
    leaf_values[node_key].clear();
    leaf_values[node_key].reserve(end - start);

    for (size_t j = start; j != end; ++j)
        leaf_values[node_key].emplace_back(data->get_y(sample_keys[j], 0));
}


inline bool TreeRegression::compare_response(
    const std::shared_ptr<const Data> data,
    const size_t lhs_key, const size_t rhs_key
) const noexcept {
    return data->get_y(lhs_key, 0) == data->get_y(rhs_key, 0);
}


inline void TreeRegression::new_node_aggregates(
    const size_t node_key, const SplitRule split_rule,
    const std::shared_ptr<const Data> data,
    const key_vector & sample_keys
) {
  /* Compute sum of responses and the sum of the response-squared in node, or in
   * the maximally selected rank statistic case, sum the scores and squared
   * scores. */
    node_sum = 0;
    if (split_rule != MAXSTAT) {
        for (size_t j = start_pos[node_key]; j != end_pos[node_key]; ++j)
            node_sum += data->get_y(sample_keys[j], 0);
    } else {
        node_var = 0;
        const size_t n_sample = get_n_sample_node(node_key);
        for (size_t j = start_pos[node_key]; j != end_pos[node_key]; ++j)
            response_scores.emplace_back(data->get_y(sample_keys[j], 0));
        response_scores = rank(response_scores);
        for (const double & score : response_scores) node_sum += score;
        for (const double & score : response_scores)
            node_var += std::pow(score - node_sum / n_sample, 2);
        node_var /= (double)(n_sample - 1);
    }
}


inline void TreeRegression::finalise_node_aggregates() const noexcept {
    response_scores.clear();
    if (save_memory) response_scores.shrink_to_fit();
}


inline void TreeRegression::prepare_candidate_loop_via_value(
    const size_t split_key, const size_t node_key, const SplitRule split_rule,
    const std::shared_ptr<const Data> data,
    const key_vector & sample_keys
) const {

    const size_t n_candidate_value = candidate_values.size();

    if (node_n_by_candidate.size() < n_candidate_value) {
        node_n_by_candidate.resize(n_candidate_value);
        node_sum_by_candidate.resize(n_candidate_value);
    }
    std::fill_n(node_n_by_candidate.begin(), n_candidate_value, 0);
    std::fill_n(node_sum_by_candidate.begin(), n_candidate_value, 0);

    if (split_rule == BETA) {
        response_by_candidate.resize(n_candidate_value);
        for (auto & responses : response_by_candidate) responses.clear();
    }

    for (size_t j = start_pos[node_key]; j != end_pos[node_key]; ++j) {

        const size_t sample_key = sample_keys[j];
        const double response = split_rule != MAXSTAT ? (
            data->get_y(sample_key, 0)
        ) : (
            response_scores[j - start_pos[node_key]]
        );
        const size_t offset = std::distance(
            candidate_values.cbegin(),
            std::lower_bound(candidate_values.cbegin(), candidate_values.cend(),
                             data->get_x(sample_key, split_key))
        );

        ++node_n_by_candidate[offset];
        node_sum_by_candidate[offset] += response;
        if (split_rule == BETA)
            response_by_candidate[offset].emplace_back(response);

    }

}


inline void TreeRegression::prepare_candidate_loop_via_index(
    const size_t split_key, const size_t node_key, const SplitRule split_rule,
    const std::shared_ptr<const Data> data, const key_vector & sample_keys
) const {

    const size_t n_candidate_value =
        data->get_n_unique_value(split_key);

    if (node_n_by_candidate.size() < n_candidate_value) {
        node_n_by_candidate.resize(n_candidate_value);
        node_sum_by_candidate.resize(n_candidate_value);
    }
    std::fill_n(node_n_by_candidate.begin(), n_candidate_value, 0);
    std::fill_n(node_sum_by_candidate.begin(), n_candidate_value, 0);

    for (size_t j = start_pos[node_key]; j != end_pos[node_key]; ++j) {
        const size_t sample_key = sample_keys[j];
        const size_t offset = data->rawget_unique_key(sample_key, split_key);

        ++node_n_by_candidate[offset];
        node_sum_by_candidate[offset] += data->get_y(sample_key, 0);
    }

    if (split_rule == BETA) {

        response_by_candidate.resize(n_candidate_value);
        for (auto & responses : response_by_candidate) responses.clear();

        for (size_t j = start_pos[node_key]; j != end_pos[node_key]; ++j) {
            const size_t sample_key = sample_keys[j];
            const size_t k = data->rawget_unique_key(sample_key, split_key);
            response_by_candidate[k].emplace_back(data->get_y(sample_key, 0));
        }

    }

}


inline void TreeRegression::finalise_candidate_loop() const noexcept {

    Tree::finalise_candidate_loop();

    if (save_memory) {
      /* NOTE: release of memory may be implementation dependent */
        node_sum_by_candidate.clear();
        node_sum_by_candidate.shrink_to_fit();
        response_by_candidate.clear();
        response_by_candidate.shrink_to_fit();
    }

}


template <SplitRule split_rule, typename UpdateT>
void TreeRegression::best_decrease_by_real_value(
    const size_t split_key, const size_t n_sample_node,
    const size_t n_candidate_value, const size_t min_leaf_n_sample,
    double & best_decrease, size_t & best_split_key, UpdateT update_best_value
) const {

  /* NOTE: Pre-condition: n_candidate_value > 1. */
    size_t n_lhs = 0;
    double sum_lhs = 0;

    for (size_t j = 0; j != n_candidate_value - 1; ++j) {

        if (node_n_by_candidate[j] == 0) continue;

        n_lhs += node_n_by_candidate[j];
        sum_lhs += node_sum_by_candidate[j];
        if (n_lhs < min_leaf_n_sample) continue;

        const size_t n_rhs = n_sample_node - n_lhs;
        if (n_rhs < min_leaf_n_sample) break;

        const double sum_rhs = node_sum - sum_lhs;
        const double decrease = evaluate_decrease<split_rule>(
            n_lhs, n_rhs, sum_lhs, sum_rhs
        );

     /* If the decrease in node impurity has improved - then we update the best
      * split for the node. */
        if (decrease > best_decrease) {
            update_best_value(j);
            best_split_key = split_key;
            best_decrease = decrease;
        }

    }

}


template <SplitRule split_rule, typename CallableT>
void TreeRegression::best_decrease_by_partition(
    const size_t split_key, const size_t node_key,
    const std::shared_ptr<const Data> data, const key_vector & sample_keys,
    const size_t n_sample_node, const size_t n_partition,
    const size_t min_leaf_n_sample,
    CallableT to_partition_key,
    double & best_decrease, size_t & best_split_key, double & best_value
) const {

    if (split_rule == BETA) {
        node_n_by_candidate.assign(2, 0);
        response_by_candidate.assign(2, dbl_vector());
    }

  /* Start from one (cannot have empty lhs) */
    for (size_t j = 1; j != n_partition; ++j) {

      /* Get the bit-encoded partition value */
        ull_bitenc partition_key = to_partition_key(j);

        double sum_lhs = 0;
        size_t n_lhs = 0;

        for (size_t k = start_pos[node_key]; k != end_pos[node_key]; ++k) {
            const size_t sample_key = sample_keys[k];
            const size_t level_bit = std::floor(
                data->get_x(sample_key, split_key) - 1
            );
            if (!partition_key.test(level_bit)) {
                sum_lhs += data->get_y(sample_key, 0);
                ++n_lhs;
            }
          /* Use the node_n_by_candidate and response_by_candidate containers to
           * store the left and right hand side values */
            if (split_rule == BETA) {
                const size_t j = (size_t)partition_key.test(level_bit);
                ++node_n_by_candidate[j];
                response_by_candidate[j].push_back(data->get_y(sample_key, 0));
            }
        }
        if (n_lhs < min_leaf_n_sample) continue;

        const size_t n_rhs = n_sample_node - n_lhs;
        if (n_rhs < min_leaf_n_sample) continue;

        const double sum_rhs = node_sum - sum_lhs;
        const double decrease = evaluate_decrease<split_rule>(
            n_lhs, n_rhs, sum_lhs, sum_rhs
        );

        if (decrease > best_decrease) {
            (unsigned long long &)best_value = partition_key.to_ullong();
            best_split_key = split_key;
            best_decrease = decrease;
        }

    }

    if (save_memory) {
        node_n_by_candidate.clear();
        node_n_by_candidate.shrink_to_fit();
        response_by_candidate.clear();
        response_by_candidate.shrink_to_fit();
    }

}


template <typename UpdateT>
void TreeRegression::best_statistic_by_real_value(
    const size_t n_sample_node, const size_t n_candidate_value,
    const size_t min_leaf_n_sample, const double min_prop,
    double & this_decrease, UpdateT update_this_value, double & this_p_value
) const {

  /* NOTE: Pre-condition: n_candidate_value > 1.
   * smallest split to consider for this node */
    const size_t min_split = std::max(0.0, n_sample_node * min_prop - 1);

    double sum_lhs = 0;
    size_t n_lhs = 0;
    size_t this_j = n_candidate_value;

    for (size_t j = 0; j != n_candidate_value - 1; ++j) {

        if (node_n_by_candidate[j] == 0) continue;

        n_lhs += node_n_by_candidate[j];
        sum_lhs += node_sum_by_candidate[j];
        if (n_lhs < std::max(min_leaf_n_sample, min_split)) continue;

        const size_t n_rhs = n_sample_node - n_lhs;
        if (n_rhs < std::max(min_leaf_n_sample, min_split)) break;

        const double sum_rhs = node_sum - sum_lhs;
        const double decrease = evaluate_decrease<MAXSTAT>(
            n_lhs, n_rhs, sum_lhs, sum_rhs
        );

        if (decrease > this_decrease) {
            this_j = j;
            this_decrease = decrease;
        }

    }

    if (this_j != n_candidate_value) {
        update_this_value(this_j);
        const double p_value_Lausen92 = maxstat_p_value_Lausen92(
            this_decrease, min_prop
        );
        const double p_value_Lausen94 = maxstat_p_value_Lausen94(
            this_decrease, n_sample_node, node_n_by_candidate, this_j + 1
        );
        this_p_value = std::min(p_value_Lausen92, p_value_Lausen94);
    }

}


template <SplitRule split_rule, enable_if_logrank<split_rule>>
inline double TreeRegression::evaluate_decrease(
    const size_t n_lhs, const size_t n_rhs,
    const double sum_lhs, const double sum_rhs
) noexcept {
    return sum_rhs * sum_rhs / n_rhs + sum_lhs * sum_lhs / n_lhs;
}


template <SplitRule split_rule, enable_if_hellinger<split_rule>>
inline double TreeRegression::evaluate_decrease(
    const size_t n_lhs, const size_t n_rhs,
    const double sum_lhs, const double sum_rhs
) noexcept {
    return -INFINITY;
}


template <SplitRule split_rule, enable_if_maxstat<split_rule>>
inline double TreeRegression::evaluate_decrease(
    const size_t n_lhs, const size_t n_rhs,
    const double sum_lhs, const double sum_rhs
) const {
    const double n = n_lhs + n_rhs;
    const double mu = node_sum / n;
    const double var = node_var;
    const double S = sum_lhs;
    const double E = n_lhs * mu;
    const double V = n_lhs * (double)n_rhs * var / n;
    return std::abs((S - E) / std::sqrt(V));
}


template <SplitRule split_rule, enable_if_beta<split_rule>>
inline double TreeRegression::evaluate_decrease(
    const size_t n_lhs, const size_t n_rhs,
    const double sum_lhs, const double sum_rhs
) const {

  /* Need at least two observations per node to estimate parameters for
   * beta distribution. */
    if (n_lhs < 2 || n_rhs < 2) return -INFINITY;

    const size_t n_candidate_value = node_n_by_candidate.size();
    size_t j_lhs = 0;
    {
        size_t count = 0;
        for (size_t j = 0; j != n_candidate_value; ++j) {
            if (count == n_lhs) { j_lhs = j; break; }
            count += node_n_by_candidate[j];
        }
    }

    const double mu_lhs = sum_lhs / n_lhs, mu_rhs = sum_rhs / n_rhs;

    double var_lhs = 0, var_rhs = 0;
  /* get variance of lhs */
    for (size_t j = 0; j != j_lhs; ++j) {
        if (node_n_by_candidate[j] == 0) continue;
        for (const double & response : response_by_candidate[j])
            var_lhs += std::pow(response - mu_lhs, 2);
    }
    var_lhs /= (double)(n_lhs - 1);

  /* get variance of rhs */
    for (size_t j = j_lhs; j != n_candidate_value; ++j) {
        if (node_n_by_candidate[j] == 0) continue;
        for (const double & response : response_by_candidate[j])
            var_rhs += std::pow(response - mu_rhs, 2);
    }
    var_rhs /= (double)(n_rhs - 1);

    if (var_lhs <= std::numeric_limits<double>::epsilon() ||
            var_rhs <= std::numeric_limits<double>::epsilon())
        return -INFINITY;

    const double nu_lhs = mu_lhs * (1 - mu_lhs) / var_lhs - 1,
                 nu_rhs = mu_rhs * (1 - mu_rhs) / var_rhs - 1;

    double beta_lnL = 0;
  /* sum beta log likelihood on lhs */
    for (size_t j = 0; j != j_lhs; ++j) {
        if (node_n_by_candidate[j] == 0) continue;
        for (const double & response : response_by_candidate[j])
            beta_lnL += beta_log_likelihood(response, mu_lhs, nu_lhs);
    }
  /* sum beta log likelihood on r */
    for (size_t j = j_lhs; j != n_candidate_value; ++j) {
        if (node_n_by_candidate[j] == 0) continue;
        for (const double & response : response_by_candidate[j])
            beta_lnL += beta_log_likelihood(response, mu_rhs, nu_rhs);
    }

    return !std::isnan(beta_lnL) ? beta_lnL : -INFINITY;

}


} /* namespace literanger */


CEREAL_REGISTER_TYPE(literanger::TreeRegression);


#endif /* LITERANGER_TREE_REGRESSION_DEFN_H */


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
#ifndef LITERANGER_TREE_REGRESSION_DECL_H
#define LITERANGER_TREE_REGRESSION_DECL_H

/* base class declaration */
#include "literanger/Tree.decl.h"

/* standard library headers */
#include <cstddef>
#include <memory>
#include <vector>
#include <unordered_map>

/* general literanger headers */
#include "literanger/enum_types.h"
#include "literanger/globals.h"
/* required literanger class declarations */
#include "literanger/Data.decl.h"
#include "literanger/TrainingParameters.h"


namespace literanger {

struct TreeRegression: Tree<TreeRegression> {

    friend struct Tree<TreeRegression>;

    public:

        /** Construct a regression tree.
         *
         */
        TreeRegression(
            const bool save_memory,
            const size_t n_predictor,
            const cbool_vector_ptr is_ordered
        );

        /** @copydoc TreeRegression::TreeRegression(bool,size_t,bool_vector_ptr)
         * @param[in] split_keys The predictor key for each node that identifies
         * the variable to split by.
         * @param[in] split_values The value for each node that determines
         * whether a data point belongs in the left or right child.
         * @param[in] child_node_keys A pair of containers for left and right
         * child-node keys. */
        TreeRegression(
            const bool save_memory,
            const size_t n_predictor,
            const cbool_vector_ptr is_ordered,
            key_vector && split_keys,
            dbl_vector && split_values,
            std::pair<key_vector,key_vector> && child_node_keys,
            std::unordered_map<size_t,dbl_vector> && leaf_values,
            std::unordered_map<size_t,double> && leaf_mean
        );

        TreeRegression(
            const bool save_memory,
            const size_t n_predictor,
            const cbool_vector_ptr is_ordered,
            const TreeRegression & tree
        );

        /** @name Basic accessors */
        /*@{*/
        const std::unordered_map<size_t,dbl_vector> &
        get_leaf_values() const noexcept;
        /*@}*/

        /** Predict response for a leaf (terminal) node.
         *
         * Uses the responses that were in-bag during training of the tree to
         * predict or draw a response.
         *
         * @param[in] node_key Identifier of node.
         * @param[out] result The predicted or otherwise-drawn value.
         * @tparam prediction_type The enumerated type of prediction to perform
         * e.g. bagged, impute.
         * @tparam result_type The type of data to return; usually a single
         * value e.g. double. */
        template <PredictionType prediction_type, typename result_type,
                  enable_if_bagged<prediction_type> = nullptr>
        void predict_from_inbag(const size_t node_key,
                                result_type & result);

        template <PredictionType prediction_type, typename result_type,
                  enable_if_inbag<prediction_type> = nullptr>
        void predict_from_inbag(const size_t node_key,
                                result_type & result);

        template <PredictionType prediction_type, typename result_type,
                  enable_if_nodes<prediction_type> = nullptr>
        void predict_from_inbag(const size_t node_key,
                                result_type & result);

        /** @name Enable cereal for TreeRegression. */
        /**@{*/
        template <typename archive_type>
        void serialize(archive_type & archive);

        template <typename archive_type>
        static void load_and_construct(
            archive_type & archive,
            cereal::construct<TreeRegression> & construct
        );
        /**@}*/


    protected:

        double node_sum;

        double node_var;

        /** Sum of the responses for each candidate split value. */
        mutable dbl_vector node_sum_by_candidate;

        /** Responses in intervals defined by each candidate split value. Each
         * interval is closed at the right. Used for the beta split rule. */
        mutable std::vector<dbl_vector> response_by_candidate;

        /** Used by max-stat rule */
        mutable dbl_vector response_scores;

        /** A map from (leaf) node keys to the values of the response for
         * in-bag observations during training; used for drawing predictions
         * with PredictionType::BAGGED */
        std::unordered_map<size_t,dbl_vector> leaf_values;

        /** A map from node keys for each leaf node to the mean value of the
         * in-bag responses during growth (training). */
        mutable std::unordered_map<size_t,double> leaf_mean;


    private:

        /** Prepare a regression tree for growth by reserving space for
         * terminal nodes.
         *
         * @param[in] data Data to grow (or train) tree with. Contains
         * observations of predictors and the response, the former has
         * predictors across columns and observations by row, and the latter is
         * usually a column vector (or matrix). */
        void new_growth(const TrainingParameters & parameters,
                        const std::shared_ptr<const Data> data) override;

        /** @copydoc TreeBase::add_terminal_node() */
        void add_terminal_node(const size_t node_key,
                               const std::shared_ptr<const Data> data,
                               const key_vector & sample_keys) override;

        /** @copydoc TreeBase::compare_response() */
        bool compare_response(
            const std::shared_ptr<const Data> data,
            const size_t lhs_key,
            const size_t rhs_key
        ) const noexcept override;

        /** @copydoc Tree::new_node_aggregates() */
        void new_node_aggregates(
            const size_t node_key,
            const SplitRule split_rule,
            const std::shared_ptr<const Data> data,
            const key_vector & sample_keys
        ) override;

        /** @copydoc Tree::finalise_node_aggregates() */
        void finalise_node_aggregates() const noexcept override;

        /** @copydoc Tree::prepare_loop_invariants_via_value() */
        void prepare_candidate_loop_via_value(
            const size_t split_key,
            const size_t node_key,
            const SplitRule split_rule,
            const std::shared_ptr<const Data> data,
            const key_vector & sample_keys
        ) const override;

        /** @copydoc Tree::prepare_loop_invariants_via_index() */
        void prepare_candidate_loop_via_index(
            const size_t split_key,
            const size_t node_key,
            const SplitRule split_rule,
            const std::shared_ptr<const Data> data,
            const key_vector & sample_keys
        ) const override;

        /** @copydoc Tree::finalise_loop_invariants() */
        void finalise_candidate_loop() const noexcept override;

        /** Search the real-valued split candidates for the best decrease in
         * impurity and update the current best key, value, and decrease.
         *
         * @param[in] split_key Identifies which predictor to evaluate.
         * @param[in] n_sample_node The number of observations in the node.
         * @param[in] n_candidate_value The number of candidate values for
         * splitting.
         * @param[in] min_leaf_n_sample The minimum number of in-bag samples in
         * a leaf node.
         * @param[in,out] best_decrease The best decrease in node impurity
         * achieved by splitting.
         * @param[in,out] best_split_key The predictor which gave the best
         * decrease in node impurity.
         * @param[in] update_best_value A function that updates the best value
         * given an index into the candidate value vector. */
        template <SplitRule split_rule, typename UpdateT>
        void best_decrease_by_real_value(
            const size_t split_key,
            const size_t n_sample_node,
            const size_t n_candidate_value,
            const size_t min_leaf_n_sample,
            double & best_decrease,
            size_t & best_split_key,
            UpdateT update_best_value
        ) const;

        /** Search the partition candidates for the best decrease in impurity
         * and update the current best key, value, and decrease.
         *
         * @param[in] split_key Identifies which predictor to evaluate.
         * @param[in] node_key Identifies the node to evaluate.
         * @param[in] data Data to train forest with. Contains observations of
         * predictors and the response, the former has predictors across
         * columns and observations by row, and the latter is usually a column
         * vector (or matrix).
         * @param[in] sample_keys The partially-sorted keys in the sample for
         * this tree.
         * @param[in] n_sample_node The number of observations in the node.
         * @param[in] n_partition The total number of partitions.
         * @param[in] min_leaf_n_sample The minimum number of in-bag samples in
         * a leaf node.
         * @param[in] to_partition_key A function that converts an integer index
         * to a partition bit-mask.
         * @param[in,out] best_decrease The best decrease in node impurity
         * achieved by splitting.
         * @param[in,out] best_split_key The predictor which gave the best
         * decrease in node impurity.
         * @param[in,out] best_value The value to split by that achieved the
         * best decrease in node impurity. */
        template <SplitRule split_rule, typename CallableT>
        void best_decrease_by_partition(
            const size_t split_key,
            const size_t node_key,
            const std::shared_ptr<const Data> data,
            const key_vector & sample_keys,
            const size_t n_sample_node,
            const size_t n_partition,
            const size_t min_leaf_n_sample,
            CallableT to_partition_key,
            double & best_decrease,
            size_t & best_split_key,
            double & best_value
        ) const;

        template <typename UpdateT>
        void best_statistic_by_real_value(
            const size_t n_sample_node,
            const size_t n_candidate_value,
            const size_t min_leaf_n_sample,
            const double min_prop,
            double & this_decrease,
            UpdateT update_this_value,
            double & this_p_value
        ) const;

        /** Evaluates the decrease in node impurity given the counts to the left
         * of the split.
         *
         * @param[in] n_lhs The number of observations in the node to the left
         * of the split.
         * @param[in] n_rhs The number of observations in the node to the right
         * of the split. */
        template <SplitRule split_rule, enable_if_logrank<split_rule> = nullptr>
        static double evaluate_decrease(
            const size_t n_lhs,
            const size_t n_rhs,
            const double sum_lhs,
            const double sum_rhs
        ) noexcept;

        template <SplitRule split_rule, enable_if_beta<split_rule> = nullptr>
        double evaluate_decrease(
            const size_t n_lhs,
            const size_t n_rhs,
            const double sum_lhs,
            const double sum_rhs
        ) const;

        template <SplitRule split_rule, enable_if_maxstat<split_rule> = nullptr>
        double evaluate_decrease(
            const size_t n_lhs,
            const size_t n_rhs,
            const double sum_lhs,
            const double sum_rhs
        ) const;

        template <SplitRule split_rule, enable_if_hellinger<split_rule> = nullptr>
        static double evaluate_decrease(
            const size_t n_lhs,
            const size_t n_rhs,
            const double sum_lhs,
            const double sum_rhs
        ) noexcept;


};


} /* namespace literanger */


#endif /* LITERANGER_TREE_REGRESSION_DECL_H */


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
#ifndef LITERANGER_TREE_BASE_DECL_H
#define LITERANGER_TREE_BASE_DECL_H

/* standard library headers */
#include <cstddef>
#include <memory>
#include <random>
#include <utility>
#include <vector>

/* general literanger headers */
#include "literanger/enum_types.h"
#include "literanger/globals.h"
/* required literanger class declarations */
#include "literanger/Data.decl.h"
#include "literanger/TrainingParameters.h"


namespace literanger {

/** Abstract base of a tree-interface
 *
 * TODO: better documentation
 */
struct TreeBase {

    public:

        using dbl_vector_ptr  = std::shared_ptr<dbl_vector>;
        using cdbl_vector_ptr = std::shared_ptr<const dbl_vector>;
        using bool_vector_ptr = std::shared_ptr<bool_vector>;
        using cbool_vector_ptr = std::shared_ptr<const bool_vector>;

        /** Non-copyable. @param[in] rhs right-hand side of copy. */
        TreeBase(const TreeBase & x) = delete;
        /** Non-assignable. @param[in] rhs right-hand side of assignment. */
        TreeBase & operator=(const TreeBase & x) = delete;
        /** Virtual destructor for pure-abstract class. */
        virtual ~TreeBase() = default;

        /** @name Simple accessors. */
        /**@{*/
        const key_vector & get_split_keys() const noexcept;
        const dbl_vector & get_split_values() const noexcept;
        const key_vector & get_left_children() const noexcept;
        const key_vector & get_right_children() const noexcept;
        /**@}*/

        /** Seed the pseudo-random number generator engine.
         * @param[in] seed Value to seed TreeBase::gen with. */
        void seed_gen(const size_t seed);

        /** Grow (train) the tree using supplied data.
         * @param[in] parameters Parameters that govern (re)sampling of
         * observations, drawing candidates, and splitting nodes, see
         * literanger::TrainingParameters for details.
         * @param[in] data Data to train forest with, see literanger::Data class
         * for further details about format.
         * @param[in] case_weights The weight for each observation (row) in
         * training.
         * @param[in] compute_oob_error Indicator of whether to return the
         * out-of-bag keys or not.
         * @returns A vector of out-of-bags keys (empty if not requested). */
        key_vector grow(const TrainingParameters & parameters,
                        const std::shared_ptr<const Data> data,
                        const cdbl_vector_ptr case_weights,
                        const bool compute_oob_error);

        /** Map the keys used to identify predictors to new values; e.g. if
         * the columns of the data set have been re-ordered.
         * @param[in] key_map a map with index 'from-key' and value 'to-key' */
        void transform_split_keys(
            const std::unordered_map<size_t,size_t> map
        );

        /** Get the number of samples contained in a node.
         * @param[in] node_key The node to query.
         * @returns The number of samples in the node. */
        size_t get_n_sample_node(const size_t node_key) const noexcept;

        /** @name Enable cereal for TreeBase. */
        /**@{*/
        template <typename archive_type>
        void serialize(archive_type & archive);
        /**@}*/


    protected:

        /** Construct a tree object.
         * @param[in] save_memory Indicator whether to aggressively release
         * memory and omit building an index (which takes up memory but speeds
         * up training).
         * @param[in] n_predictor The number of predictors the tree must be
         * trained on or predict with.
         * @param[in] is_ordered Indicators for each predictor of whether or
         * not it is ordered. */
        TreeBase(const bool save_memory,
                 const size_t n_predictor,
                 const cbool_vector_ptr is_ordered);

        /** @copydoc TreeBase::TreeBase(bool,size_t,bool_vector_ptr)
         * @param[in] split_keys The predictor key for each node that identifies
         * the variable to split by.
         * @param[in] split_values The value for each node that determines
         * whether a data point belongs in the left or right child.
         * @param[in] child_node_keys A pair of containers for left and right
         * child-node keys. */
        TreeBase(const bool save_memory,
                 const size_t n_predictor,
                 const cbool_vector_ptr is_ordered,
                 key_vector && split_keys,
                 dbl_vector && split_values,
                 std::pair<key_vector,key_vector> && child_node_keys);

        TreeBase(const bool save_memory,
                 const size_t n_predictor,
                 const cbool_vector_ptr is_ordered,
                 const TreeBase & tree);

        /** @name Generic (immutable) tree parameters. */
        /*@{*/
        /** Aggressively release resources and use a unique value mapping. */
        const bool save_memory;
        /** The number of predictors that the tree must be trained on or predict
         * with. */
        const size_t n_predictor;
        /** Indicators for each predictor whether it is (treated as) ordered. */
        const cbool_vector_ptr is_ordered;
        /*@}*/

        /** Pseudo-random number generator for sampling observations (cases) and
         * drawing candidates. */
        std::mt19937_64 gen;

        /** The predictor key for each node that identifies the variable to
         * split by. */
        key_vector split_keys;

        /** The value for each node that determines whether a data point belongs
         * in the left or right child (given the predictor). */
        dbl_vector split_values;

        /** A pair of containers for left and right child-node keys. */
        std::pair<key_vector,key_vector> child_node_keys;

        /** Reference to the left child-node keys. */
        key_vector & left_children = child_node_keys.first;

        /** Reference to the left child-node keys. */
        key_vector & right_children = child_node_keys.second;

        /** The starting offset of the observations within a container of
         * partially-sorted observation keys for each node. */
        count_vector start_pos;

        /** The past-the-end offset of the observations within a container of
         * partially-sorted observation keys for each node. */
        count_vector end_pos;

        /** Count of the number of observations for each candidate split
         * value. */
        mutable count_vector node_n_by_candidate;

        /** Storage for candidate value (index) when selecting split. */
        mutable dbl_vector candidate_values;


    private:

        /** */
        void push_back_empty_node();

        /** Bootstrap/draw a sample from the set of keys `[0, 1, 2, ..., N-1]`
         * and optionally return the values _not_ drawn.
         *
         * @param[in] n_sample Both the number of keys to (randomly) draw and
         * the size of the set of keys.
         * @param[in] replace Whether to sample with replacement when training.
         * @param[in] sample_fraction The fraction of observations to use when
         * training; can be a vector for response-specific fractions.
         * @param[in] get_oob_keys Indicator for returning the out-of-bag
         * keys.
         * @param[out] sample_keys The randomly-drawn keys from the set.
         * @param[out] oob_keys The 'out-of-bag' keys - i.e. the keys that
         * aren't in `sample_keys`. */
        void resample_unweighted(
            const size_t n_sample,
            const bool replace,
            const cdbl_vector_ptr sample_fraction,
            const bool get_oob_keys,
            key_vector & sample_keys,
            key_vector & oob_keys
        );

        /** Boostrap/draw a sample from a set of keys `[0, 1, 2, ..., N-1]` where
         * each key has a user-provided probability of selection, and optionally
         * return the values _not_ drawn.
         *
         * @param[in] n_sample Both the number of keys to (randomly) draw and
         * the size of the set of keys.
         * @param[in] replace Whether to sample with replacement when training.
         * @param[in] sample_fraction The fraction of observations to use when
         * training; can be a vector for response-specific fractions.
         * @param[in] weights The weights or probabilities for each key.
         * @param[in] get_oob_keys Indicator for returning the out-of-bag
         * keys.
         * @param[out] sample_keys The randomly-drawn keys from the set.
         * @param[out] oob_keys The 'out-of-bag' keys - i.e. the keys that
         * aren't in `sample_keys`. */
        void resample_weighted(
            const size_t n_sample,
            const bool replace,
            const cdbl_vector_ptr sample_fraction,
            const cdbl_vector_ptr weights,
            const bool get_oob_keys,
            key_vector & sample_keys,
            key_vector & oob_keys
        );

        /** Bootsrap/draw a sample from the set of keys `[0, 1, 2, ..., N-1]`
         * with a user-specified fraction for each response value, and
         * optionally return the values _not_ drawn.
         *
         * @param[in] data Data to use for growth (training); the number of rows
         * is used to identify the size of the set to draw from.
         * @param[in] replace Whether to sample with replacement when training.
         * @param[in] sample_fraction The fraction of observations to use when
         * training; can be a vector for response-specific fractions.
         * @param[in] get_oob_keys Indicator for returning the out-of-bag
         * keys.
         * @param[out] sample_keys The randomly-drawn keys from the set.
         * @param[out] oob_keys The 'out-of-bag' keys - i.e. the keys that
         * aren't in `sample_keys`. */
        void resample_response_wise(
            const std::shared_ptr<const Data> data,
            const bool replace,
            const cdbl_vector_ptr sample_fraction,
            const bool get_oob_keys,
            key_vector & sample_keys,
            key_vector & oob_keys
        );

        /** Base-class implementation of response-wise resampling does nothing.
         *
         * @param[in] data Data used for growth (training).
         * @param[in] replace Whether to sample with replacement when training.
         * @param[in] sample_fraction The fraction of observations to use when
         * training; can be a vector for response-specific fractions.
         * @param[out] sample_keys A container of randomly drawn keys (i.e.
         * row-offsets) of observations.
         * @param[out] inbag_counts A container of counts of the number of
         * timees each observation appears in-bag. */
        virtual void resample_response_wise_impl(
            const std::shared_ptr<const Data> data,
            const bool replace,
            const cdbl_vector_ptr sample_fraction,
            key_vector & sample_keys,
            count_vector & inbag_counts
        );

        /**
         * Split a node using rules for selecting candidate predictors,
         * evaluating decrease in impurity, and selecting candidate values to
         * split by.
         *
         * @param[in] node_key The key (offset of split_vars vector etc) of the
         * current node.
         * @param[in] depth The current depth of the tree.
         * @param[in] last_left_node_key The most-recently generated left node
         * at the current depth.
         * @param[in] parameters Parameters that govern (re)sampling of
         * observations, drawing candidates, and splitting nodes, see
         * literanger::TrainingParameters for details.
         * @param[in] data Data to used for growth (training).
         * @param[out] sample_keys The partially-ordered keys where any key to
         * the right of another key is placed later in the container.
         * @returns Indicator for whether a split was performed: this is the
         * opposite of original ranger. */
        bool split_node(
            const size_t node_key,
            const size_t depth,
            const size_t last_left_node_key,
            const TrainingParameters & parameters,
            const std::shared_ptr<const Data> data,
            key_vector & sample_keys
        );

        /** Draw candidate predictors for splitting.
         * @returns A vector of predictor keys (column offsets) that are
         * candidates for splitting. */
        key_vector draw_candidates(const TrainingParameters & parameters);

        /** Prepare a tree for growth by reserving space for terminal nodes.
         *
         * @param[in] parameters Parameters that govern (re)sampling of
         * observations, drawing candidates, and splitting nodes, see
         * literanger::TrainingParameters for details.
         * @param[in] data Data to grow (or train) tree with. Contains
         * observations of predictors and the response, the former has
         * predictors across columns and observations by row, and the latter is
         * usually a column vector (or matrix). */
        virtual void new_growth(const TrainingParameters & parameters,
                                const std::shared_ptr<const Data> data) = 0;

        /** Default finalisation (do nothing) for growth phase.
         *
         * Implementation may use this to do any post-processing for terminal
         * nodes. */
        virtual void finalise_growth() const noexcept;

        /** */
        virtual void push_back_empty_node_impl();

        /** Store the observed values in the leaf (terminal) node container.
         *
         * @param[in] node_key The key for a new leaf (terminal) node.
         * @param[in] data Data to train forest with. Contains observations of
         * predictors and the response, the former has predictors across
         * columns and observations by row, and the latter is usually a column
         * vector (or matrix).
         * @param[in] sample_keys Container of partially ordered observation
         * keys (row offsets) used to grow the tree; any node that is left of
         * another is found later in the container. */
        virtual void add_terminal_node(const size_t node_key,
                                       const std::shared_ptr<const Data> data,
                                       const key_vector & sample_keys) = 0;

        /** Compare two responses for equality.
         * @param[in] data Data for a random forest (prediction or training).
         * Contains observations of predictors and the response, the latter is
         * usually a column vector (or matrix) with one observation (or case)
         * per row.
         * @param[in] lhs_key The row-offset of the left-hand-side of the
         * comparison.
         * @param[in] rhs_key THe row-offset of the right-hand-side of the
         * comparison.
         * @returns True if the response values are numerically equal. */
        virtual bool compare_response(const std::shared_ptr<const Data> data,
                                      const size_t lhs_key,
                                      const size_t rhs_key) const noexcept = 0;

        /** Add the best-performing split for a specified node; if no split
         * decreases impurity then do nothing.
         * @param[in] node_key The node to evaluate.
         * @param[in] parameters Parameters that govern (re)sampling of
         * observations, drawing candidates, and splitting nodes, see
         * literanger::TrainingParameters for details.
         * @param[in] data Data to use for growth (training).
         * @param[in] sample_keys The partially-sorted keys in the sample for
         * this tree.
         * @param[in] split_candidate_keys Identifies the predictors that are
         * candidates for splitting.
         * @returns Whether a split was added. */
        virtual bool push_best_split(
            const size_t node_key,
            const TrainingParameters & parameters,
            const std::shared_ptr<const Data> data,
            const key_vector & sample_keys,
            const key_vector & split_candidate_keys
        ) = 0;


};


template <typename T, typename... ArgsT>
std::unique_ptr<TreeBase> make_tree(ArgsT &&... args);


} /* namespace literanger */


#endif /* LITERANGER_TREE_BASE_DECL_H */


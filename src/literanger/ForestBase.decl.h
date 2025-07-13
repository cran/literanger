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
#ifndef LITERANGER_FOREST_BASE_DECL_H
#define LITERANGER_FOREST_BASE_DECL_H

/* standard library headers */
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

/* general literanger headers */
#include "literanger/enum_types.h"
#include "literanger/globals.h"
#include "literanger/utility.h" // toggle_print
#include "literanger/utility_interrupt.h" // interruptor
/* required literanger class declarations */
#include "literanger/Data.decl.h"
#include "literanger/TreeBase.decl.h"
#include "literanger/TrainingParameters.h"


namespace literanger {

/** Abstract base of a random forest interface. */
struct ForestBase {

    public:

        using dbl_vector_ptr  = std::shared_ptr<dbl_vector>;
        using cdbl_vector_ptr = std::shared_ptr<const dbl_vector>;
        using bool_vector_ptr  = std::shared_ptr<bool_vector>;
        using cbool_vector_ptr = std::shared_ptr<const bool_vector>;

        /** Non-copyable. @param[in] rhs right-hand side of copy. */
        ForestBase(const ForestBase & x) = delete;
        /** Non-assignable. @param[in] rhs right-hand side of assignment. */
        ForestBase & operator=(const ForestBase & x) = delete;
        /** Virtual destructor for pure-abstract class. */
        virtual ~ForestBase() = default;

        /** @name Simple accessors */
        /*@{*/
        size_t size() const noexcept;

        size_t get_n_predictor() const noexcept;

        cbool_vector_ptr get_is_ordered() const noexcept;

        const std::vector<std::unique_ptr<TreeBase>> &
        peek_trees() const noexcept;
        /*@}*/

        /** Seed the pseudo-random number generator engine.
         * @param[in] seed Value to seed ForestBase::gen with. */
        void seed_gen(const size_t seed);

        /** Plant and grow (train) trees in a random forest using supplied data.
         * @param[in] n_predictor The number of predictors that will be set for
         * every tree in the forest.
         * @param[in] is_ordered Indicators for each predictor whether it is
         * ordered or not.
         * @param[in] forest_parameters A container of TrainingParameters that
         * are passed one-by-one to the tree-training method.
         * @param[in] data Data to train forest with, see literanger::Data class
         * for further details about format.
         * @param[in] case_weights The weight for each observation (row) during
         * training.
         * @param[in] seed The seed for the pseudo-random number generator
         * engine.
         * @param[in] n_thread Number of threads to split trees across during
         * growth (training) and prediction (out-of-bag error).
         * @param[in] compute_oob_error Indicate whether to estimate the
         * out-of-bag error during training or not.
         * @param[in] user_interrupt An operator that checks for user interrupt.
         * @param[out] oob_error The value of the out-of-bag error if requested.
         * @param[out] print_out A toggle-able printer for outputting progress
         * during training and prediction. */
        virtual void plant(
            const size_t n_predictor,
            const bool_vector_ptr is_ordered,
            const std::vector<TrainingParameters> & forest_parameters,
            const std::shared_ptr<const Data> data,
            const cdbl_vector_ptr case_weights,
            const size_t seed,
            const size_t n_thread,
            const bool compute_oob_error,
            const interruptor & user_interrupt,
            double & oob_error,
            toggle_print & print_out
        ) = 0;


        /** @name Enable cereal for ForestBase. */
        /**@{*/
        template <typename archive_type>
        void serialize(archive_type & archive);
        /**@}*/


    protected:

        /** Construct a random forest object.
         * @param[in] save_memory Indicate whether to aggressively release
         * memory and omit building predictor index. */
        ForestBase(const bool save_memory);

        /** @copydoc ForestBase::ForestBase(bool)
         * @param[in] n_predictor The number of predictors that will be set for
         * every tree in the forest.
         * @param[in] is_ordered Indicators for each predictor whether it is
         * ordered or not.
         * @param[in] trees The (constructed) trees for the random forest. */
        ForestBase(const bool save_memory,
                   const size_t n_predictor,
                   const bool_vector_ptr is_ordered,
                   std::vector<std::unique_ptr<TreeBase>> && trees);

        /** Show the proportion of completed events in a particular phase.
         * @param[in] operation A suffix string that describes the current
         * phase or process (e.g. "Growing trees").
         * @param[in] max_events The total number of events in the process.
         * @param[in] n_thread Number of threads in use.
         * @param[in] user_interrupt An operator that checks for user interrupt.
         * @param[out] print_out A toggle-able printer for outputting progress
         * during training and prediction. */
        void show_progress(std::string operation, const size_t max_events,
                           const size_t n_thread,
                           const interruptor & user_interrupt,
                           toggle_print & print_out);


        /** Aggressively release resources and do not construct predictor
         * index. */
        const bool save_memory;

        /** Number of predictors in the random forest model; zero when forest
         * not yet trained. */
        size_t n_predictor;

        /** Indicators for each predictor whether it is ordered; points to
         * nullptr when forest not yet trained. */
        bool_vector_ptr is_ordered;

        /** Pseudo-random number generator for bootstrapping and also for
         * seeding each tree's own pseudo-rng during the growth (training)
         * phase. */
        std::mt19937_64 gen;

        /** Count of the completed events in a 'queue', e.g. the number of
         * trees currently grown. */
        size_t event_count;

        /** Indicator of whether a 'queue' has been interrupted. */
        bool interrupted;

        /** Mutex for updating event_count or interrupted members */
        std::mutex mutex;

        /** Condition variable for the progress report loop. */
        std::condition_variable condition_variable;

        /** Intervals (usually trees) of work to perform in each thread. */
        count_vector work_intervals;

        /** A container for the trees in the forest. */
        std::vector<std::unique_ptr<TreeBase>> trees;


};


/** Make a unique ForestBase resource by forwarding arguments.
 * @param[in] args Arguments forwarded to a random forest constructor
 * @returns A unique pointer to the constructed random forest.
 * @tparam T The derived random forest type to construct.
 * @tparam ArgsT The argument types of a constructor for the derived type.
 * @see ForestBase::ForestBase(std::vector<TreeParameters>,bool)
 */
template <typename T, typename... ArgsT>
std::unique_ptr<ForestBase> make_forest(ArgsT &&... args);


} /* namespace literanger */


#endif /* LITERANGER_FOREST_BASE_DECL_H */


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
#ifndef LITERANGER_FOREST_BASE_DEFN_H
#define LITERANGER_FOREST_BASE_DEFN_H

/* class declaration */
#include "literanger/ForestBase.decl.h"

/* standard library headers */
#include <chrono>
#include <cmath>
#include <ctime>   /* localtime */
/* put_time is not available in GCC < 5*/
#if !defined(__GNUC__) || __GNUC__ >= 5
  #include <iomanip>
#endif /* !defined(__GNU_C__) || __GNUC__ >= 5 */
#include <iterator>
#include <ostream> /* endl */
#include <sstream>
#include <stdexcept>
#include <utility>

/* for serializing tree_parameters */
#include "cereal/types/memory.hpp"
#include "cereal/types/vector.hpp"


namespace literanger {

/* construction call definition(s) */

template <typename T, typename... ArgsT>
std::unique_ptr<ForestBase> make_forest(ArgsT &&... args) {
    return std::unique_ptr<ForestBase>(new T(std::forward<ArgsT>(args)...));
}


inline ForestBase::ForestBase(const bool save_memory) :
    save_memory(save_memory), n_predictor(0), is_ordered(nullptr)
{ }


inline ForestBase::ForestBase(
    const bool save_memory, const size_t n_predictor,
    bool_vector_ptr is_ordered,
    std::vector<std::unique_ptr<TreeBase>> && trees
) :
    save_memory(save_memory), n_predictor(n_predictor), is_ordered(is_ordered),
    trees(std::move(trees))
{ }


/* member definitions (non-interface) */

inline size_t ForestBase::size() const noexcept { return trees.size(); }


inline size_t ForestBase::get_n_predictor() const noexcept {
    return n_predictor;
}


inline ForestBase::cbool_vector_ptr
ForestBase::get_is_ordered() const noexcept {
    return is_ordered;
}


inline const std::vector<std::unique_ptr<TreeBase>> &
ForestBase::peek_trees() const noexcept {
    return trees;
}


inline void ForestBase::seed_gen(const size_t seed) {
    if (seed == 0) {
        std::random_device random_device;
        gen.seed(random_device());
    } else {
        gen.seed(seed);
    }
}


template <typename archive_type>
void ForestBase::serialize(archive_type & archive) {
    archive(save_memory, n_predictor, is_ordered, trees);
}


inline void ForestBase::show_progress(std::string operation,
                                      const size_t max_events,
                                      const size_t n_thread,
                                      const interruptor & user_interrupt,
                                      toggle_print & print_out) {

    using std::chrono::steady_clock;
    using std::chrono::system_clock;
    using std::chrono::duration_cast;
    using std::chrono::seconds;

    steady_clock::time_point t_start = steady_clock::now();
    steady_clock::time_point t_last = steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex);

    while (event_count < max_events && !(interrupted |= user_interrupt())) {

      /* Release lock until we receive a notification (i.e. an item of work is
       * done) or an error has occured. */
        condition_variable.wait(lock);

        seconds t_elapsed = duration_cast<seconds>(steady_clock::now() - t_last);

        if (event_count > 0 && t_elapsed.count() > STATUS_INTERVAL) {
            const double proportion = (double)event_count / (double)max_events;
            const auto t_remain = duration_cast<seconds>(
                (steady_clock::now() - t_start) * (1 / proportion - 1)
            );
          /* Kludge: Use the system clock to represent the remaining time in
           * HH:MM:SS format using std::put_time */
            const auto abs_t = system_clock::to_time_t(
                                   system_clock::time_point(t_remain)
                               );
            std::stringstream out_fmt;
          #if !defined(__GNUC__) || __GNUC__ >= 5
            out_fmt << operation <<  " Progress: " <<
                std::to_string(std::round(100 * proportion)) <<
                "%. Estimated remaining time: " <<
                std::put_time(std::localtime(&abs_t), "%H:%M:%S") << "." <<
                std::endl;
          #else
          /* put_time not defined - use strftime equivalent */
            char fmt_time[32];
            bool result = std::strftime(fmt_time, sizeof(fmt_time), "%H:%M:%S",
                                        std::localtime(&abs_t));
            if (result == 0)
                out_fmt << operation <<  " Progress: " <<
                    std::to_string(std::round(100 * proportion)) <<
                    "%. Estimated remaining time: " << fmt_time << "." <<
                    std::endl;
          #endif
            print_out((out_fmt.str()).c_str());
            t_last = steady_clock::now();

        }

    }

}


} /* namespace literanger */


#endif /* LITERANGER_FOREST_BASE_DEFN_H */


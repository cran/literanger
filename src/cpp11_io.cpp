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

/* general literanger headers */
#include "literanger/globals.h"
/* required literanger class headers */
#include "literanger/ForestClassification.h"
#include "literanger/ForestRegression.h"

/* literanger R package headers */
// #include "cpp11_utility.h"


[[cpp11::register]]
cpp11::raws cpp11_serialize(cpp11::list object, const bool verbose) {

    using namespace literanger;

    toggle_print print_out { verbose, Rprintf };

    print_out("Entered serialization wrapper");

    std::stringstream ss;
    {
        std::unique_ptr<ForestBase> forest_ptr {
            cpp11::as_cpp<cpp11::external_pointer<ForestBase>>(
                object["cpp11_ptr"]
            ).get(),
        };
        cereal::BinaryOutputArchive oarchive(ss); // Create an output archive
        oarchive(forest_ptr);
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

    toggle_print print_out { verbose, Rprintf };

    print_out("Entered deserialization wrapper");

    std::stringstream ss;
    std::copy(object.cbegin(), object.cend(), std::ostream_iterator<char>(ss));

    std::unique_ptr<ForestBase> forest_ptr;

    {
        cereal::BinaryInputArchive iarchive(ss); // Read from input archive
        iarchive(forest_ptr);
    }
    cpp11::writable::list result;

    result.push_back({
        "cpp11_ptr"_nm = cpp11::external_pointer<ForestBase>(
            forest_ptr.release()
        )
    });

    return result;

}


# ------------------------------------------------------------------------------
# This file is part of 'literanger'. literanger was adapted from the 'ranger'
# package for R statistical software. ranger was authored by Marvin N Wright
# with the GNU General Public License version 3. The adaptation was performed by
# stephematician in 2023. literanger carries the same license, terms, and
# permissions as ranger.
#
# literanger is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# literanger is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with literanger. If not, see <https://www.gnu.org/licenses/>.
#
# Written by:
#
#   stephematician
#   stephematician@gmail.com
#   Australia
# ------------------------------------------------------------------------------

#' \pkg{literanger}: Fast Serializable Random Forests based on 'ranger'.
#'
#' 'literanger' is an adaptation of the 'ranger' R package for training and
#' predicting from random forest models within multiple imputation algorithms.
#' ranger is a fast implementation of random forests (Breiman, 2001) or
#' recursive partitioning, particularly suited for high dimensional data
#' (Wright et al, 2017a). literanger enables random forests to be embedded in
#' the fully conditional specification framework for multiple imputation known
#' as "Multiple Imputation via Chained Equations" (Van Buuren 2007).
#'
#' literanger trains classification and regression forests. The trained forest
#' retains information about the in-bag responses in each terminal node, thus
#' facilitating computationally efficient prediction within multiple imputation
#' with random forests proposed by Doove et al (2014). This multiple imputation
#' algorithm has better predictive distribution properties than competing
#' approaches which use predictive mean matching. Alternatively, the usual
#' bagged prediction may be used as in the imputation algorithm called
#' 'missForest' (Stekhoven et al, 2014).
#'
#' Efficient serialization, i.e. reading and writing, of a trained random forest
#' is provided via the cereal library <https://uscilab.github.io/cereal/>.
#'
#' Classification and regression forests are implemented as in the original
#' Random Forest (Breiman, 2001) or using extremely randomized trees (Geurts et
#' al, 2006). 'data.frame', 'matrix', and sparse matrices ('dgCmatrix') are
#' supported.
#'
#' Split selection may be based on improvement in metrics such as the variance,
#' Gini impurity, beta log-likelihood (Weinhold et al, 2019), Hellinger distance
#' (Cieslak et al, 2012) or maximally selected rank statistics (Wright et
#' al, 2017b).
#'
#' See <https://github.com/imbs-hl/ranger> for the development version of ranger
#' or <https://gitlab.com/stephematician/literanger> for development version of
#' this package.
#'
#' For alternative approaches to multiple imputation that employ random forests,
#' see 'missRanger' <https://cran.r-project.org/package=missRanger> and
#' 'miceRanger' <https://cran.r-project.org/package=miceRanger>, which use
#' predictive mean matching combined with the original 'ranger' algorithm.
#'
#' This package was adapted from the 'ranger' package for R Statistical
#' Software. The C++ core is provided under the same license terms as the
#' original C++ core in the 'ranger' package, namely the MIT license
#' <https://www.r-project.org/Licenses/MIT>. The wrappers in this package around
#' the core are licensed under the same terms of the corresponding components in
#' the 'ranger' R package, namely the GPL3 license
#' <https://www.r-project.org/Licenses/GPL-3>, <https://www.gnu.org/licenses/>.
#'
#'
#' # License
#'
#' 'literanger' was adapted from the 'ranger' package for R statistical
#' software. ranger was authored by Marvin N. Wright with the GNU General Public
#' License version 3 for the R package (interface), while the C++ core of ranger
#' has the MIT license. The adaptation was performed by stephematician in 2023.
#' literanger carries the same license, terms, and permissions as ranger,
#' including the GNU General Public License 3 for the R package interface, and
#' the MIT license for the C++ core.
#'
#' License statement for C++ core of ranger:
#'
#' ```
#' MIT License
#'
#' Copyright (c) [2014-2018] [Marvin N. Wright]
#'
#' Permission is hereby granted, free of charge, to any person obtaining a copy
#' of this software and associated documentation files (the “Software”), to deal
#' in the Software without restriction, including without limitation the rights
#' to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#' copies of the Software, and to permit persons to whom the Software is
#' furnished to do so, subject to the following conditions:
#'
#' The above copyright notice and this permission notice shall be included in
#' all copies or substantial portions of the Software.
#'
#' THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#' IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#' FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#'  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#' LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#' OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#' SOFTWARE.
#' ```
#'
#' License statement for the ranger R package interface:
#'
#' ```
#' Ranger is free software: you can redistribute it and/or modify it under
#' the terms of the GNU General Public License as published by the Free
#' Software Foundation, either version 3 of the License, or (at your option)
#' any later version.
#'
#' Ranger is distributed in the hope that it will be useful, but WITHOUT ANY
#' WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#' FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#' details.
#'
#' You should have received a copy of the GNU General Public License along
#' with Ranger. If not, see <https://www.gnu.org/licenses/>.
#'
#' Written by:
#' Marvin N. Wright
#' Institut für Medizinische Biometrie und Statistik
#' Universität zu Lübeck
#' Ratzeburger Allee 160
#' 23562 Lübeck
#' https://www.imbs.uni-luebeck.de/institut
#' ```
#'
#' @references
#'
#' -   Breiman, L. (2001). Random forests. _Machine Learning_, 45, 5-32.
#'     \doi{10.1023/A:1010933404324}.
#' -   Cieslak, D. A., Hoens, T. R., Chawla, N. V., & Kegelmeyer, W. P. (2012).
#'     Hellinger distance decision trees are robust and skew-insensitive. _Data
#'     Mining and Knowledge Discovery_, 24, 136-158.
#'     \doi{10.1007/s10618-011-0222-1}.
#' -   Doove, L. L., Van Buuren, S., & Dusseldorp, E. (2014). Recursive
#'     partitioning for missing data imputation in the presence of interaction
#'     effects. _Computational Statistics & Data Analysis_, 72, 92-104.
#'     \doi{10.1016/j.csda.2013.10.025}.
#' -   Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees.
#'     _Machine Learning_, 63, 3-42. \doi{10.1007/s10994-006-6226-1}.
#' -   Stekhoven, D.J. and Buehlmann, P. (2012). MissForest--non-parametric
#'     missing value imputation for mixed-type data. _Bioinformatics_, 28(1),
#'     112-118. \doi{10.1093/bioinformatics/btr597}.
#' -   Van Buuren, S. (2007). Multiple imputation of discrete and continuous
#'     data by fully conditional specification. _Statistical Methods in Medical
#'     Research_, 16(3), 219-242. \doi{10.1177/0962280206074463}.
#' -   Weinhold, L., Schmid, M., Wright, M. N., & Berger, M. (2019). A random
#'     forest approach for modeling bounded outcomes. _arXiv preprint_,
#'     arXiv:1901.06211. \doi{10.48550/arXiv.1901.06211}.
#' -   Wright, M. N., & Ziegler, A. (2017a). ranger: A Fast Implementation of
#'     Random Forests for High Dimensional Data in C++ and R. _Journal of
#'     Statistical Software_, 77, 1-17. \doi{10.18637/jss.v077.i01}.
#' -   Wright, M. N., Dankowski, T., & Ziegler, A. (2017b). Unbiased split
#'     variable selection for random survival forests using maximally selected
#'     rank statistics. _Statistics in medicine_, 36(8), 1272-1284.
#'     \doi{10.1002/sim.7212}.
#'
#' @author stephematician <stephematician@gmail.com>, Marvin N Wright (original
#' 'ranger' package)
#'
#' @useDynLib literanger, .registration = TRUE
#'
#' @keywords internal
#' @docType package
#' @md
"_PACKAGE"

# A recommended practise; unload this package's dynamic libraries,
# see https://r-pkgs.had.co.nz/src.html
.onUnload <- function (lib_path)
    library.dynam.unload("literanger", lib_path)


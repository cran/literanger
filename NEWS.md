# Changelog - literanger

## [0.1.1](https://gitlab.com/stephematician/literanger/-/tags/v0.1.1) - 2024-09-13

Performance improved. R interface and C++ core have been separated.

### Changed

-   Set `Depends` to R >= 3.6.0.
-   Training speed was increased by ~25% by reducing memory allocations
    [`19e7c475`](https://gitlab.com/stephematician/literanger/-/commit/19e7c475)
    and inlining access to data.
    [`233063e0`](https://gitlab.com/stephematician/literanger/-/commit/233063e0).
-   Source code underwent a minor re-organisation to separate the R-specific
    components from the C++ library.

###  Added

-   `DataVector` class to read, write, and pass data without R.


## [0.1.0](https://gitlab.com/stephematician/literanger/-/tags/v0.1.0) - 2024-09-03

_New feature_! literanger can now serialize trained random forests using
[cereal](https://uscilab.github.io/cereal/).

The project has been moved to GitLab:
<https://gitlab.com/stephematician/literanger>.

### Changed

-   The value-type returned by `predict` now matches the response type in
    training [`ea67c83e`](https://gitlab.com/stephematician/literanger/-/commit/ea67c83e)
-   Bump [cpp11](https://cpp11.r-lib.org/) to 0.4.7.

### Added

-   Functions `read_literanger` and `write_literanger` for serialization.

### Fixed

-   Fixed bug in implementation of always-selected candidates for splitting,
    e.g. the `names_of_always_draw` argument [`6d31d7f3`](https://gitlab.com/stephematician/literanger/-/commit/6d31d7f3)
-   Minor performance tweak [`9a3b639a`](https://gitlab.com/stephematician/literanger/-/commit/9a3b639a)
    in particular for 'maxstat' [`37580d9b`](https://gitlab.com/stephematician/literanger/-/commit/37580d9b)


## [0.0.2](https://gitlab.com/stephematician/literanger/-/tags/v0.0.2) - 2023-07-11

Update to pass CRAN's ASAN check

### Changed

-   Improve performance of node splitting ([`d3f6424`](https://gitlab.com/stephematician/literanger/-/commit/d3f64245))

### Added

-   Add re-entrant log gamma to speed up beta splitting rule
    ([`d7f058d`](https://gitlab.com/stephematician/literanger/-/commit/d7f058dd))
-   Minor fixes to documentation ([`91b6c6d`](https://gitlab.com/stephematician/literanger/-/commit/91b6c6d),
    [`0f62d02`](https://gitlab.com/stephematician/literanger/-/commit/0f62d027))

### Fixed

-   Fix potential illegal access and incorrect unweighted sampling without
    replacement ([`b6df5d9`](https://gitlab.com/stephematician/literanger/-/commit/b6df5d9))


## [0.0.1](https://gitlab.com/stephematician/literanger/-/tags/v0.0.1) - 2023-06-25

_First release_

A refactoring and adaptation of the ranger package
<https://github.com/imbs-hl/ranger> for random forests. Has faster prediction
mode intended for embedding into the multiple imputation algorithm proposed by
Doove et al in:

Doove, L. L., Van Buuren, S., & Dusseldorp, E. (2014). Recursive partitioning
for missing data imputation in the presence of interaction effects.
_Computational statistics & data analysis_, 72, 92-104.

### Added

-   Fit classification and regression trees
-   Prediction via most frequent value or mean
-   Get predictions as terminal node identifiers in each tree or as a random
    draw from inbag values in a random tree


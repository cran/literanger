literanger: Fast serializable random forests for multiple imputation
====================================================================

[![R CMD check status](https://gitlab.com/stephematician/literanger/badges/main/pipeline.svg?job=check_r_package-job-bash&key_text=R+CMD+Check&key_width=90)](https://gitlab.com/stephematician/literanger/-/commits/main)
[![coverage report](https://gitlab.com/stephematician/literanger/badges/main/coverage.svg)](https://gitlab.com/stephematician/literanger/-/commits/main)
[![Common Changelog](https://common-changelog.org/badge.svg)](https://common-changelog.org)

by _stephematician_

`literanger` is an adaptation of the [`ranger`][ranger_cran] R package for
training and predicting from random forest models within multiple imputation
algorithms. `ranger` is a fast implementation of random forests
([Breiman, 2001][breiman2001_doi]) or recursive partitioning, particularly
suited for high dimensional data ([Wright et al, 2017][wright2017_doi]).
`literanger` redesigned the `ranger` interface to achieve faster prediction, and
is now available as a backend for random forests within 'Multiple Imputation
via Chained Equations' ([Van Buuren, 2007][vanbuuren2007_doi]) in the
R package [`mice`][mice_cran].

Efficient serialization, i.e. reading and writing, of a trained random forest is
provided via the [cereal][cereal_url] library.

<!-- A multiple imputation algorithm using this package is under development: called
[`mimputest`][mimputest_gitlab].
-->

[cereal_url]: https://uscilab.github.io/cereal/
[mice_cran]: https://cran.r-project.org/package=mice
[ranger_cran]: https://cran.r-project.org/package=ranger
[mimputest_gitlab]: https://gitlab.com/stephematician/mimputest


## Example

```r
require(literanger)

train_idx <- sample(nrow(iris), 2/3 * nrow(iris))
iris_train <- iris[ train_idx, ]
iris_test  <- iris[-train_idx, ]
rf_iris <- train(data=iris_train, response_name="Species")
pred_iris_bagged <- predict(rf_iris, newdata=iris_test,
                            prediction_type="bagged")
pred_iris_inbag  <- predict(rf_iris, newdata=iris_test,
                            prediction_type="inbag")
# compare bagged vs actual test values
table(iris_test$Species, pred_iris_bagged$values)
# compare bagged prediction vs in-bag draw
table(pred_iris_bagged$values, pred_iris_inbag$values)
```

Literanger supports reading/writing random forests (serialization). We can
save `rf_iris` above using the function call:

```r
write_literanger(rf_iris, "rf_iris.literanger")
```

In a new R session, we can read the random forest object in and predict for
a new test set:

```r
test_idx <- sample(nrow(iris), 1/3 * nrow(iris))
iris_test  <- iris[test_idx, ]
rf_iris_copy <- read_literanger("rf_iris.literanger")
table(iris_test$Specis, predict(rf_iris_copy, newdata=iris_test)$values)
```

_Experimental feature_: As of v0.2.0, forests can be merged, naively, by copying
the trees from two forests to a new one:

```r
rf <- replicate(
    2, train(data=iris, response_name="Sepal.Length"), simplify=FALSE
)
rf_merged <- merge(rf[[1]], rf[[2]])
```


## Installation

The release can be installed via:

```r
install.packages('literanger')
```

The development version can be installed using [`remotes`][remotes_cran]:

```r
remotes::install_gitlab('stephematician/literanger')
```

[literanger_cran]: https://cran.r-project.org/package=literanger
[remotes_cran]: https://cran.r-project.org/package=remotes


## Technical details

A minor variation on `mice`'s use of random forests is available; each
prediction is drawn from in-bag samples from a random tree - thus the
computational effort is constant with respect to the size of the forest (number
of trees) compared to the original implementation in `mice`.

The interface of `ranger` was redesigned such that the trained forest
object can be recycled, and the data for training and prediction are passed
without (unnecessary) copies, see `ranger`
[issue #304](https://github.com/imbs-hl/ranger/issues/304).


## To-do

Non-exhaustive:

-   implement variable importance measures;
-   probability and survival forests.


## References

Breiman, L. (2001). Random forests. _Machine learning_, 45, pp. 5-32.
[doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).

Doove, L.L., Van Buuren, S. and Dusseldorp, E., 2014. Recursive partitioning for
missing data imputation in the presence of interaction effects. _Computational
Statistics & Data Analysis_, 72, pp. 92-104.
[doi:10.1016/j.csda.2013.10.025](https://doi.org/10.1016/j.csda.2013.10.025).

Grant, W. S., and Voorhies, R., 2017. _cereal - A C++11 library for
serialization_. [https://uscilab.github.io/cereal][cereal_url].

Van Buuren, S. 2007. Multiple imputation of discrete and continuous  data by
fully conditional specification. _Statistical Methods in Medical Research_,
16(3), pp. 219-242.
[doi:10.1177/0962280206074463](https://doi.org/10.1177/0962280206074463).

Wright, M. N. and Ziegler, A., 2017. ranger: A fast implementation of random
forests for high dimensional data in C++ and R. _Journal of Statistical
Software_, 77(i01), pp. 1-17.
[doi:10.18637/jss.v077.i01](https://doi.org/10.18637/jss.v077.i01).

[breiman2001_doi]: https://doi.org/10.1023/A:1010933404324
[doove2014_doi]: https://doi.org/10.1016/j.csda.2013.10.025
[vanbuuren2007_doi]: https://doi.org/10.1177/0962280206074463
[wright2017_doi]: https://doi.org/10.18637/jss.v077.i01


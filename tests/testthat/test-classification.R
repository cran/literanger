
iris_mat <- data.matrix(iris)
set.seed(42)
rf_class_df <- train(data=iris, response_name="Species")
set.seed(42)
rf_ordered_df <- suppressWarnings(train(
    data=modifyList(iris, list(Species=as.ordered(iris$Species))),
    response_name="Species"
))
set.seed(42)
rf_class_mat <- train(data=iris_mat, response_name="Species",
                      classification=TRUE)

test_that("tree type is 'classification'", {
    expect_equal(rf_class_df$tree_type, "classification")
    expect_equal(rf_ordered_df$tree_type, "classification")
    expect_equal(rf_class_mat$tree_type, "classification")
})

test_that("trained forest object has 'response' item", {
    expect_true(hasName(rf_class_df, "response"))
    expect_true(hasName(rf_ordered_df, "response"))
    expect_true(hasName(rf_class_mat, "response"))
})

test_that("trained forest object has 'training' item", {
    expect_true(hasName(rf_class_df, "training"))
    expect_true(hasName(rf_ordered_df, "training"))
    expect_true(hasName(rf_class_mat, "training"))
})

test_that("can use 'save_memory' option when training", {
    expect_silent(
        rf <- train(data=iris, response_name="Species", save_memory=TRUE)
    )
})

test_that("get error when all class-specific sample fraction is zero", {
    expect_error(
        train(data=iris, response_name="Species", sample_fraction=rep(0, 3)),
        paste("'sample_fraction' too small (results in zero samples)."),
        fixed=TRUE
    )
})

test_that("get error when using class-specific sample fraction and (case) weights", {
    expect_error(
        train(data=iris, response_name="Species",
              sample_fraction=c(0.2, 0.3, 0.4),
              case_weights=rbinom(nrow(iris), 1, 0.5)),
        paste("Combination of 'case_weights' argument and class-wise sampling",
              "not supported."),
        fixed=TRUE
    )
})

test_that("get error when 'sample_fraction' is wrong size", {
    expect_error(
        train(data=iris, response_name="Species", sample_fraction=c(0.1, 0.2)),
        paste("Invalid value for 'sample_fraction'. Expecting 3 values,",
              "provided 2."),
        fixed=TRUE
    )
})

test_that("get error when element of 'sample_fraction' outside [0,1]", {
    expect_error(
        train(data=iris, response_name="Species",
              sample_fraction=c(0.1, 1.1, 0.3)),
        paste("Invalid value for 'sample_fraction'. Please give a value in",
              "(0,1] or a vector of values in [0,1]."),
        fixed=TRUE
    )
    expect_error(
        train(data=iris, response_name="Species",
              sample_fraction = c(-3, 0.5, 0.3)),
        paste("Invalid value for 'sample_fraction'. Please give a value in",
              "(0,1] or a vector of values in [0,1]."),
        fixed=TRUE
    )
})

test_that("get error when not enough samples for non-replacement sampling", {
    expect_error(
        train(data=iris, response_name="Species",
              sample_fraction=c(0.2, 0.3, 0.4), replace=FALSE),
        paste("Not enough samples in class virginica; available: 50,",
              "requested: 60."),
        fixed=TRUE
    )
})

test_that("can use class-specific weights when sampling with replacement", {
    expect_silent(train(data=iris, response_name="Species",
                        sample_fraction=c(0.2, 0.3, 0.4), replace=TRUE))
})

test_that("can predict a single new observation", {
    expect_silent(pred_df <- predict(rf_class_df, newdata=head(iris, 1)))
    expect_equal(pred_df$values, iris[1,"Species"])
    expect_silent(pred_ordered_df <- predict(rf_ordered_df,
                                             newdata=head(iris, 1)))
    expect_equal(pred_ordered_df$values, as.ordered(iris[,"Species"])[1])
    expect_silent(pred_mat <- predict(rf_class_mat,
                                      newdata=iris_mat[1,,drop=FALSE]))
    expect_equal(pred_mat$values, unname(iris_mat[1,"Species"]))
})

test_that("prediction has acceptable accuracy", {
    pred_df <- predict(rf_class_df, newdata=iris)
    expect_gt(mean(iris$Species == pred_df$values), 0.9)
    pred_ordered_df <- predict(rf_ordered_df, newdata=iris)
    expect_gt(mean(as.ordered(iris$Species) == pred_ordered_df$values), 0.9)
    pred_mat <- predict(rf_class_mat, newdata=iris_mat)
    expect_gt(mean(iris_mat[,'Species'] == pred_mat$values), 0.9)
})

test_that("value-type for predicted factor is a factor", {
    expect_silent(pred <- predict(rf_class_df, newdata=iris))
    expect_is(pred$values, "factor")
    expect_null(dim(pred$values))
})

test_that("value-type for predicted ordered factor is ordered", {
    expect_silent(pred <- predict(rf_ordered_df, newdata=iris))
    expect_is(pred$values, "ordered")
    expect_null(dim(pred$values))
})

test_that("value-type for predicted numeric is a numeric", {
    pred <- predict(rf_class_mat, newdata=iris)
    expect_is(pred$values, "numeric")
    expect_null(dim(pred$values))
})

test_that("value-type for logical response is preserved", {
    dat <- iris
    dat[["Species"]] <- dat[["Species"]] == "setosa"
    rf <- train(data=dat, response_name="Species")
    pred <- predict(rf, newdata=iris)
    expect_is(pred$values, "logical")
    expect_null(dim(pred$values))
})

test_that("value-type for character response is preserved", {
    dat <- modifyList(iris, list(Species=as.character(iris$Species)))
    expect_warning(rf <- train(data=dat, response_name="Species"),
                   "Converting character response to factor")
    pred <- predict(rf, newdata=iris)
    expect_is(pred$values, "character")
    expect_null(dim(pred$values))
})

test_that("get error when 'split_rule' is 'variance'", {
    expect_error(train(data=iris, response_name="Species",
                        split_rule="variance"))
})

test_that("default split metric is 'gini'", {
    set.seed(42)
    rf_class_gini <- train(data=iris, response_name="Species", split_rule="gini")

    expect_equal(rf_class_df$training$split_rule, "gini")
    expect_equal(rf_class_mat$training$split_rule, "gini")
    expect_equal(rf_class_gini$training$split_rule, "gini")
    expect_equal(rf_class_df$oob_error, rf_class_gini$oob_error)
    expect_equal(rf_class_mat$oob_error, rf_class_gini$oob_error)
})

test_that("can train with numerically near-identical splits", {
    dat <- data.frame(a = factor(1:2),
                      z = c(1.7629414498915687570246291215880773,
                            1.7629414498915689790692340466193854))
    expect_silent(train(data=dat, response_name="a", n_thread=1, n_tree=1))
})

test_that("get warning when unused factor levels in response", {
    expect_warning(
        rf <- train(data=iris[1:100, ], response_name="Species"),
        "Dropped unused factor level(s) in response variable: virginica.",
        fixed=TRUE
    )
    pred <- predict(rf, newdata=iris)
    expect_equal(levels(pred$values), levels(iris$Species))
    expect_equal(sum(is.na(pred$values)), 0)
})



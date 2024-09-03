
test_that("result same with x/y interface, classification", {
    set.seed(300)
    rf <- train(data=iris, response_name="Species")
    pred <- predict(rf, newdata=iris)

    set.seed(300)
    rf_xy <- train(y=iris[, 5], x=iris[, -5])
    pred_xy <- predict(rf, newdata=iris[, -5])

    expect_equal(rf$oob_error, rf_xy$oob_error)
    expect_equal(pred$values, pred_xy$values)
})

test_that("result same with x/y interface, regression", {
    set.seed(300)
    rf <- train(data=iris, response_name="Sepal.Length")
    pred <- predict(rf, newdata=iris)

    set.seed(300)
    expect_silent(rf_xy <- train(y=iris[, 1], x=iris[, -1]))
    expect_silent(pred_xy <- predict(rf_xy, newdata=iris[, -1]))

    expect_equal(rf$oob_error, rf_xy$oob_error)
    expect_equal(pred$values, pred_xy$values)
})

test_that("result same with x/y interface, ordered", {
    set.seed(300)
    rf <- train(data=modifyList(iris, list(Species=as.ordered(iris$Species))),
                response_name="Species")
    pred <- predict(rf, newdata=iris)

    set.seed(300)
    expect_silent(rf_xy <- train(y=as.ordered(iris[, 5]), x=iris[, -5]))
    expect_silent(pred_xy <- predict(rf_xy, newdata=iris[, -5]))

    expect_equal(rf$oob_error, rf_xy$oob_error)
    expect_equal(pred$values, pred_xy$values)
})

test_that("result same with x/y interface, logical", {
    set.seed(300)
    rf <- train(data=modifyList(iris, list(Species=iris$Species == 'setosa')),
                response_name="Species")
    pred <- predict(rf, newdata=iris)

    set.seed(300)
    expect_silent(rf_xy <- train(y=iris[, 5] == 'setosa', x=iris[, -5]))
    expect_silent(pred_xy <- predict(rf_xy, newdata=iris[, -5]))

    expect_equal(rf$oob_error, rf_xy$oob_error)
    expect_equal(pred$values, pred_xy$values)
})

test_that("result same with x/y interface, character", {
    set.seed(300)
    expect_warning(
        rf <- train(data=modifyList(iris, list(Species=as.character(iris$Species))),
                    response_name="Species"),
        "Converting character response to factor"
    )
    pred <- predict(rf, newdata=iris)

    set.seed(300)
    expect_warning(rf_xy <- train(y=as.character(iris[, 5]), x=iris[, -5]),
                   "Converting character response to factor")
    expect_silent(pred_xy <- predict(rf_xy, newdata=iris[, -5]))

    expect_equal(rf$oob_error, rf_xy$oob_error)
    expect_equal(pred$values, pred_xy$values)
})

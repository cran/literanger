set.seed(42)
rf_class_df <- train(data=iris, response_name="Species")

set.seed(42)
rf_df <- train(data=iris, response_name="Sepal.Length")


test_that("can read and write a classification forest", {

    file_name <- tempfile()
    on.exit(unlink(file_name))

    expect_no_condition(write_literanger(rf_class_df, file=file_name))
    expect_no_condition(rf_copy <- read_literanger(file=file_name))

    set.seed(123)
    expected_prediction <- predict(rf_class_df, newdata=iris)
    set.seed(123)
    prediction_from_copy <- predict(rf_copy, newdata=iris)

    expect_equal(expected_prediction, prediction_from_copy)

})

test_that("can read and write a regression forest", {

    file_name <- tempfile()
    on.exit(unlink(file_name))

    expect_no_condition(write_literanger(rf_df, file=file_name))
    expect_no_condition(rf_copy <- read_literanger(file=file_name))

    set.seed(123)
    expected_prediction <- predict(rf_df, newdata=iris)
    set.seed(123)
    prediction_from_copy <- predict(rf_copy, newdata=iris)

    expect_equal(expected_prediction, prediction_from_copy)

})


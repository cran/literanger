## Tests for character data

## Initialize random forests
dat <- iris
dat$Test <- paste0("AA", as.character(1:nrow(dat)))

## Tests
test_that("can train and predict if predictors contain character vector", {
    expect_silent(rf <- train(data=dat, response_name="Species"))
    expect_silent(predict(rf, newdata=dat))
})

test_that("can train and predict if response is character vector", {
    expect_warning(rf <- train(data=dat, response_name="Test"),
                   "Converting character response to factor")
    expect_silent(predict(rf, newdata=dat))
    expect_type(predict(rf, newdata=dat)$values, "character")
})


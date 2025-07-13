

test_that("can merge classification forests trained on iris data", {
    set.seed(42L)
    rf <- replicate(
        2L,
        train(data=iris, response_name="Species", classification=TRUE),
        simplify=FALSE
    )
    rf_merged <- merge(rf[[1]], rf[[2]])

    expect_equal(rf_merged$tree_type, rf[[1]]$tree_type)
    expect_equal(rf_merged$n_tree, rf[[1]]$n_tree + rf[[2]]$n_tree)
    expect_identical(rf_merged$predictors, rf[[1]]$predictors)

    pred_nodes <- lapply(rf, predict, newdata=iris, prediction_type="nodes")

    pred_merged_nodes <- predict(rf_merged, newdata=iris,
                                 prediction_type="nodes")
    expect_identical(pred_merged_nodes$nodes,
                     cbind(pred_nodes[[1]]$nodes, pred_nodes[[2]]$nodes))
})

test_that("can merge regression forests trained on iris data", {
    set.seed(42L)
    rf <- replicate(
        2L,
        train(data=iris, response_name="Sepal.Length", classification=FALSE),
        simplify=FALSE
    )
    rf_merged <- merge(rf[[1]], rf[[2]])

    expect_equal(rf_merged$tree_type, rf[[1]]$tree_type)
    expect_equal(rf_merged$n_tree, rf[[1]]$n_tree + rf[[2]]$n_tree)
    expect_identical(rf_merged$predictors, rf[[1]]$predictors)

    pred_nodes <- lapply(rf, predict, newdata=iris, prediction_type="nodes")
    pred_bagged <- lapply(rf, predict, newdata=iris, prediction_type="bagged")

    pred_merged_nodes <- predict(rf_merged, newdata=iris,
                                 prediction_type="nodes")
    pred_merged_bagged <- predict(rf_merged, newdata=iris,
                                  prediction_type="bagged")

    expect_identical(pred_merged_nodes$nodes,
                     cbind(pred_nodes[[1]]$nodes, pred_nodes[[2]]$nodes))
    expect_equal(pred_merged_bagged$values,
                 rowMeans(cbind(pred_bagged[[1]]$values,
                                pred_bagged[[2]]$values)))
})

test_that("can merge classification forests trained on reordered data", {
    set.seed(42L)
  # add types to prompt the correct forest type
    typed_mtcars <- mtcars
    factor_pred <- c("cyl", "vs", "am", "gear", "carb")
    typed_mtcars[factor_pred] <- lapply(typed_mtcars[factor_pred], as.factor)

  # train forest on original data; then re-order and train another forest
    rf <- list()
    rf$orig <- train(data=typed_mtcars, response_name="cyl")
    rrow_j <- rev(seq_len(nrow(typed_mtcars)))
    rcol_j <- rev(seq_len(ncol(typed_mtcars)))
    rf$rev <- train(data=typed_mtcars[rrow_j,rcol_j], response_name="cyl")
  # merge forests
    rf_merged <- merge(rf$orig, rf$rev)

    expect_equal(rf_merged$tree_type, "classification")
    expect_equal(rf_merged$tree_type, rf$orig$tree_type)
    expect_equal(rf_merged$n_tree, rf$orig$n_tree + rf$rev$n_tree)
    expect_identical(rf_merged$predictors, rf$orig$predictors)

    pred_nodes <- lapply(rf, predict, newdata=typed_mtcars,
                         prediction_type="nodes")

    pred_merged_nodes <- predict(rf_merged, newdata=typed_mtcars,
                                 prediction_type="nodes")

    expect_identical(pred_merged_nodes$nodes,
                     cbind(pred_nodes$orig$nodes, pred_nodes$rev$nodes))
})

test_that("can merge regression forests trained on reordered data", {
    set.seed(42L)
  # add types to prompt the correct forest type
    typed_mtcars <- mtcars
    factor_pred <- c("cyl", "vs", "am", "gear", "carb")
    typed_mtcars[factor_pred] <- lapply(typed_mtcars[factor_pred], as.factor)

  # train forest on original data; then re-order and train another forest
    rf <- list()
    rf$orig <- train(data=typed_mtcars, response_name="mpg")
    rrow_j <- rev(seq_len(nrow(typed_mtcars)))
    rcol_j <- rev(seq_len(ncol(typed_mtcars)))
    rf$rev <- train(data=typed_mtcars[rrow_j,rcol_j], response_name="mpg")
  # merge forests
    rf_merged <- merge(rf$orig, rf$rev)

    expect_equal(rf_merged$tree_type, "regression")
    expect_equal(rf_merged$tree_type, rf$orig$tree_type)
    expect_equal(rf_merged$n_tree, rf$orig$n_tree + rf$rev$n_tree)
    expect_identical(rf_merged$predictors, rf$orig$predictors)

    pred_nodes <- lapply(rf, predict, newdata=typed_mtcars,
                         prediction_type="nodes")
    pred_bagged <- lapply(rf, predict, newdata=typed_mtcars,
                          prediction_type="bagged")

    pred_merged_nodes <- predict(rf_merged, newdata=typed_mtcars,
                                 prediction_type="nodes")
    pred_merged_bagged <- predict(rf_merged, newdata=typed_mtcars,
                                  prediction_type="bagged")

    expect_identical(pred_merged_nodes$nodes,
                     cbind(pred_nodes$orig$nodes, pred_nodes$rev$nodes))
    expect_equal(pred_merged_bagged$values,
                 rowMeans(cbind(pred_bagged$orig$values,
                                pred_bagged$rev$values)))
})


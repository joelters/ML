mad <- mad2019[1:250,]
X <- dplyr::select(mad,-Y)
Y <- mad$Y

test_that("Lasso has no error and gives S3 class", {
  expect_no_error(m <- modest(X,Y,"Lasso"))
  expect_s3_class(m, "cv.glmnet")
})

test_that("Ridge has no error and gives S3 class", {
  expect_no_error(m <- modest(X,Y,"Ridge"))
  expect_s3_class(m, "cv.glmnet")
})
#
test_that("RF has no error and gives S3 class", {
  expect_no_error(m <- modest(X,Y,"RF"))
  expect_s3_class(m, "ranger")
})
#
test_that("CIF has no error and gives S4 class", {
  expect_no_error(m <- modest(X,Y,"CIF"))
  expect_s4_class(m, "RandomForest")
})
#
test_that("XGB has no error and gives S3 class", {
  expect_no_error(m <- modest(X,Y,"XGB"))
  expect_s3_class(m, "xgb.Booster")
})

test_that("CB has no error and gives S3 class", {
  expect_no_error(m <- modest(X,Y,"CB"))
  expect_s3_class(m, "catboost.Model")
})
#
test_that("SL has no error and gives S3 class", {
  mad <- mad2019[1:250,]
  X <- dplyr::select(mad,-Y)
  Y <- mad$Y
  expect_no_error(m <- modest(X,Y,"SL",
                              ensemble = c("SL.Lasso","SL.Ridge","SL.RF",
                                           "SL.CIF","SL.XGB","SL.CB")))
  expect_s3_class(m, "SuperLearner")
})



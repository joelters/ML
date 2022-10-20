mad <- mad2019[1:250,]
X <- dplyr::select(mad,-Y)
Y <- mad$Y

test_that("Lasso has no error and gives S3 class", {
  expect_no_error(m <- MLest(X,Y,"Lasso"))
  expect_s3_class(m[[1]], "cv.glmnet")
  expect_length(m[[2]], nrow(X))
})

test_that("Ridge has no error and gives S3 class", {
  expect_no_error(m <- MLest(X,Y,"Ridge"))
  expect_s3_class(m[[1]], "cv.glmnet")
  expect_length(m[[2]], nrow(X))
})
#
test_that("RF has no error and gives S3 class", {
  expect_no_error(m <- MLest(X,Y,"RF"))
  expect_s3_class(m[[1]], "ranger")
  expect_length(m[[2]], nrow(X))
})
#
test_that("CIF has no error and gives S4 class", {
  expect_no_error(m <- MLest(X,Y,"CIF"))
  expect_s4_class(m[[1]], "RandomForest")
  expect_length(m[[2]], nrow(X))
})
#
test_that("XGB has no error and gives S3 class", {
  expect_no_error(m <- MLest(X,Y,"XGB"))
  expect_s3_class(m[[1]], "xgb.Booster")
  expect_length(m[[2]], nrow(X))
})

test_that("CB has no error and gives S3 class", {
  expect_no_error(m <- MLest(X,Y,"CB"))
  expect_s3_class(m[[1]], "catboost.Model")
  expect_length(m[[2]], nrow(X))
})
#
# test_that("SL has no error and gives S3 class", {
#   mad <- mad2019[1:250,]
#   X <- dplyr::select(mad,-Y)
#   Y <- mad$Y
#   expect_no_error(m <- MLest(X,Y,"SL",
#                               ensemble = c("SL.Lasso","SL.Ridge","SL.RF",
#                                            "SL.CIF","SL.XGB","SL.CB")))
#   expect_s3_class(m[[1]], "SuperLearner")
#   expect_length(m[[2]], nrow(X))
# })

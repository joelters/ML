mad <- mad2019[1:250,]
X <- dplyr::select(mad,-Y)
Y <- mad$Y

test_that("Lasso has no error and gives vector of correct length", {
  m <- modest(X,Y,"Lasso")
  expect_no_error(FV1 <- FVest(m,X,Y,X[1:5,],Y[1:5],ML = "Lasso"))
  expect_no_error(FV2 <- FVest(m,X,Y,ML = "Lasso"))
  expect_length(FV1, 5)
  expect_length(FV2, nrow(X))
})

test_that("Ridge has no error and gives vector of correct length", {
  m <- modest(X,Y,"Ridge")
  expect_no_error(FV1 <- FVest(m,X,Y,X[1:5,],Y[1:5],ML = "Ridge"))
  expect_no_error(FV2 <- FVest(m,X,Y,ML = "Ridge"))
  expect_length(FV1, 5)
  expect_length(FV2, nrow(X))
})
#
test_that("RF has no error and gives vector of correct length", {
  m <- modest(X,Y,"RF")
  expect_no_error(FV1 <- FVest(m,X,Y,X[1:5,],Y[1:5],ML = "RF"))
  expect_no_error(FV2 <- FVest(m,X,Y,ML = "RF"))
  expect_length(FV1, 5)
  expect_length(FV2, nrow(X))
})
#
test_that("CIF has no error and gives vector of correct length", {
  m <- modest(X,Y,"CIF")
  expect_no_error(FV1 <- FVest(m,X,Y,X[1:5,],Y[1:5],ML = "CIF"))
  expect_no_error(FV2 <- FVest(m,X,Y,ML = "CIF"))
  expect_length(FV1, 5)
  expect_length(FV2, nrow(X))
})
#
test_that("XGB has no error and gives vector of correct length", {
  m <- modest(X,Y,"XGB")
  expect_no_error(FV1 <- FVest(m,X,Y,X[1:5,],Y[1:5],ML = "XGB"))
  expect_no_error(FV2 <- FVest(m,X,Y,ML = "XGB"))
  expect_length(FV1, 5)
  expect_length(FV2, nrow(X))
})
#
test_that("CB has no error and gives vector of correct length", {
  m <- modest(X,Y,"CB")
  expect_no_error(FV1 <- FVest(m,X,Y,X[1:5,],Y[1:5],ML = "CB"))
  expect_no_error(FV2 <- FVest(m,X,Y,ML = "CB"))
  expect_length(FV1, 5)
  expect_length(FV2, nrow(X))
})
#
test_that("SL has no error and gives vector of correct length", {
  m <- modest(X,Y,"SL",
              ensemble = c("SL.Lasso","SL.Ridge","SL.RF",
                           "SL.CIF","SL.XGB","SL.CB"))
  expect_no_error(FV1 <- FVest(m,X,Y,X[1:5,],Y[1:5],ML = "SL"))
  expect_no_error(FV2 <- FVest(m,X,Y,ML = "SL"))
  expect_length(FV1, 5)
  expect_length(FV2, nrow(X))
})



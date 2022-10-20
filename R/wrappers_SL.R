#WRAPPERS FOR SUPERLEARNER

SL.Lasso <- function (...) {
  SL.glmnet(...)
}



SL.Ridge <- function (...) {
  SL.glmnet(...,alpha = 0)
}

SL.RF <- function (...) {
  SL.ranger(...,mtry = max(floor(ncol(X)/3), 1))
}

SL.CIF <- function (...) {
  SL.cforest(...,controls = cforest_unbiased(mtry = max(floor(ncol(X)/3), 1)))
}

SL.XGB <- function (...) {
  SL.xgboost(...,nrounds = 500)
}

SL.CB <- function (Y, X, newX, family, obsWeights, ...) 
{
  if (is.matrix(X)) {
    X = as_tibble(X)
  }
  fit <- modest(X, Y, ML = "CB", weights = obsWeights)
  if (is.matrix(newX)) {
    newX = as_tibble(newX)
  }
  ifelse2 <- function(A = TRUE, B,C){if(A == TRUE){return(B)} else{return(C)}}
  pred <- catboost.predict(fit, ifelse2(is.null(newX),
                                catboost.load_pool(X,label = rep(0,nrow(X))),
                                catboost.load_pool(newX,label = rep(0,nrow(newX)))))
  fit <- list(object = fit)
  class(fit) <- "SL.CB"
  out <- list(pred = pred, fit = fit)
  return(out)
}

SL.lm2 <- function (Y, X, newX, family, obsWeights, model = TRUE, ...) 
{
  if (is.matrix(X)) {
    X = as.data.frame(X)
  }
  fit <- stats::lm(Y ~ ., data = X, weights = obsWeights, model = model)
  if (is.matrix(newX)) {
    newX = as.data.frame(newX)
  }
  pred <- predict(fit, newdata = newX)
  if (family$family == "binomial") {
    pred = pmin(pmax(pred, 0), 1)
  }
  fit <- list(object = fit, family = family)
  class(fit) <- "SL.lm"
  out <- list(pred = pred, fit = fit)
  return(out)
}

#WRAPPERS FOR SUPERLEARNER

SL.Lasso <- function (Y,X,newX,...) {
  X <- stats::model.matrix(~.,X)
  newX <- stats::model.matrix(~.,newX)
  SuperLearner::SL.glmnet(Y,X,newX,...)
}

SL.Logit_lasso <- function (Y,X,newX,...) {
  X <- stats::model.matrix(~.,X)
  newX <- stats::model.matrix(~.,newX)
  SuperLearner::SL.glmnet(Y,X, family = "binomial", newX,...)
}

SL.Ridge <- function (Y,X,newX,...) {
  X <- stats::model.matrix(~.,X)
  newX <- stats::model.matrix(~.,newX)
  SuperLearner::SL.glmnet(Y,X,newX,alpha = 0,...)
}

SL.RF <- function (Y, X,...) {
  SuperLearner::SL.ranger(Y,X,mtry = max(floor(ncol(X)/3), 1),...)
}

SL.CIF <- function (Y, X,...) {
  SuperLearner::SL.cforest(Y,X,
                           controls = party::cforest_unbiased(mtry = max(floor(ncol(X)/3), 1)),
                           ...)
}

SL.XGB <- function (...) {
  SuperLearner::SL.xgboost(...,nrounds = 200,
                           max.depth = 1,)
}

SL.CB <- function (Y, X, newX, family, obsWeights, ...)
{
  if (is.matrix(X)) {
    X = dplyr::as_tibble(X)
  }
  fit <- modest(X, Y, ML = "CB", weights = obsWeights)
  if (is.matrix(newX)) {
    newX = dplyr::as_tibble(newX)
  }
  ifelse2 <- function(A = TRUE, B,C){if(A == TRUE){return(B)} else{return(C)}}
  pred <- catboost::catboost.predict(fit, ifelse2(is.null(newX),
                                catboost::catboost.load_pool(X,label = rep(0,nrow(X))),
                                catboost::catboost.load_pool(newX,label = rep(0,nrow(newX)))))
  fit <- list(object = fit)
  class(fit) <- "SL.CB"
  out <- list(pred = pred, fit = fit)
  return(out)
}

# SL.lm2 <- function (Y, X, newX, family, obsWeights, model = TRUE, ...)
# {
#   # X <- model.matrix(~.,X)
#   # newX <- model.matrix(~.,newX)
#   fit <- stats::lm(Y ~ ., data = data.frame(cbind(Y = Y,X)),
#                    weights = obsWeights, model = model)
#   pred <- predict(fit, newdata = newX)
#   if (family$family == "binomial") {
#     pred = pmin(pmax(pred, 0), 1)
#   }
#   fit <- list(object = fit, family = family)
#   class(fit) <- "SL.lm"
#   out <- list(pred = pred, fit = fit)
#   return(out)
# }

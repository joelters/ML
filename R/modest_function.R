#' Estimate Machine Learning model
#'
#' `modest` estimates the model for a specified machine learner,
#'  possible options are Lasso, Ridge, Random Forest, Conditional
#' Inference Forest, Extreme Gradient Boosting, Catboosting or any
#' combination of these using the SuperLearner package
#'
#' @param X is a dataframe containing all the features
#' @param Y is a vector containing the label
#' @param ML is a string specifying which machine learner to use
#' @param ensemble is a string vector specifying which learners
#' to combine in the SuperLearner. Only needed if ML = "SL".
#' @param weights is a vector containing survey weights adding up to 1
#' @returns the object that the machine learner package returns
#' @examples
#' X <- dplyr::select(mad2019,-c(Y,weight))
#' Y <- mad2019$Y
#' modest(X,Y,"RF")
#' modest(X,Y,"XGB")
#' modest(stats::model.matrix(X),Y,"Lasso")
#'
#' @details Note that the glmnet package which implements Lasso and Ridge
#' does not handle factor variables (such as the ones in mad2019)
#' hence if X contains factors dummy encoding needs to be used. This
#' can be done by simply using stats::model.matrix(X) instead of X
#' @export
modest <- function(X,
                   Y,
                   ML = c("Lasso","Ridge","RF","CIF","XGB","CB", "SL"),
                   ensemble = c("SL.Lasso","SL.Ridge","SL.RF","SL.CIF","SL.XGB","SL.CB"),
                   weights = NULL){
  ML = match.arg(ML)
  dta <- dplyr::as_tibble(cbind(Y = Y,X))
  X <- dplyr::as_tibble(X)
  if (ML == "SL"){
    #Estimate model
    model <- SuperLearner::SuperLearner(Y, X, SL.library = ensemble, family = gaussian(),
                          cvControl = list(V = 5), obsWeights = weights)
  }

  else if (ML == "Lasso"){
    # XX <- model.matrix(Y ~., dta)
    model <- glmnet::cv.glmnet(as.matrix(X),Y,alpha = 1, weights = weights)
  }

  else if (ML == "Ridge"){
    # XX <- model.matrix(Y ~., dta)
    model <- glmnet::cv.glmnet(as.matrix(X),Y,alpha = 0, weights = weights)
  }

  else if (ML == "RF"){
    model <- ranger::ranger(Y ~ .,
                    data = dta,
                    mtry = max(floor(ncol(X)/3), 1),
                    case.weights = weights,
                    respect.unordered.factors = 'partition')
  }

  else if (ML == "CIF"){
    model <- party::cforest(Y ~ .,
                     data = dta,
                     controls = party::cforest_unbiased(mtry = max(floor(ncol(X)/3), 1)),
                     weights = weights)
  }

  else if (ML == "XGB"){
    xgb_data = xgboost::xgb.DMatrix(data = data.matrix(X), label = Y)
    model <- xgboost::xgboost(data = xgb_data,
                     nrounds = 200,
                     max.depth = 1,
                     verbose = 0,
                     weight = weights)
  }

  else if (ML == "CB"){
    warning("CB is treats all features as categorical in this version")
    CB.data <- catboost::catboost.load_pool(X,
                                  label = Y,
                                  cat_features = c(1:ncol(X)),
                                  weight = weights)
    model <- catboost::catboost.train(CB.data,
                            params = list(iterations = 500,
                                          logging_level = 'Silent'))
  }
  return(model)
}

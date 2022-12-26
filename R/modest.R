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
#' X <- dplyr::select(mad2019,-Y)
#' Y <- mad2019$Y
#' modest(X,Y,"RF")
#' modest(X,Y,"XGB")
#' modest(X,Y,"Lasso")
#' modest(X,Y,"SL",
#' ensemble = c("SL.Lasso","SL.Ridge","SL.RF","SL.CIF","SL.XGB","SL.CB"))
#'
#' @details Note that the glmnet package
#' which implements Lasso and Ridge does not handle factor variables
#' (such as the ones in mad2019), hence for this machine learners,
#' modest turns X into model.matrix(~.,X) which will perform dummy
#' encoding on factor variables.
#' @export
modest <- function(X,
                   Y,
                   ML = c("Lasso","Ridge","RF","CIF","XGB","CB", "SL"),
                   ensemble = c("SL.Lasso","SL.Ridge","SL.RF","SL.CIF","SL.XGB","SL.CB"),
                   weights = NULL){
  ML = match.arg(ML)
  dta <- dplyr::as_tibble(cbind(Y = Y,X))
  if (ML == "SL"){
    if (!requireNamespace("SuperLearner", quietly = TRUE)) {
      stop(
        "Package \"SuperLearner\" must be installed to use this function.",
        call. = FALSE
      )
    }
    #Estimate model
    model <- SuperLearner::SuperLearner(Y, X, SL.library = ensemble,
                                        family = stats::gaussian(),
                                        cvControl = list(V = 5),
                                        obsWeights = weights)
  }

  else if (ML == "Lasso"){
    # XX <- model.matrix(Y ~., dta)
    model <- glmnet::cv.glmnet(stats::model.matrix(~.,X),Y,alpha = 1, weights = weights)
  }

  else if (ML == "Ridge"){
    # XX <- model.matrix(Y ~., dta)
    model <- glmnet::cv.glmnet(stats::model.matrix(~.,X),Y,alpha = 0, weights = weights)
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
    if (!requireNamespace("xgboost", quietly = TRUE)) {
      stop(
        "Package \"xgboost\" must be installed to use this function.",
        call. = FALSE
      )
    }
    xgb_data = xgboost::xgb.DMatrix(data = data.matrix(X), label = Y)
    model <- xgboost::xgboost(data = xgb_data,
                     nrounds = 200,
                     max.depth = 1,
                     verbose = 0,
                     weight = weights)
  }

  else if (ML == "CB"){
    if (!requireNamespace("catboost", quietly = TRUE)) {
      stop(
        "Package \"catboost\" must be installed to use this function.
        https://catboost.ai/en/docs/installation/r-installation-binary-installation",
        call. = FALSE
      )
    }
    # CB.data <- catboost::catboost.load_pool(X,
    #                               label = Y,
    #                               cat_features = c(1:ncol(X)),
    #                               weight = weights)
    CB.data <- catboost::catboost.load_pool(X,
                                            label = Y,
                                            weight = weights)
    model <- catboost::catboost.train(CB.data,
                            params = list(iterations = 500,
                                          logging_level = 'Silent'))
  }
  return(model)
}

#' Estimate machine learning model and optionally give fitted values
#'
#' `MLest` estimates a machine learning model (Lasso, Ridge,
#' Random Forest, Conditional Inference Forest,
#' Extreme Gradient Boosting, Catboosting or any
#' combination of these using the SuperLearner package) and (optionally)
#' returns the predicted fitted values.
#'
#' @param X is a dataframe containing all the features on which the
#' model was estimated
#' @param Y is a vector containing the labels for which the model
#' was estimated
#' @param ML is a string specifying which machine learner to use
#' @param ensemble is a string vector specifying which learners
#' should be used in the SuperLearner
#' @param FVs a logical indicating whether FVs should be computed
#' @param weights survey weights adding up to 1
#' @returns list containing model and fitted values
#' @examples
#' X <- dplyr::select(mad2019,-Y)
#' Y <- mad2019$Y
#' m <- MLest(X, Y, "RF", FVs = TRUE)
#'
#' m <- MLest(X, Y, "XGB", FVs = TRUE)
#'
#' m <- MLest(X,Y,"SL",
#' ensemble = c("SL.Lasso","SL.RF","SL.XGB"),
#' FVs = TRUE)
#'
#'
#' @details See documentation of modest and FVest.
#' @export
MLest <- function(X,
                  Y,
                  ML = c("Lasso","Ridge","RF","CIF","XGB","CB","SL"),
                  ensemble = c("SL.Lasso","SL.Ridge","SL.RF","SL.CIF","SL.XGB","SL.CB"),
                  FVs = TRUE,
                  weights = NULL){
  ML = match.arg(ML)
  X <- dplyr::as_tibble(X)
  if (ML == "SL"){
    #Estimate model
    m <- SuperLearner(Y, X, SL.library = ensemble, family = gaussian(),
                      cvControl = list(V = 5), obsWeights = weights)
    #Fitted values
    if (FVs == TRUE){
      FVs <- m$SL.predict
      return(list("model" = m, "FVs" = FVs, shares = m$coef))
    }
    else{
      return(list("model" = m, shares = m$coef))
    }

  }
  else{
    #Estimate model
    m <- modest(X, Y, ML, weights = weights)
    #Fitted values
    if (FVs == TRUE){
      FVs <- FVest(m, X, Y, X, Y, ML)
      return(list("model" = m, "FVs" = FVs))
    }
    else{
      return(list("model" = m))
    }
  }
}

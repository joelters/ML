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

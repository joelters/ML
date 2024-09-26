#' Perform CV on list of machine learners
#'
#' `MLcv` computes cross-validated RMSE for a list including up to
#' Lasso, Ridge,
#' Random Forest, Conditional Inference Forest, Logit lasso,
#' Extreme Gradient Boosting and Catboosting. Returns ML with minimum
#' RMSE and the RMSE.
#'
#' @param X is a dataframe containing all the features on which the
#' model was estimated
#' @param Y is a vector containing the labels for which the model
#' was estimated
#' @param ML string vector specifying which machine learners to use
#' @param Kcv number of folds
#' @param ensemble is a string vector specifying which learners
#' should be used in ensemble methods (e.g. OLSensemble, SuperLearner)
#' @returns list containing ML attaining minimum RMSE and RMSE
#'
#'
#' @export
MLcv <- function(X,
                 Y,
                 ML = c("Lasso","Ridge","RF","CIF","XGB","CB",
                        "Logit_lasso","OLS","grf","OLSensemble"),
                 ensemble = c("Lasso","Ridge","RF","CIF","XGB","CB",
                              "Logit_lasso","OLS","grf","OLSensemble"),
                 Kcv = 5,
                 rf.cf.ntree = 500,
                 rf.depth = NULL,
                 ensemblefolds = 2,
                 polynomial = 1,
                 verbose = FALSE){
  n <- length(Y)
  X <- dplyr::as_tibble(X)
  ind <- split(seq(n), seq(n) %% Kcv)
  fv <- rep(0,n)
  res <- lapply(ML,function(u){
    for (i in 1:Kcv){
      if (verbose == TRUE){
        print(paste("Fold ",i, " of ", Kcv, " of ML ",u, sep = ""))
      }
      m <- ML::modest(X[-ind[[i]],],Y[-ind[[i]]],ML = u,
                      ensemble = ensemble,
                      rf.cf.ntree = rf.cf.ntree,
                      rf.depth = rf.depth,
                      polynomial = polynomial,
                      ensemblefolds = ensemblefolds)
      if (u == "OLSensemble"){
        coefs = m$coefs
        m = m$models
      } else{coefs = NULL}
      fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                 X[ind[[i]],],Y[ind[[i]]],ML = u,
                                polynomial = polynomial,
                                coefs = coefs)
    }
    rmse <- sqrt(mean((Y-fv)^2))
  })
  names(res) <- ML
  res <- unlist(res)
  return(list(mlbest = names(res[which.min(res)]),
              rmsebest = min(res),
              rmses = res))
}

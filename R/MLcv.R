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
#' @param OLSensemble is a string vector specifying which learners
#' should be used in OLS ensemble method
#' @param SL.library is a string vector specifying which learners
#' should be used in SuperLearner
#' @param Kcv number of folds in cross-validation
#' @param rf.cf.ntree how many trees should be grown when using RF or CIF
#' @param rf.depth how deep should trees be grown in RF (NULL is default from ranger)
#' @param mtry how many variables to consider at each split in RF
#' @param ensemblefolds is an integer specifying how many folds to use in ensemble
#' methods such as OLSensemble or SuperLearner
#' @param polynomial degree of polynomial to be fitted when using Lasso, Ridge,
#' Logit Lasso or OLS. 1 just fits the input X. 2 squares all variables and adds
#' all pairwise interactions. 3 squares and cubes all variables and adds all
#' pairwise and threewise interactions...
#' @param xgb.nrounds is an integer specifying how many rounds to use in XGB
#' @param xgb.max.depth is an integer specifying how deep trees should be grown in XGB
#' @param cb.iterations The maximum number of trees that can be built in CB
#' @param cb.depth The depth of the trees in CB
#' @param verbose logical specifying whether to print progress
#' @returns list containing ML attaining minimum RMSE and RMSE
#'
#'
#' @export
MLcv <- function(X,
                 Y,
                 ML = c("Lasso","Ridge","RF","CIF","XGB","CB",
                        "Logit_lasso","OLS","grf","OLSensemble"),
                 OLSensemble,
                 SL.library,
                 Kcv = 5,
                 rf.cf.ntree = 500,
                 rf.depth = NULL,
                 mtry = max(floor(ncol(X)/3), 1),
                 ensemblefolds = 2,
                 polynomial = 1,
                 xgb.nrounds = 200,
                 xgb.max.depth = 6,
                 cb.iterations = 1000,
                 cb.depth = 6,
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
                      OLSensemble = OLSensemble,
                      SL.library = SL.library,
                      rf.cf.ntree = rf.cf.ntree,
                      rf.depth = rf.depth,
                      mtry = mtry,
                      polynomial = polynomial,
                      xgb.nrounds = xgb.nrounds,
                      xgb.max.depth = xgb.max.depth,
                      cb.iterations = cb.iterations,
                      cb.depth = cb.depth,
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

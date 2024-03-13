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
#' @returns list containing ML attaining minimum RMSE and RMSE
#'
#'
#' @export
MLcv <- function(X,
                 Y,
                 ML = c("Lasso","Ridge","RF","CIF","XGB","CB","Logit_lasso"),
                 Kcv = 5,
                 rf.cf.ntree = 500,
                 rf.depth = NULL){
  n <- length(Y)
  X <- dplyr::as_tibble(X)
  ind <- split(seq(n), seq(n) %% Kcv)
  fv <- rep(0,n)
  res <- lapply(ML,function(u){
    for (i in 1:Kcv){
      m <- ML::modest(X[-ind[[i]],],Y[-ind[[i]]],ML = u,
                      rf.cf.ntree = rf.cf.ntree,
                      rf.depth = rf.depth)
      fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                 X[ind[[i]],],Y[ind[[i]]],ML = u)
    }
    rmse <- sqrt(mean((Y-fv)^2))
  })
  names(res) <- ML
  res <- unlist(res)
  return(list(mlbest = names(res[which.min(res)]),
              rmse = min(res)))
}

#' Perform CV on list of machine learners
#'
#' `MLtuning` performs hyperparameter tuning through cross-validation for
#' Lasso, Ridge, Random Forest, Conditional Inference Forest, Logit lasso,
#' Extreme Gradient Boosting and Catboosting (some might be missing still)
#' Returns hyperparameters with minimum
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
#' @param rf.cf.ntree.grid how many trees should be grown when using RF or CIF
#' @param rf.depth.grid how deep should trees be grown in RF (NULL is always tried)
#' @param mtry.grid how many variables to consider at each split in RF,
#' defaults floor(sqrt(ncol(X))) and floor(ncol(X)/3) are always tried
#' @param ensemblefolds.grid is an integer specifying how many folds to use in ensemble
#' methods such as OLSensemble or SuperLearner
#' @param polynomial.grid degree of polynomial to be fitted when using Lasso, Ridge,
#' Logit Lasso or OLS. 1 just fits the input X. 2 squares all variables and adds
#' all pairwise interactions. 3 squares and cubes all variables and adds all
#' pairwise and threewise interactions...
#' @param xgb.nrounds.grid is an integer specifying how many rounds to use in XGB
#' @param xgb.max.depth.grid is an integer specifying how deep trees should be grown in XGB
#' @param cb.iterations The maximum number of trees that can be built in CB
#' @param cb.depth The depth of the trees in CB
#' @param verbose logical specifying whether to print progress
#' @returns list containing ML attaining minimum RMSE and RMSE
#'
#'
#' @export
MLtuning <- function(X,
                 Y,
                 ML = c("Lasso","Ridge","RF","CIF","XGB","CB",
                        "Logit_lasso","OLS","grf","OLSensemble"),
                 OLSensemble,
                 SL.library,
                 Kcv = 5,
                 rf.cf.ntree.grid = c(100,300,500),
                 rf.depth.grid = c(2,4,6,10),
                 mtry.grid = c(1,3,5),
                 ensemblefolds.grid = c(2,5),
                 polynomial.grid = c(1,2,3),
                 xgb.nrounds.grid = c(100,200,500),
                 xgb.max.depth.grid = c(1,3,6),
                 cb.iterations.grid = c(100,500,1000),
                 cb.depth.grid = c(1,3,6,10),
                 verbose = FALSE){
  n <- length(Y)
  X <- dplyr::as_tibble(X)
  ind <- split(seq(n), seq(n) %% Kcv)
  restuning <- lapply(ML,function(u){
    if (u == "Lasso" | u == "Ridge" | u == "Logit_lasso" | u == "OLS"){
      combs = expand.grid(polynomial.grid)
      names(combs) = "polynomial"
      res = lapply(1:nrow(combs),function(j){
        polynomial = combs$polynomial[j]
        fv <- rep(0,n)
        for (i in 1:Kcv){
          if (verbose == TRUE){
            print(paste("Fold ",i, " of ", Kcv, " of ML ",u, sep = ""))
          }
          m <- ML::modest(X[-ind[[i]],],Y[-ind[[i]]],ML = u,
                          polynomial = polynomial)
          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u,
                                    polynomial = polynomial)
        }
        data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2)))
      })
      res = do.call(rbind,res)
      res = data.frame(combs,res)
    } else if (u == "RF"){
      if (!is.null(rf.depth.grid)){
        rf.depth.grid = c(rf.depth.grid,23101995)
      }
      if (max(floor(sqrt(ncol(X))),1) %in% mtry.grid == FALSE){
        mtry.grid = c(mtry.grid,max(floor(sqrt(ncol(X))),1))
      }
      if (max(floor(ncol(X)/3),1) %in% mtry.grid == FALSE){
        mtry.grid = c(mtry.grid,max(floor(ncol(X)/3),1))
      }
      combs = expand.grid(rf.cf.ntree.grid,rf.depth.grid,mtry.grid)
      names(combs) = c("rf.cf.ntree","rf.depth","mtry")
      res = lapply(1:nrow(combs),function(j){
        rf.cf.ntree = combs$rf.cf.ntree[j]
        rf.depth = combs$rf.depth[j]
        if (rf.depth == 23101995){
          rf.depth = NULL
        }
        mtry = combs$mtry[j]
        fv <- rep(0,n)
        for (i in 1:Kcv){
          if (verbose == TRUE){
            print(paste("Fold ",i, " of ", Kcv, " of ML ",u, sep = ""))
          }
          m <- ML::modest(X[-ind[[i]],],Y[-ind[[i]]],ML = u,
                          rf.cf.ntree = rf.cf.ntree,
                          rf.depth = rf.depth,
                          mtry = mtry)
          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u)
        }
        data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2)))
      })
      res = do.call(rbind,res)
      res = data.frame(combs,res)
    } else if (u == "CIF"){
      if (max(floor(sqrt(ncol(X))),1) %in% mtry.grid == FALSE){
        mtry.grid = c(mtry.grid,max(floor(sqrt(ncol(X))),1))
      }
      if (max(floor(ncol(X)/3),1) %in% mtry.grid == FALSE){
        mtry.grid = c(mtry.grid,max(floor(ncol(X)/3),1))
      }
      combs = expand.grid(rf.cf.ntree.grid,mtry.grid)
      names(combs) = c("rf.cf.ntree","mtry")
      res = lapply(1:nrow(combs),function(j){
        rf.cf.ntree = combs$rf.cf.ntree[j]
        mtry = combs$mtry[j]
        fv <- rep(0,n)
        for (i in 1:Kcv){
          if (verbose == TRUE){
            print(paste("Fold ",i, " of ", Kcv, " of ML ",u, sep = ""))
          }
          m <- ML::modest(X[-ind[[i]],],Y[-ind[[i]]],ML = u,
                          rf.cf.ntree = rf.cf.ntree,
                          mtry = mtry)
          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u)
        }
        data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2)))
      })
      res = do.call(rbind,res)
      res = data.frame(combs,res)
    } else if (u == "XGB"){
      combs = expand.grid(xgb.nrounds.grid,xgb.max.depth.grid)
      names(combs) = c("xgb.nrounds","xgb.max.depth")
      res = lapply(1:nrow(combs),function(j){
        xgb.nrounds = combs$xgb.nrounds[j]
        xgb.max.depth = combs$xgb.max.depth[j]
        fv <- rep(0,n)
        for (i in 1:Kcv){
          if (verbose == TRUE){
            print(paste("Fold ",i, " of ", Kcv, " of ML ",u, sep = ""))
          }
          m <- ML::modest(X[-ind[[i]],],Y[-ind[[i]]],ML = u,
                          xgb.nrounds = xgb.nrounds,
                          xgb.max.depth = xgb.max.depth)
          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u)
        }
        data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2)))
      })
      res = do.call(rbind,res)
      res = data.frame(combs,res)
    } else if (u == "CB"){
      combs = expand.grid(cb.iterations.grid,cb.depth.grid)
      names(combs) = c("cb.iterations","cb.depth")
      res = lapply(1:nrow(combs),function(j){
        cb.iterations = combs$cb.iterations[j]
        cb.depth = combs$cb.depth[j]
        fv <- rep(0,n)
        for (i in 1:Kcv){
          if (verbose == TRUE){
            print(paste("Fold ",i, " of ", Kcv, " of ML ",u, sep = ""))
          }
          m <- ML::modest(X[-ind[[i]],],Y[-ind[[i]]],ML = u,
                          cb.iteration = cb.iterations,
                          cb.depth = cb.depth)
          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u)
        }
        data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2)))
      })
      res = do.call(rbind,res)
      res = data.frame(combs,res)
    } else if (u == "OLSensemble"){
      res0 = lapply(OLSensemble, function(v){
        a = MLtuning(X = X,
                 Y = Y,
                 ML = v,
                 Kcv = Kcv,
                 rf.cf.ntree.grid = rf.cf.ntree.grid,
                 rf.depth.grid = rf.depth.grid,
                 mtry.grid = mtry.grid,
                 polynomial.grid = polynomial.grid,
                 xgb.nrounds.grid = xgb.nrounds.grid,
                 xgb.max.depth.grid = xgb.max.depth.grid)
        a$results_best[[1]]
      })
      for (jt in 1:length(res0)){
        aux = ncol(res0[[jt]]) - 2
        for (jr in 1:aux){
          assign(colnames(res0[[jt]])[jr],res0[[jt]][1,jr])
        }
      }
      `%notin%` <- Negate(`%in%`)
      if ("polynomial" %notin% ls()){
        polynomial = 1
      }
      if ("rf.cf.ntree" %notin% ls()){
        rf.cf.ntree = 500
      }
      if ("rf.depth" %in% ls()){
        if (rf.depth == 23101995){
          rf.depth = NULL
        }
      }
      if ("rf.depth" %notin% ls()){
        rf.depth = NULL
      }
      if ("mtry" %notin% ls()){
        mtry = 1
      }
      if ("xgb.nrounds" %notin% ls()){
        xgb.nrounds = 200
      }
      if ("xgb.max.depth" %notin% ls()){
        xgb.max.depth = 6
      }
      if ("cb.iterations" %notin% ls()){
        cb.iterations = 1000
      }
      if ("cb.depth" %notin% ls()){
        cb.depth = 6
      }
      if (1 %in% ensemblefolds.grid){
        stop("ensemblefolds has to be an integer larger than 1")
      }
      combs = expand.grid(ensemblefolds.grid)
      names(combs) = c("ensemblefolds")
      res = lapply(1:nrow(combs),function(j){
        ensemblefolds = combs$ensemblefolds[j]
        fv <- rep(0,n)
        for (i in 1:Kcv){
          if (verbose == TRUE){
            print(paste("Fold ",i, " of ", Kcv, " of ML ",u, sep = ""))
          }
          m <- ML::modest(X[-ind[[i]],],Y[-ind[[i]]],ML = u,
                          OLSensemble = OLSensemble,
                          rf.cf.ntree = rf.cf.ntree,
                          rf.depth = rf.depth,
                          polynomial = polynomial,
                          mt = mtry,
                          xgb.nrounds = xgb.nrounds,
                          xgb.max.depth = xgb.max.depth,
                          cb.iterations = cb.iterations,
                          cb.depth = cb.depth)

          coefs = m$coefs
          # m = m$models

          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u,
                                    polynomial = polynomial,
                                    coefs = coefs)
        }
        data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2)))
      })
      res = do.call(rbind,res)
      res = data.frame(combs,res)
    }
  })
  names(restuning) <- ML
  resbest = lapply(restuning, function(u){
    u[which.min(u$rmse),]
  })
  best.across.ML = resbest[[which.min(sapply(resbest,function(u){u$rmse}))]]

  return(list(results = restuning,
              results_best = resbest,
              best.across.ml = best.across.ML))
}

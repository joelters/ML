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
#' @param polynomial.Lasso.grid degree of polynomial to be fitted when using Lasso.
#' 1 just fits the input X. 2 squares all variables and adds
#' all pairwise interactions. 3 squares and cubes all variables and adds all
#' pairwise and threewise interactions...
#' @param polynomial.Ridge.grid degree of polynomial to be fitted when using Ridge,
#' see polynomial.Lasso for more info.
#' @param polynomial.Logit_lasso.grid degree of polynomial to be fitted when using Logit_lasso,
#' see polynomial.Lasso for more info.
#' @param polynomial.OLS.grid degree of polynomial to be fitted when using OLS,
#' see polynomial.Lasso for more info.
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
                 var_penalization = 0,
                 OLSensemble,
                 SL.library,
                 Kcv = 5,
                 rf.cf.ntree.grid = c(100,300,500),
                 rf.depth.grid = c(2,4,6,10),
                 mtry.grid = c(1,3,5),
                 ensemblefolds.grid = c(2,5),
                 polynomial.Lasso.grid = c(1,2,3),
                 polynomial.Ridge.grid = c(1,2,3),
                 polynomial.Logit_lasso.grid = c(1,2,3),
                 polynomial.OLS.grid = c(1,2,3),
                 xgb.nrounds.grid = c(100,200,500),
                 xgb.max.depth.grid = c(1,3,6),
                 cb.iterations.grid = c(100,500,1000),
                 cb.depth.grid = c(1,3,6,10),
                 torch.hidden_units.grid = list(c(64, 32), c(128, 64), c(256, 128, 64)),
                 torch.lr.grid = c(0.001, 0.01, 0.1),
                 torch.dropout.grid = c(0.1, 0.2, 0.3),
                 torch.epochs.grid = c(50, 100),
                 verbose = FALSE,
                 weights = NULL){
  n <- length(Y)
  X <- dplyr::as_tibble(X)
  ind <- split(seq(n), seq(n) %% Kcv)
  restuning <- lapply(ML,function(u){
    if (u == "Lasso" | u == "Ridge" | u == "Logit_lasso" | u == "OLS"){
      if (u == "Lasso"){
        polynomial.grid = polynomial.Lasso.grid
        combs = expand.grid(polynomial.grid)
        names(combs) = "polynomial.Lasso"
      }
      else if (u == "Ridge"){
        polynomial.grid = polynomial.Ridge.grid
        combs = expand.grid(polynomial.grid)
        names(combs) = "polynomial.Ridge"
      }
      else if (u == "Logit_lasso"){
        polynomial.grid = polynomial.Logit_lasso.grid
        combs = expand.grid(polynomial.grid)
        names(combs) = "polynomial.Logit_lasso"
      }
      else if (u == "OLS"){
        polynomial.grid = polynomial.OLS.grid
        combs = expand.grid(polynomial.grid)
        names(combs) = "polynomial.OLS"
      }
      res = lapply(1:nrow(combs),function(j){
        polynomial = combs$polynomial[j]
        fv <- rep(0,n)
        for (i in 1:Kcv){
          if (verbose == TRUE){
            print(paste("Fold ",i, " of ", Kcv, " of ML ",u, sep = ""))
          }
          m <- ML::modest(X[-ind[[i]],],Y[-ind[[i]]],ML = u,
                          polynomial.Lasso = polynomial,
                          polynomial.Ridge = polynomial,
                          polynomial.Logit_lasso = polynomial,
                          polynomial.OLS = polynomial,
                          weights = weights[-ind[[i]]])
          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u,
                                    polynomial.Lasso = polynomial,
                                    polynomial.Ridge = polynomial,
                                    polynomial.Logit_lasso = polynomial,
                                    polynomial.OLS = polynomial)
        }
        list(resMLrmse = data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2) + var_penalization*var(fv))), fvs = fv)
      })
      resMLrmse = lapply(1:length(res), function(uu){res[[uu]]$resMLrmse})
      fvs = lapply(1:length(res), function(uu){res[[uu]]$fvs})
      res = do.call(rbind,resMLrmse)
      res = data.frame(combs,res)
      list(res = res, fvs = fvs)
    }
    else if (u == "RF"){
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
                          mtry = mtry,
                          weights = weights[-ind[[i]]])
          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u)
        }
        list(resMLrmse = data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2) + var_penalization*var(fv))), fvs = fv)
      })
      resMLrmse = lapply(1:length(res), function(uu){res[[uu]]$resMLrmse})
      fvs = lapply(1:length(res), function(uu){res[[uu]]$fvs})
      res = do.call(rbind,resMLrmse)
      res = data.frame(combs,res)
      list(res = res, fvs = fvs)
    }
    else if (u == "CIF"){
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
                          mtry = mtry,
                          weights = weights[-ind[[i]]])
          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u)
        }
        list(resMLrmse = data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2) + var_penalization*var(fv))), fvs = fv)
      })
      resMLrmse = lapply(1:length(res), function(uu){res[[uu]]$resMLrmse})
      fvs = lapply(1:length(res), function(uu){res[[uu]]$fvs})
      res = do.call(rbind,resMLrmse)
      res = data.frame(combs,res)
      list(res = res, fvs = fvs)
    }
    else if (u == "XGB"){
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
                          xgb.max.depth = xgb.max.depth,
                          weights = weights[-ind[[i]]])
          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u)
        }
        list(resMLrmse = data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2) + var_penalization*var(fv))), fvs = fv)
      })
      resMLrmse = lapply(1:length(res), function(uu){res[[uu]]$resMLrmse})
      fvs = lapply(1:length(res), function(uu){res[[uu]]$fvs})
      res = do.call(rbind,resMLrmse)
      res = data.frame(combs,res)
      list(res = res, fvs = fvs)
    }
    else if (u == "CB"){
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
                          cb.depth = cb.depth,
                          weights = weights[-ind[[i]]])
          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u)
        }
        list(resMLrmse = data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2) + var_penalization*var(fv))), fvs = fv)
      })
      resMLrmse = lapply(1:length(res), function(uu){res[[uu]]$resMLrmse})
      fvs = lapply(1:length(res), function(uu){res[[uu]]$fvs})
      res = do.call(rbind,resMLrmse)
      res = data.frame(combs,res)
      list(res = res, fvs = fvs)
    }
    else if (u == "Torch"){
      combs = expand.grid(I(torch.hidden_units.grid), # Keep lists intact
                          torch.lr.grid,
                          torch.dropout.grid,
                          torch.epochs.grid)
      names(combs) = c("torch.hidden_units","torch.lr",
                       "torch.dropout", "torch.epochs")
      res = lapply(1:nrow(combs),function(j){
        torch.hidden_units = combs$torch.hidden_units[[j]]
        torch.lr = combs$torch.lr[j]
        torch.dropout = combs$torch.dropout[j]
        torch.epochs = combs$torch.epochs[j]
        fv <- rep(0,n)
        for (i in 1:Kcv){
          if (verbose == TRUE){
            print(paste("Fold ",i, " of ", Kcv, " of ML ",u, sep = ""))
          }
          m <- ML::modest(X[-ind[[i]],],Y[-ind[[i]]],ML = u,
                          torch.hidden_units = torch.hidden_units,
                          torch.lr = torch.lr,
                          torch.dropout = torch.dropout,
                          torch.epochs = torch.epochs,
                          weights = weights[-ind[[i]]])
          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u)
        }
        list(resMLrmse = data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2) + var_penalization*var(fv))), fvs = fv)
      })
      resMLrmse = lapply(1:length(res), function(uu){res[[uu]]$resMLrmse})
      fvs = lapply(1:length(res), function(uu){res[[uu]]$fvs})
      res = do.call(rbind,resMLrmse)
      res = data.frame(combs,res)
      list(res = res, fvs = fvs)
    }
    else if (u == "OLSensemble"){
      res0 = lapply(OLSensemble, function(v){
        a = MLtuning(X = X,
                 Y = Y,
                 ML = v,
                 Kcv = Kcv,
                 rf.cf.ntree.grid = rf.cf.ntree.grid,
                 rf.depth.grid = rf.depth.grid,
                 mtry.grid = mtry.grid,
                 polynomial.Lasso.grid = polynomial.Lasso.grid,
                 polynomial.Ridge.grid = polynomial.Ridge.grid,
                 polynomial.Logit_lasso.grid = polynomial.Logit_lasso.grid,
                 polynomial.OLS.grid = polynomial.OLS.grid,
                 xgb.nrounds.grid = xgb.nrounds.grid,
                 xgb.max.depth.grid = xgb.max.depth.grid,
                 weights = weights)
        a$results_best[[1]]
      })
      for (jt in 1:length(res0)){
        aux = ncol(res0[[jt]]) - 2
        for (jr in 1:aux){
          assign(colnames(res0[[jt]])[jr],res0[[jt]][1,jr])
        }
      }
      `%notin%` <- Negate(`%in%`)
      if ("polynomial.Lasso" %notin% ls()){
        polynomial.Lasso = 1
      }
      if ("polynomial.Ridge" %notin% ls()){
        polynomial.Ridge = 1
      }
      if ("polynomial.Logit_lasso" %notin% ls()){
        polynomial.Logit_lasso = 1
      }
      if ("polynomial.OLS" %notin% ls()){
        polynomial.OLS = 1
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
                          polynomial.Lasso = polynomial.Lasso,
                          polynomial.Ridge = polynomial.Ridge,
                          polynomial.Logit_lasso = polynomial.Logit_lasso,
                          polynomial.OLS = polynomial.OLS,
                          mt = mtry,
                          xgb.nrounds = xgb.nrounds,
                          xgb.max.depth = xgb.max.depth,
                          cb.iterations = cb.iterations,
                          cb.depth = cb.depth,
                          weights = weights[-ind[[i]]])

          coefs = m$coefs
          # m = m$models

          fv[ind[[i]]] <- ML::FVest(m,X[-ind[[i]],],Y[-ind[[i]]],
                                    X[ind[[i]],],Y[ind[[i]]],ML = u,
                                    polynomial.Lasso = polynomial.Lasso,
                                    polynomial.Ridge = polynomial.Ridge,
                                    polynomial.Logit_lasso = polynomial.Logit_lasso,
                                    polynomial.OLS = polynomial.OLS,
                                    coefs = coefs)
        }
        list(resMLrmse = data.frame(ML = u, rmse = sqrt(mean((Y-fv)^2) + var_penalization*var(fv))), fvs = fv)
      })
      resMLrmse = lapply(1:length(res), function(uu){res[[uu]]$resMLrmse})
      fvs = lapply(1:length(res), function(uu){res[[uu]]$fvs})
      res = do.call(rbind,resMLrmse)
      res = data.frame(combs,res)
      list(res = res, fvs = fvs)
    }
  })
  names(restuning) <- ML
  resbest = lapply(restuning, function(u){
    list(u$res[which.min(u$res$rmse),],
    u$fvs[[which.min(u$res$rmse)]])
  })
  best.across.ML = resbest[[which.min(sapply(resbest,function(u){u[[1]]$rmse}))]]
  best.res.across.ML = best.across.ML[[1]]
  best.fvs.across.ML = best.across.ML[[2]]

  return(list(results = lapply(restuning, function(u){u$res}),
              results_best = lapply(resbest, function(u){u[[1]]}),
              fvs_best = lapply(resbest, function(u){u[[2]]}),
              best.across.ml = best.res.across.ML,
              best.fvs.across.ml = best.fvs.across.ML))
}

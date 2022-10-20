#' @export
FVest <- function(model,
                  X,
                  Y,
                  Xnew,
                  Ynew,
                  ML = c("Lasso","Ridge","RF","CIF","XGB","CB","SL")){
  ML = match.arg(ML)
  #note that Y in dta is not used for anything so we just want it
  #to be consistent in the dimensions
  dta <- dplyr::as_tibble(cbind(Y = rep(0,nrow(Xnew)),Xnew))

  if (ML == "Lasso"){
    # XX <- model.matrix(Y ~., dta)
    lstar <- model$lambda.min
    FVs = predict(model,as.matrix(Xnew),s = lstar)
  }

  else if (ML == "Ridge"){
    # XX <- model.matrix(Y ~., dta)
    lstar <- model$lambda.min
    FVs = predict(model,as.matrix(Xnew),s = lstar)
  }

  else if (ML == "RF"){
    FVs <- predict(model,Xnew)
    FVs <- FVs$predictions
  }

  else if (ML == "CIF"){
    FVs <- predict(model, newdata = Xnew)
  }

  else if (ML == "XGB"){
    #Again label should not have any use here
    xgb_data = xgb.DMatrix(data = data.matrix(Xnew), label = rep(0,nrow(Xnew)))
    FVs = predict(model, xgb_data)
  }

  else if (ML == "CB"){
    #Again label should not have any use here
    CB.data <- catboost.load_pool(Xnew,
                                  label = rep(0,nrow(Xnew)),
                                  cat_features = c(1:ncol(X)))
    FVs <- catboost.predict(model,CB.data)
  }
  else if (ML == "SL"){
    ens <- model$SL.library$library$predAlgorithm
    # FVs <- model$SL.predict
    if ("SL.CB" %in% ens){
      FVs <- unlist(lapply(ens, function (x){
        sl <- get(x)
        aux <- SuperLearner::sl(Y, X, Xnew, family = gaussian(), obsWeights = rep(1,length(Y)))
        aux$pred
      }))
      FVs <- matrix(FVs, nrow(Xnew), length(ens))
      sharemat <- matrix(rep(model$coef,nrow(Xnew)), nrow(Xnew), length(ens), byrow = TRUE)
      FVs <- rowSums(sharemat*FVs)
    }
    else{
      FVs = predict(model, Xnew, onlySL = TRUE)
      FVs <- FVs$pred
    }
  }
  if (min(FVs) <= 0){
    warning("There are negative/zero FVs which have been set to 1")
    FVs = FVs*(FVs > 0) + 1*(FVs <= 0)
  }
  return(FVs)
}

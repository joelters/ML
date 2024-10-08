#' Predict fitted values from Machine Learning model
#'
#' `FVest` takes an estimated machine learning model (Lasso, Ridge,
#' Random Forest, Conditional Inference Forest,
#' Extreme Gradient Boosting, Catboosting, Logit lasso or any
#' combination of these using the SuperLearner package) and returns
#' the predicted fitted values for Xnew.
#'
#'@param model is an estimated Machine Learning model. Typically
#' a class S3 or S4 object.
#' @param X is a dataframe containing all the features on which the
#' model was estimated
#' @param Y is a vector containing the labels for which the model
#' was estimated
#' @param Xnew is a dataframe containing the features at which we
#' we want the predictions. Default is Xnew = X.
#' @param Ynew is a vector which needs to have length equal to the
#' rows of Xnew. It only matters that it has correct length so one
#' could use a vectors of zeros.
#' @param ML is a string specifying which machine learner to use
#' @param polynomial degree of polynomial to be fitted when using Lasso, Ridge
#' or Logit Lasso. 1 just fits the input X. 2 squares all variables and adds
#' all pairwise interactions. 3 squares and cubes all variables and adds all
#' pairwise and threewise interactions...
#' @param coefs optimal coefficients for OLSensemble, computed in modest
#' @returns vector with fitted values
#' @examples
#' X <- dplyr::select(mad2019,-Y)
#' Y <- mad2019$Y
#' m <- modest(X,Y,"RF")
#' FVest(m,X,Y,X[1:5,],Y[1:5],ML = "RF")
#'
#' m <- modest(X,Y,"XGB")
#' FVest(m,X,Y,ML = "XGB")
#'
#' m <- modest(X,Y,"SL",
#' ensemble = c("SL.Lasso","SL.RF","SL.XGB"))
#' FVest(m,X,Y,ML = "SL")
#'
#'
#' @details Note that the glmnet package which implements Lasso and Ridge
#' does not handle factor variables (such as the ones in mad2019)
#' hence for this machine learners, modest turns X into model.matrix(~.,X)
#' which will perform dummy encoding on factor variables.
#' @export
FVest <- function(model,
                  X,
                  Y,
                  Xnew = X,
                  Ynew = Y,
                  ML = c("Lasso","Ridge","RF","CIF","XGB","CB",
                         "Logit_lasso","OLS","grf","SL","OLSensemble"),
                  polynomial = 1,
                  coefs = NULL){
  ML = match.arg(ML)
  Ynew <- as.numeric(Ynew)

  if (!("data.frame" %in% class(X))){
    X <- data.frame(X)
  }
  if (!("data.frame" %in% class(Xnew))){
    Xnew <- data.frame(Xnew)
  }

  #note that Y in dta is not used for anything so we just want it
  #to be consistent in the dimensions
  dta <- dplyr::as_tibble(cbind(Y = rep(0,nrow(Xnew)),Xnew))
  colnames(dta)[1] <- "Y"


  if (ML == "Lasso" | ML == "Ridge" | ML == "Logit_lasso" | ML == "OLS"){
    if (polynomial == 1){
      MM <- stats::model.matrix(~(.), Xnew)
    }
    else if (polynomial >= 2){
      M <- stats::model.matrix(~(.), Xnew)
      M2 <- as.matrix(M[,2:ncol(M)])
      if (ncol(M) == 2){
        colnames(M2) <- colnames(M)[2]
      }
      M <- M2
      Mnon01 <- colnames(M)[!apply(M,2,function(u){all(u %in% 0:1)})]
      if (length(Mnon01) != 0){
        A <- lapply(2:polynomial, function(u){
          B <- M[,Mnon01]^u
        })
        A <- do.call(cbind,A)
        colnames(A) <- c(sapply(2:polynomial, function(u){paste(Mnon01,"tothe",u,sep = "")}))
      }
      else{
        A <- NULL
      }
      fml<- as.formula(paste("~(.)^",polynomial,sep=""))
      MM <- cbind(stats::model.matrix(fml,Xnew),A)
    }
    else{
      stop("polynomial has to be an integer larger or equal than 1")
    }
    if (ncol(MM) > 2){
      Xnew <- as.matrix(MM[,2:ncol(MM)])
    }
    else{
      Xnew <- MM
    }
  }

  if (ML == "Lasso" | ML == "Logit_lasso"){
    lstar <- model$lambda.min
    FVs = stats::predict(model,Xnew,
                         type = "response", s = lstar)
  }

  else if (ML == "Ridge"){
    lstar <- model$lambda.min
    FVs = stats::predict(model, Xnew, s = lstar)
  }

  else if (ML == "OLS"){
    FVs = stats::predict(model, data.frame(Xnew))
  }

  else if (ML == "RF"){
    if (length(model$forest$independent.variable.names) == length(names(Xnew))){
      if (!all(model$forest$independent.variable.names == names(Xnew))){
        names(Xnew) = model$forest$independent.variable.names
      }
    }
    else{
      stop("Model was trained on a different number of features than the ones acting
         as input for prediction.")
    }

    FVs <- stats::predict(model,Xnew)
    FVs <- FVs$predictions
  }

  else if (ML == "CIF"){
    FVs <- stats::predict(model, newdata = Xnew)
  }

  else if (ML == "XGB"){
    if (!requireNamespace("xgboost", quietly = TRUE)) {
      stop(
        "Package \"xgboost\" must be installed to use this function.",
        call. = FALSE
      )
    }
    if (length(model$feature_names) == length(names(Xnew))){
      if (!all(model$feature_names == names(Xnew))){
        names(Xnew) = model$feature_names
      }
    }
    else{
      stop("Model was trained on a different number of features than the ones acting
         as input for prediction.")
    }

    #Again label should not have any use here
    xgb_data = xgboost::xgb.DMatrix(data = data.matrix(Xnew), label = rep(0,nrow(Xnew)))
    FVs = stats::predict(model, xgb_data)
  }

  else if (ML == "CB"){
    if (!requireNamespace("catboost", quietly = TRUE)) {
      stop(
        "Package \"catboost\" must be installed to use this function.
        https://catboost.ai/en/docs/installation/r-installation-binary-installation",
        call. = FALSE
      )
    }
    #Again label should not have any use here
    # CB.data <- catboost::catboost.load_pool(Xnew,
    #                               label = rep(0,nrow(Xnew)),
    #                               cat_features = c(1:ncol(Xnew)))
    CB.data <- catboost::catboost.load_pool(Xnew,
                                            label = rep(0,nrow(Xnew)))
    FVs <- catboost::catboost.predict(model,CB.data)
  }

  else if (ML == "grf"){
    FVs = stats::predict(model, newdata = Xnew)$predictions
  }
  else if (ML == "SL"){
    if (!requireNamespace("SuperLearner", quietly = TRUE)) {
      stop(
        "Package \"SuperLearner\" must be installed to use this function.",
        call. = FALSE
      )
    }
    ens <- model$SL.library$library$predAlgorithm
    # FVs <- model$SL.predict
    if ("SL.CB" %in% ens){
      FVs <- unlist(lapply(ens, function (x){
        sl <- get(x)
        aux <- sl(Y, X, Xnew, family = stats::gaussian(), obsWeights = rep(1,length(Y)))
        aux$pred
      }))
      FVs <- matrix(FVs, nrow(Xnew), length(ens))
      sharemat <- matrix(rep(model$coef,nrow(Xnew)), nrow(Xnew), length(ens), byrow = TRUE)
      FVs <- rowSums(sharemat*FVs)
    }
    else{
      FVs = stats::predict(model, Xnew, onlySL = TRUE)
      FVs <- FVs$pred
    }
  }

  else if (ML == "OLSensemble"){
    if(class(model) != "list"){
      stop("For OLSensemble, model has to be a list of the models used for
           the ensemble")
    }
    ensemble = names(model)
    if (class(coefs) != "numeric" | length(coefs) != (length(ensemble)+1)){
      stop("coefs has to be numeric and have the same dimension as ensemble plus 1
           (the intercept)")
    }
    nnew = length(Ynew)
    Xpred = matrix(rep(NA,nnew*length(ensemble)),nnew,length(ensemble))
    for (ii in 1:length(ensemble)){
      Xpred[,ii] = ML::FVest(model[[ii]], X, Y, Xnew, Ynew,
                             ML = ensemble[ii],polynomial = polynomial)
    }
    Xpred = cbind(rep(1,nnew),Xpred)
    FVs = Xpred%*%coefs
  }
  return(FVs)
}

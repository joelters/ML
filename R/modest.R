#' Estimate Machine Learning model
#'
#' `modest` estimates the model for a specified machine learner,
#'  possible options are Lasso, Ridge, Random Forest, Conditional
#' Inference Forest, Extreme Gradient Boosting, Catboosting, Logit lasso
#' or any combination of these using the SuperLearner package
#'
#' @param X is a dataframe containing all the features
#' @param Y is a vector containing the label
#' @param ML is a string specifying which machine learner to use
#' @param OLSensemble is a string vector specifying which learners
#' should be used in OLS ensemble method
#' @param SL.library is a string vector specifying which learners
#' should be used in SuperLearner
#' @param rf.cf.ntree how many trees should be grown when using RF or CIF
#' @param rf.depth how deep should trees be grown in RF (NULL is default from ranger)
#' @param polynomial degree of polynomial to be fitted when using Lasso, Ridge,
#' Logit Lasso or OLS. 1 just fits the input X. 2 squares all variables and adds
#' all pairwise interactions. 3 squares and cubes all variables and adds all
#' pairwise and threewise interactions...
#' @param ensemblefolds is an integer specifying how many folds to use in ensemble
#' methods such as OLSensemble or SuperLearner
#' @param xgb.nrounds is an integer specifying how many rounds to use in XGB
#' @param xgb.max.depth is an integer specifying how deep trees should be grown in XGB
#' @param weights is a vector containing survey weights adding up to 1
#' @returns the object that the machine learner package returns, in case of OLSensemble
#' it returns the coefficients assigned to each machine learner in ensemble
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
                   ML = c("Lasso","Ridge","RF","CIF","XGB","CB",
                          "Logit_lasso","OLS","grf","SL","OLSensemble"),
                   OLSensemble,
                   SL.library,
                   rf.cf.ntree = 500,
                   rf.depth = NULL,
                   mtry = max(floor(ncol(X)/3), 1),
                   polynomial = 1,
                   ensemblefolds = 10,
                   xgb.nrounds = 200,
                   xgb.max.depth = 6,
                   weights = NULL){
  Y <- as.numeric(Y)
  ML = match.arg(ML)
  dta <- dplyr::as_tibble(cbind(Y = Y,X))
  colnames(dta)[1] <- "Y"

  if (!("data.frame" %in% class(X))){
    X <- data.frame(X)
  }

  if (ML == "Lasso" | ML == "Ridge" | ML == "Logit_lasso" | ML == "OLS"){
    if (polynomial == 1){
      MM <- stats::model.matrix(~(.), X)
    }
    else if (polynomial >= 2){
      M <- stats::model.matrix(~(.), X)
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
      MM <- cbind(stats::model.matrix(fml,X),A)
    }
    else{
      stop("polynomial has to be an integer larger or equal than 1")
    }
    if (ncol(MM) > 2){
      X <- as.matrix(MM[,2:ncol(MM)])
    }
    else{
      X <- MM
    }
  }

  if (ML == "SL"){
    if (!requireNamespace("SuperLearner", quietly = TRUE)) {
      stop(
        "Package \"SuperLearner\" must be installed to use this function.",
        call. = FALSE
      )
    }
    #Estimate model
    model <- SuperLearner::SuperLearner(Y, X, SL.library = SL.library,
                                        family = stats::gaussian(),
                                        cvControl = list(V = ensemblefolds),
                                        obsWeights = weights)
  }

  else if (ML == "Lasso"){
    # XX <- model.matrix(Y ~., dta)
    model <- glmnet::cv.glmnet(X,as.matrix(Y),alpha = 1, weights = weights)
  }

  else if (ML == "Logit_lasso"){
    # XX <- model.matrix(Y ~., dta)
    model <- glmnet::cv.glmnet(X,as.matrix(Y), family = "binomial",
                               alpha = 1, weights = weights)
  }

  else if (ML == "Ridge"){
    # XX <- model.matrix(Y ~., dta)
    model <- glmnet::cv.glmnet(X,as.matrix(Y),alpha = 0, weights = weights)
  }

  else if (ML == "OLS"){
    # XX <- model.matrix(Y ~., dta)
    model <- stats::lm(Y ~ ., data = data.frame(Y = as.numeric(Y), X), weights = weights)
  }

  else if (ML == "RF"){
    model <- ranger::ranger(Y ~ .,
                    data = dta,
                    mtry = mtry,
                    num.trees = rf.cf.ntree,
                    max.depth = rf.depth,
                    case.weights = weights,
                    respect.unordered.factors = 'partition')
  }

  else if (ML == "CIF"){
    model <- party::cforest(Y ~ .,
                     data = dta,
                     controls = party::cforest_unbiased(mtry = mtry,
                                                        ntree = rf.cf.ntree),
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
                     nrounds = xgb.nrounds,
                     max.depth = xgb.max.depth,
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

  else if (ML == "grf"){
    model <- grf::regression_forest(X = X, Y = Y, sample.weights = weights)
  }

  else if(ML == "OLSensemble"){
    n <- length(Y)
    ind <- split(seq(n), seq(n) %% ensemblefolds)
    res = sapply(OLSensemble, function(u){
      pred = rep(NA,n)
      for (ii in 1:ensemblefolds){
        mm = ML::modest(X[-ind[[ii]],], Y[-ind[[ii]]], ML = u,
                        rf.cf.ntree = rf.cf.ntree,
                        rf.depth = rf.depth,
                        polynomial = polynomial,
                        weights = weights[-ind[[ii]]])

        pred[ind[[ii]]] = ML::FVest(mm,X[-ind[[ii]],],Y[-ind[[ii]]],
                         X[ind[[ii]],],Y[ind[[ii]]],ML = u,
                         polynomial = polynomial)
      }
      pred
    })
    dfens = data.frame(Y = Y,res)
    names(dfens) = c("Y",OLSensemble)
    ols = lm(Y~., data = dfens)
    coefs = ols$coefficients
    ms = lapply(OLSensemble, function(u){
      ML::modest(X, Y, ML = u,
                 rf.cf.ntree = rf.cf.ntree,
                 rf.depth = rf.depth,
                 polynomial = polynomial,
                 weights = weights)
    })
    names(ms) = OLSensemble
    return(list(models = ms, coefs = coefs))
  }
}

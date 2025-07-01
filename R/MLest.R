#' Estimate machine learning model and optionally give fitted values
#'
#' `MLest` estimates a machine learning model (Lasso, Ridge,
#' Random Forest, Conditional Inference Forest,
#' Extreme Gradient Boosting, Catboosting, Logit lasso or any
#' combination of these using the SuperLearner package) and (optionally)
#' returns the predicted fitted values.
#'
#' @param X is a dataframe containing all the features on which the
#' model was estimated
#' @param Y is a vector containing the labels for which the model
#' was estimated
#' @param ML is a string specifying which machine learner to use
#' @param OLSensemble is a string vector specifying which learners
#' should be used in OLS ensemble method
#' @param SL.library is a string vector specifying which learners
#' should be used in SuperLearner
#' @param rf.cf.ntree how many trees should be grown when using RF or CIF
#' @param rf.depth how deep should trees be grown in RF (NULL is default from ranger)
#' @param cf.depth how deep should trees be grown in CIF (Inf is default from partykit)
#' @param mtry how many variables to consider at each split in RF
#' @param polynomial.Lasso degree of polynomial to be fitted when using Lasso.
#' 1 just fits the input X. 2 squares all variables and adds
#' all pairwise interactions. 3 squares and cubes all variables and adds all
#' pairwise and threewise interactions...
#' @param polynomial.Ridge degree of polynomial to be fitted when using Ridge,
#' see polynomial.Lasso for more info.
#' @param polynomial.Logit_lasso degree of polynomial to be fitted when using Logit_lasso,
#' see polynomial.Lasso for more info.
#' @param polynomial.OLS degree of polynomial to be fitted when using OLS,
#' see polynomial.Lasso for more info.
#' @param polynomial.NLLS_exp degree of polynomial to be fitted when using NLLS_exp,
#' see polynomial.Lasso for more info.
#' @param polynomial.loglin degree of polynomial to be fitted when using loglin,
#' see polynomial.Lasso for more info.
#' @param xgb.nrounds is an integer specifying how many rounds to use in XGB
#' @param xgb.max.depth is an integer specifying how deep trees should be grown in XGB
#' @param cb.iterations The maximum number of trees that can be built in CB
#' @param cb.depth The depth of the trees in CB
#' @param start_nlls List with the starting values of the parameters. Default is log(mean(Y))
#' for the intercept and zero for all the rest.
#' @param torch.epochs is an integer specifying the number of epochs (full passes through the dataset)
#' to use when training the Torch neural network.
#' @param torch.hidden_units is a numeric vector specifying the number of neurons
#' in each hidden layer of the Torch neural network.
#' @param torch.lr is a numeric value specifying the learning rate to be used for the
#' optimizer when training the Torch neural network.
#' @param torch.dropout is a numeric value between 0 and 1 specifying the dropout rate
#' for regularization in the Torch neural network.
#' @param FVs a logical indicating whether FVs should be computed
#' @param weights survey weights adding up to 1
#' @param ensemblefolds number of folds to split in OLSensemble method
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
                  ML = c("Lasso","Ridge","RF","CIF","XGB","CB", "Torch",
                         "Logit_lasso","OLS", "NLLS_exp",
                        "loglin", "grf","SL","OLSensemble"),
                  OLSensemble,
                  SL.library,
                  rf.cf.ntree = 500,
                  rf.depth = NULL,
                  mtry = max(floor(ncol(X)/3), 1),
                  cf.depth = Inf,
                  polynomial.Lasso = 1,
                  polynomial.Ridge = 1,
                  polynomial.Logit_lasso = 1,
                  polynomial.OLS = 1,
                  polynomial.NLLS_exp = 1,
                  polynomial.loglin = 1,
                  start_nlls = NULL,
                  xgb.nrounds = 200,
                  xgb.max.depth = 6,
                  cb.iterations = 500,
                  cb.depth = 6,
                  torch.epochs = 50,
                  torch.hidden_units = c(64, 32),
                  torch.lr = 0.01,
                  torch.dropout = 0.2,
                  FVs = TRUE,
                  ensemblefolds = 2,
                  weights = NULL){
  ML = match.arg(ML)
  X <- dplyr::as_tibble(X)
  if (ML == "SL"){
    require("ranger")
    #Estimate model
    m <- SuperLearner::SuperLearner(Y, X, SL.library = SL.library,
                                    family = stats::gaussian(),
                                    cvControl = list(V = 5),
                                    obsWeights = weights)
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
    m <- modest(X, Y, ML, OLSensemble = OLSensemble,
                SL.library = SL.library,
                weights = weights,
                rf.cf.ntree = rf.cf.ntree,
                rf.depth = rf.depth,
                mtry = mtry,
                cf.depth = cf.depth,
                polynomial.Lasso = polynomial.Lasso,
                polynomial.Ridge = polynomial.Ridge,
                polynomial.Logit_lasso = polynomial.Logit_lasso,
                polynomial.OLS = polynomial.OLS,
                polynomial.NLLS_exp = polynomial.NLLS_exp,
                polynomial.loglin = polynomial.loglin,
                start_nlls = start_nlls,
                xgb.nrounds = xgb.nrounds,
                xgb.max.depth = xgb.max.depth,
                cb.iterations = cb.iterations,
                cb.depth = cb.depth,
                torch.epochs = torch.epochs,
                torch.hidden_units = torch.hidden_units,
                torch.lr = torch.lr,
                torch.dropout = torch.dropout,
                ensemblefolds = ensemblefolds)
    if (ML == "OLSensemble"){
      coefs = m$coefs
      # m = m$models
    } else{coefs = NULL}
    #Fitted values
    if (FVs == TRUE){
      FVs <- FVest(m, X, Y, X, Y, ML, polynomial.Lasso = polynomial.Lasso,
                   polynomial.Ridge = polynomial.Ridge,
                   polynomial.Logit_lasso = polynomial.Logit_lasso,
                   polynomial.OLS = polynomial.OLS,
                   polynomial.NLLS_exp = polynomial.NLLS_exp,
                   polynomial.loglin = polynomial.loglin,
                   coefs = coefs)
      return(list("model" = m, "FVs" = FVs, "coefs" = coefs))
    }
    else{
      return(list("model" = m, "coefs" = coefs))
    }
  }
}

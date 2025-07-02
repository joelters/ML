#' Estimate Machine Learning model
#'
#' `modest` estimates the model for a specified machine learner,
#'  possible options are Lasso, Ridge, Random Forest, Conditional
#' Inference Forest, Extreme Gradient Boosting, Catboosting, Logit Lasso,
#' NLLS with exp(x'b), loglin with exp(x'b) and x'b from OLS of lnY on X
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
#' @param cf.depth how deep should trees be grown in CIF (Inf is default from partykit)
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
#' @param ensemblefolds is an integer specifying how many folds to use in ensemble
#' methods such as OLSensemble or SuperLearner
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
                   ML = c("Lasso","Ridge","RF","CIF","XGB","CB", "Torch",
                          "NLLS_exp", "loglin", "Logit_lasso","OLS",
                          "grf","SL","OLSensemble"),
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
                   ensemblefolds = 10,
                   xgb.nrounds = 200,
                   xgb.max.depth = 6,
                   cb.iterations = 500,
                   cb.depth = 6,
                   torch.epochs = 50,
                   torch.hidden_units = c(64, 32),
                   torch.lr = 0.01,
                   torch.dropout = 0.2,
                   weights = NULL){
  Y <- as.numeric(Y)
  ML = match.arg(ML)
  dta <- dplyr::as_tibble(cbind(Y = Y,X))
  colnames(dta)[1] <- "Y"
  if (!("data.frame" %in% class(X))){
    X <- data.frame(X)
  }

  if (ML == "Lasso" | ML == "Ridge" | ML == "Logit_lasso" | ML == "OLS" |
      ML == "NLLS_exp"| ML == "loglin"){
    if (ML == "Lasso"){
      polynomial = polynomial.Lasso
    }
    else if (ML == "Ridge"){
      polynomial = polynomial.Ridge
    }
    else if (ML == "Logit_lasso"){
      polynomial = polynomial.Logit_lasso
    }
    else if (ML == "OLS"){
      polynomial = polynomial.OLS
    }
    else if (ML == "NLLS_exp"){
      polynomial = polynomial.NLLS_exp
    }
    else if (ML == "loglin"){
      polynomial = polynomial.loglin
    }
    if (polynomial == 1){
      if(ncol(X) == 0){
        X = data.frame(rep(1,nrow(X)))
      }
      MM <- stats::model.matrix(~(.), X)
      if(ncol(X) == 1 & length(unique(X[, 1])) == 1){
        aa = as.matrix(MM[,1])
        colnames(aa) = colnames(MM)[1]
        MM = aa
      }
    }
    else if (polynomial >= 2){
      if(ncol(X) == 0){
        X = data.frame(rep(1,nrow(X)))
      }
      M <- stats::model.matrix(~(.), X)
      if(ncol(X) == 1 & length(unique(X[, 1])) == 1){
        aa = as.matrix(M[,1])
        colnames(aa) = colnames(M)[1]
        M = aa
      }
      if (ncol(M) == 1){
        MM = M
      } else{
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
  else if (ML == "loglin"){
    # XX <- model.matrix(Y ~., dta)
    model <- stats::lm(log(Y) ~ ., data = data.frame(Y = as.numeric(Y), X), weights = weights)
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
    model <- partykit::cforest(Y ~ .,
                     data = dta,
                     ntree = rf.cf.ntree,
                     mtry = mtry,
                     control = partykit::ctree_control(maxdepth = cf.depth),
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
                            params = list(iterations = cb.iterations,
                                          depth = cb.depth,
                                          logging_level = 'Silent'))
  }
  else if (ML == "Torch"){
    X <- stats::model.matrix(~ ., X)
    # Convert data to tensors
    X_tensor <- torch::torch_tensor(as.matrix(X), dtype = torch::torch_float())
    Y_tensor <- torch::torch_tensor(as.matrix(Y), dtype = torch::torch_float())

    # Define the Neural Network
    model <- torch::nn_module(
      initialize = function(input_size, hidden_units, dropout_rate) {
        self$fc1 <- torch::nn_linear(input_size, hidden_units[1])
        self$bn1 <- torch::nn_batch_norm1d(hidden_units[1])
        self$fc2 <- torch::nn_linear(hidden_units[1], hidden_units[2])
        self$bn2 <- torch::nn_batch_norm1d(hidden_units[2])
        self$fc3 <- torch::nn_linear(hidden_units[2], 1)
        self$dropout <- torch::nn_dropout(dropout_rate)
      },
      forward = function(x) {
        x %>%
          self$fc1() %>% self$bn1() %>% torch::nnf_relu() %>% self$dropout() %>%
          self$fc2() %>% self$bn2() %>% torch::nnf_relu() %>% self$dropout() %>%
          self$fc3()
      }
    )

    # Instantiate the Model
    net <- model(input_size = ncol(X), hidden_units = torch.hidden_units, dropout_rate = torch.dropout)

    # Define Optimizer & Loss Function
    optimizer <- torch::optim_adam(net$parameters, lr = torch.lr)
    loss_fn <- torch::nn_mse_loss()

    # Training Loop
    for (epoch in 1:torch.epochs) {
      optimizer$zero_grad()
      output <- net(X_tensor)
      loss <- loss_fn(output, Y_tensor)
      loss$backward()
      optimizer$step()
    }
    model <- net
    return(model)  # Return trained model
  }
  else if (ML == "grf"){
    model <- grf::regression_forest(X = X, Y = Y, sample.weights = weights)
  }
  else if (ML == "NLLS_exp"){
    if (ncol(X) == 1 & length(unique(X[, 1])) == 1){
      nls_formula <- Y ~ exp(beta0)
      start_nlls <- list(beta0 = log(mean(Y)))
    } else{
      if(colnames(X)[1] == "(Intercept)"){
        X = as.matrix(X[,-1, drop = FALSE])
      }
      regs = colnames(data.frame(X))
      params <- paste0("beta", seq_along(regs))
      names(params) <- regs
      formula_str <- paste0("Y ~ exp(beta0 +", paste(params, regs, sep = "*", collapse = " + "), ")")
      nls_formula = as.formula(formula_str)
      if (is.null(start_nlls) == TRUE){
        start_nlls <- as.list(c(log(mean(Y)), rnorm(length(params),0,0.05)))
        names(start_nlls) = paste0("beta", seq_along(c(1,regs)) - 1)
      }
      else {
        names(start_nlls) = paste0("beta", seq_along(c(1,regs)) - 1)
      }
    }
    if(is.null(weights)){
      model = stats::nls(formula = nls_formula, data = data.frame(Y = Y, X),
                         start = start_nlls,
                         control = nls.control(maxiter = 500))
    } else{
      model = stats::nls(formula = nls_formula, data = data.frame(Y = Y, X),
                         start = start_nlls,
                         control = nls.control(maxiter = 500),
                         weights = weights)
    }
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
                        mtry = mtry,
                        cf.depth = cf.depth,
                        polynomial.Lasso = polynomial.Lasso,
                        polynomial.Ridge = polynomial.Ridge,
                        polynomial.Logit_lasso = polynomial.Logit_lasso,
                        polynomial.OLS = polynomial.OLS,
                        xgb.nrounds = xgb.nrounds,
                        xgb.max.depth = xgb.max.depth,
                        cb.iterations = cb.iterations,
                        cb.depth = cb.depth,
                        weights = weights[-ind[[ii]]])

        pred[ind[[ii]]] = ML::FVest(mm,X[-ind[[ii]],],Y[-ind[[ii]]],
                         X[ind[[ii]],],Y[ind[[ii]]],ML = u,
                         polynomial.Lasso = polynomial.Lasso,
                         polynomial.Ridge = polynomial.Ridge,
                         polynomial.Logit_lasso = polynomial.Logit_lasso,
                         polynomial.OLS = polynomial.OLS)
      }
      pred
    })
    dfens = data.frame(Y = Y,res)
    names(dfens) = c("Y",OLSensemble)
    ols = lm(Y~., data = dfens)
    coefs = ols$coefficients
    # take out models if there are NAs due to multicollinearity
    OLSensemble <- OLSensemble[!is.na(coefs)[2:length(coefs)]]
    coefs <- coefs[!is.na(coefs)]
    ms = lapply(OLSensemble, function(u){
      ML::modest(X, Y, ML = u,
                 rf.cf.ntree = rf.cf.ntree,
                 rf.depth = rf.depth,
                 mtry = mtry,
                 cf.depth = cf.depth,
                 polynomial.Lasso = polynomial.Lasso,
                 polynomial.Ridge = polynomial.Ridge,
                 polynomial.Logit_lasso = polynomial.Logit_lasso,
                 polynomial.OLS = polynomial.OLS,
                 xgb.nrounds = xgb.nrounds,
                 xgb.max.depth = xgb.max.depth,
                 cb.iterations = cb.iterations,
                 cb.depth = cb.depth,
                 weights = weights)
    })
    names(ms) = OLSensemble
    return(list(models = ms, coefs = coefs))
  }
}

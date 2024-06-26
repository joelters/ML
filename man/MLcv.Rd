% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/MLcv.R
\name{MLcv}
\alias{MLcv}
\title{Perform CV on list of machine learners}
\usage{
MLcv(
  X,
  Y,
  ML = c("Lasso", "Ridge", "RF", "CIF", "XGB", "CB", "Logit_lasso", "OLS"),
  Kcv = 5,
  rf.cf.ntree = 500,
  rf.depth = NULL
)
}
\arguments{
\item{X}{is a dataframe containing all the features on which the
model was estimated}

\item{Y}{is a vector containing the labels for which the model
was estimated}

\item{ML}{string vector specifying which machine learners to use}

\item{Kcv}{number of folds}
}
\value{
list containing ML attaining minimum RMSE and RMSE
}
\description{
\code{MLcv} computes cross-validated RMSE for a list including up to
Lasso, Ridge,
Random Forest, Conditional Inference Forest, Logit lasso,
Extreme Gradient Boosting and Catboosting. Returns ML with minimum
RMSE and the RMSE.
}

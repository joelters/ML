---
output: github_document
---

<!-- README.md is generated from README.Rmd. Please edit that file -->

```{r, include = FALSE}
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.path = "man/figures/README-",
  out.width = "100%"
)
```

<!-- badges: start -->
<!-- badges: end -->

The goal of ML is to provide a simple to use package to estimate Lasso, Ridge, Random Forest (RF),
Conditional Inference Forest (CIF), Extreme Gradient Boosting (XGB), Catboosting (CB) and 
a SuperLearner combining all of these learners. The package has three user functions: modest,
FVest and MLest. modest estimates only the model of the chosen Machine Learner. FVest takes
a model and gives the predicted fitted values for new features of choice. MLest combines both 
functions but only computes fitted values of the same observations used to build the model.

## Installation

You can install the development version of ML from [GitHub](https://github.com/) with:
      
``` r
# install.packages("devtools")
devtools::install_github("joelters/ML")
```



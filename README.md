The goal of ML is to provide a simple to use package to estimate Lasso, Ridge, Random Forest (RF),
Conditional Inference Forest (CIF), Extreme Gradient Boosting (XGB), Catboosting (CB) and 
a SuperLearner combining all of these learners. The package has three user functions: modest,
FVest and MLest. modest estimates only the model of the chosen Machine Learner. FVest takes
a model and gives the predicted fitted values for new features of choice. MLest combines both 
functions but only computes fitted values of the same observations used to build the model.
The package includes a survey with income and circumstances for Madrid in 2018 (from
Encuesta de Condiciones de vida (ECV) 2019).

## Installation

You can install the development version of ML from [GitHub](https://github.com/) with:
      
``` r
# install.packages("devtools")
devtools::install_github("joelters/ML")
```
Examples of the three functions are

``` r
X <- dplyr::select(mad2019,-Y)
Y <- mad2019$Y

m1 <- modest(X,Y,"RF")

m2 <- modest(X,Y,"SL",
        ensemble = c("SL.Lasso","SL.Ridge","SL.RF","SL.CIF","SL.XGB","SL.CB"))
      
FVs1 <- FVest(m1,X,Y,X[1:5,],Y[1:5],ML = "RF")

FVs2 <- FVest(m2,X,Y,ML = "SL")

m3 <- MLest(X,Y,"Lasso")

m4 <- MLest(X,Y,"SL",
        ensemble = c("SL.Lasso","SL.Ridge","SL.RF","SL.CIF","SL.XGB","SL.CB"))
```


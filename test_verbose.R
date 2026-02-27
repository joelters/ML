# Test the improved verbose output in MLtuning
library(ML)

# Load the data
data("mad2019")

# Use a small subset for quick testing
set.seed(123)
sample_idx <- sample(1:nrow(mad2019), 200)
data_subset <- mad2019[sample_idx, ]

# Prepare X and Y
Y <- data_subset$Y
X <- data_subset[, -which(names(data_subset) == "Y")]

# Test with Lasso (simple, fast method) with a small grid
cat("Testing Lasso with verbose=TRUE:\n")
cat("=====================================\n\n")

result_lasso <- MLtuning(
  X = X,
  Y = Y,
  ML = "Lasso",
  Kcv = 3,  # Only 3 folds for speed
  polynomial.Lasso.grid = c(1, 2),  # Only 2 combinations
  verbose = TRUE
)

cat("\n\nBest result:\n")
print(result_lasso$best.across.ml)

# Test with RF to see more complex output
cat("\n\n=====================================\n")
cat("Testing RF with verbose=TRUE:\n")
cat("=====================================\n\n")

result_rf <- MLtuning(
  X = X,
  Y = Y,
  ML = "RF",
  Kcv = 3,
  rf.cf.ntree.grid = c(100, 200),
  rf.depth.grid = c(2, Inf),
  mtry.grid = c(1, 3),
  verbose = TRUE
)

cat("\n\nBest result:\n")
print(result_rf$best.across.ml)

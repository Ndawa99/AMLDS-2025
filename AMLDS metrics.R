# Load required libraries
library(dplyr)
library(caret)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(faux)
library(kernlab)
library(naivebayes)
library(moments)
library(xtable)
library(doParallel)
library(nnet)

# Load custom metric functions
source("utils/auc.R")
source("utils/brier_score.R")
source("utils/mce_ece.R")

# Parameters
n <- 5000
g <- 10
seeds <- 123:152

# Beta distribution settings
distributions <- list(
  list(name = "uniform",    alpha = 1,   beta = 1),
  list(name = "asymmetric", alpha = 0.8, beta = 2.4),
  list(name = "bell",       alpha = 2.4, beta = 2.4),
  list(name = "u_shape",    alpha = 0.5, beta = 0.5)
)

# TrainControl settings
fitControl_none <- trainControl(method = "none", classProbs = TRUE, savePredictions = TRUE, allowParallel = TRUE)
fitControl_svm  <- trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = TRUE, allowParallel = TRUE)
fitControl_cv   <- trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = TRUE, allowParallel = TRUE)

# Models and tuning grids
models <- list(
  lr = list(method = "glm", control = fitControl_none),
  svm = list(method = "svmLinear2", control = fitControl_svm, tuneGrid = expand.grid(cost = c(0.001, 0.01, 0.1, 1))),
  random_forest = list(method = "rf", control = fitControl_cv, tuneGrid = expand.grid(mtry = c(2, 4, 6))),
  naivebayes = list(method = "naive_bayes", control = fitControl_cv, tuneGrid = expand.grid(usekernel = TRUE, laplace = c(0, 0.5, 1), adjust = c(0.75, 1, 1.25, 1.5))),
  neural_net = list(method = "nnet", control = fitControl_cv, tuneGrid = expand.grid(size = 3:10, decay = seq(0.1, 0.5, 0.1)))
)

# Log-loss function
log_loss <- function(actual, predicted) {
  eps <- 1e-9
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
}

# Parallel setup
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)
clusterEvalQ(cl, {
  library(dplyr)
  library(caret)
  library(randomForest)
  library(kernlab)
  library(naivebayes)
  library(nnet)
  library(faux)
})

# Global results
final_results <- list()

# Loop over distribution types
for (dist in distributions) {
  cat("\n=== Processing distribution:", dist$name, "===\n")
  all_metrics <- list()

  for (s in seeds) {
    set.seed(s)
    cat("  Seed:", s, "\n")
    
    # Generate data
    prob <- rbeta(n, dist$alpha, dist$beta)
    lin_pred <- qlogis(prob)
    y <- rbinom(n, 1, prob)
    x <- matrix(rnorm(n * 9), nrow = n)
    
    x[,1] <- rnorm_pre(lin_pred, r = 0.3)
    x[,2] <- rnorm_pre(lin_pred, r = -0.3)
    x[,3] <- rnorm_pre(lin_pred, r = -0.2)
    x[,4] <- rnorm_pre(lin_pred, r = 0.4)
    x[,5:6] <- apply(x[,5:6], 2, function(col) runif(n, min = -3, max = 3))
    coef <- c(2, -1, -1, 0.5, 3, -4, 1, -0.5, -3)
    x <- cbind(x, lin_pred - (x %*% coef))
    
    data <- as.data.frame(cbind(x, y))
    colnames(data)[ncol(data)] <- "y"
    
    # Split
    train_ind <- createDataPartition(data$y, p = 0.5, list = FALSE)
    train <- data[train_ind, ]
    test  <- data[-train_ind, ]

    results <- list()
    
    # Model loop
    for (model_name in names(models)) {
      cat("    Training:", model_name, "\n")
      m <- models[[model_name]]
      
      tryCatch({
        fit <- train(factor(y, labels = c("zero", "un")) ~ ., data = train,
                     method = m$method, trControl = m$control,
                     tuneGrid = m$tuneGrid, preProc = c("center", "scale"))
        
        p <- predict(fit, test[, 1:ncol(x)], type = "prob")[,2]
        
        # Create calibration groups
        mtx <- data.frame(y = test$y, prob = p, id = rownames(test))
        mtx <- mtx[order(mtx$prob), ]
        bins <- split(mtx, rep(1:ceiling(nrow(test)/g), each = nrow(test)/g, length.out = nrow(test)))
        moy_y <- unlist(lapply(bins, function(b) rep(mean(b$y), nrow(b))))
        mtx <- mtx[order(as.numeric(mtx$id)), ]
        test$C <- moy_y

        # Metrics
        results[[model_name]] <- list(
          fit = fit,
          p_valueks = ks.test(prob[-train_ind], p)$p.value,
          true_mse = mean((p - prob[-train_ind])^2),
          log_loss_true = log_loss(prob[-train_ind], p),
          epistemic_ll = log_loss(prob[-train_ind], p) - log_loss(prob[-train_ind], prob[-train_ind]),
          irreducible_ll = log_loss(test$y, prob[-train_ind]),
          calibration_ll = log_loss(test$C, p) - log_loss(test$C, test$C),
          refinement_ll = log_loss(test$y, test$C),
          log_loss_obs = log_loss(test$y, p),
          auc = auc(p, test$y),
          brier = brier_score(test$y, p),
          epistemic_bs = brier_score(prob[-train_ind], p) - brier_score(prob[-train_ind], prob[-train_ind]),
          irreducible_bs = brier_score(test$y, prob[-train_ind]),
          calibration_bs = brier_score(test$C, p) - brier_score(test$C, test$C),
          refinement_bs = brier_score(test$y, test$C),
          ece = ece_mce(test$y, p, g, method = 'C')$ece
        )
      }, error = function(e) {
        message("Error training ", model_name, ": ", e$message)
      })
    }
    
    # Format results
    if (length(results) > 0) {
      indicateurs_df <- do.call(rbind, lapply(names(results), function(name) {
        res <- results[[name]]
        data.frame(
          seed = s,
          model = name,
          p_valueks = res$p_valueks,
          True_MSE = res$true_mse,
          LL_p = res$log_loss_true,
          Epistemic_Loss_LL = res$epistemic_ll,
          Irreducible_Loss_LL = res$irreducible_ll,
          Calibration_Loss_LL = res$calibration_ll,
          Refinement_Loss_LL = res$refinement_ll,
          LL_y = res$log_loss_obs,
          ECE = res$ece,
          AUC = res$auc,
          Brier = res$brier,
          Epistemic_Loss_BS = res$epistemic_bs,
          Calibration_Loss_BS = res$calibration_bs,
          Refinement_Loss_BS = res$refinement_bs
        )
      }))
      all_metrics[[as.character(s)]] <- indicateurs_df
    }
  }

  # Combine all seeds
  if (length(all_metrics) > 0) {
    final_df <- do.call(rbind, all_metrics)
    final_results[[dist$name]] <- final_df
    
    # Save CSV summary
    pval_summary <- final_df %>%
      group_by(model) %>%
      summarise(count_pval_gt_005 = sum(p_valueks > 0.05),
                total = n(),
                proportion = count_pval_gt_005 / total)
    
    write.csv(pval_summary, paste0("output/pvalue_counts_", dist$name, ".csv"), row.names = FALSE)
    print(pval_summary)
  }
}

# Stop cluster
stopCluster(cl)

# Export LaTeX tables
for (prob_name in names(final_results)) {
  cat("\n=== Summary for", prob_name, "===\n")
  summary_metrics <- final_results[[prob_name]] %>%
    group_by(model) %>%
    summarise(across(where(is.numeric), list(mean = mean, sd = sd), .names = "{.col}_{.fn}"))
  
  formatted <- summary_metrics %>%
    rename_with(~gsub("_mean", "", .), ends_with("_mean")) %>%
    mutate(across(-model, ~sprintf("%.3f (%.3f)", ., summary_metrics[[paste0(cur_column(), "_sd")]]))) %>%
    select(-ends_with("_sd"))
  
  latex_table <- xtable(t(formatted))
  print(latex_table)
  save(list = "latex_table", file = paste0("output/latex_table_", prob_name, ".RData"))
}

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
start_time <- Sys.time()
# Load required libraries
library(dplyr)
library(ggplot2)
library(gridExtra)
library(caret)
library(faux)
library(kernlab)
library(randomForest)
library(naivebayes)
library(nnet)
library(moments)
library(xtable)
library(doParallel)

# Load custom metric functions
source("utils/auc.R")
source("utils/brier_score.R")
source("utils/mce_ece.R")

# Ensure output directory exists
if (!dir.exists("plots")) dir.create("plots")

# Global settings
n <- 5000
set.seed(123)
# Define model configurations
fitControl_none <- trainControl(method = "none", classProbs = TRUE, savePredictions = TRUE)
fitControl_svm  <- trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = TRUE)
fitControl_cv   <- trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = TRUE)

tuneGrid_svm <- expand.grid(cost = c(0.001, 0.01, 0.1, 1))
tuneGrid_rf  <- expand.grid(mtry = c(2, 4, 6))
tuneGrid_nb  <- expand.grid(usekernel = TRUE, laplace = c(0, 0.5, 1), adjust = c(0.75, 1, 1.25, 1.5))
tuneGrid_nn  <- expand.grid(size = 3:10, decay = seq(0.1, 0.5, 0.1))

models <- list(
  glm = list(method = "glm", control = fitControl_none),
  svm = list(method = "svmLinear2", control = fitControl_svm, tuneGrid = tuneGrid_svm),
  random_forest = list(method = "rf", control = fitControl_cv, tuneGrid = tuneGrid_rf),
  naivebayes = list(method = "naive_bayes", control = fitControl_cv, tuneGrid = tuneGrid_nb),
  neural_net = list(method = "nnet", control = fitControl_cv, tuneGrid = tuneGrid_nn, trace = FALSE)
)

# Define distribution settings
distributions <- list(
  u_shape  = list(alpha = 0.5, beta = 0.5),
  bell     = list(alpha = 2.4, beta = 2.4),
  asym     = list(alpha = 0.8, beta = 2.4),
  uniform  = list(alpha = 1,   beta = 1)
)
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
# Helper function to generate plots
generate_plots <- function(dist_name, alpha, beta) {
  prob <- rbeta(n, alpha, beta)
  lin_pred <- qlogis(prob)
  y <- rbinom(n, 1, prob)

  x <- matrix(rnorm(n * 9), nrow = n)
  x[,1] <- rnorm_pre(lin_pred, r = 0.3)
  x[,2] <- rnorm_pre(lin_pred, r = -0.3)
  x[,3] <- rnorm_pre(lin_pred, r = -0.2)
  x[,4] <- rnorm_pre(lin_pred, r = 0.4)
  x[,5]=runif(n,min=-3,max=3)
  x[,6]=runif(n,min=-3,max=3)
  coef=c(2,-1,-1,0.5,3,-4,1,-0.5,-3)
  x=cbind(x,lin_pred-(x%*%coef))
  data=as.data.frame(cbind(x,y))
  # Split
  train_ind <- createDataPartition(data$y, p = 0.5, list = FALSE)
  train <- data[train_ind, ]
  test  <- data[-train_ind, ]

  results <- list()

  for (model in names(models)) {
    cat("Training", model, "on", dist_name, "\n")
    m <- models[[model]]
    
    fit <- train(factor(y, labels = c("zero", "un")) ~ ., data = train,
                 method = m$method,
                 trControl = m$control,
                 tuneGrid = m$tuneGrid,
                 preProc = c("center", "scale"))

    p <- predict(fit, test[,1:ncol(x)], type = "prob")[,2]
    error <- prob[-train_ind] - p
    df <- data.frame(prob = prob[-train_ind], p = p, error = error)

    plot_hist <- ggplot(df, aes(x = p)) +
      geom_histogram(aes(y = after_stat(density)), bins = 100, fill = "blue", color = "black") +
      geom_density(aes(x = prob), color = "red", size = 1) +
      labs(
        title = bquote("Estimated Probabilities –" ~ .(model)),
        x = expression(hat(pi)),
        y = paste(dist_name, "-shaped density")
      ) +
      coord_cartesian(ylim = c(0, 6)) +
      theme_minimal()

    plot_error <- ggplot(df, aes(x = prob, y = error)) +
      geom_point(color = "blue", alpha = 0.5) +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      scale_y_continuous(limits = c(-0.6, 0.6)) +
      labs(
        title = bquote("Scatter plot of" ~ pi ~ "vs" ~ "|" ~ pi - hat(pi) ~ "|" ~ "–" ~ .(model)),
        x = expression(pi),
        y = paste(dist_name, "-error")
      ) +
      theme_minimal()

    results[[model]] <- list(hist = plot_hist, error = plot_error)
  }

  return(results)
}

# Generate and collect all plots
all_plots <- list()
for (dist in names(distributions)) {
  infos <- distributions[[dist]]
  all_plots[[dist]] <- generate_plots(dist, infos$alpha, infos$beta)
}

# Combine and save grid of histograms
hist_plots <- unlist(lapply(all_plots, function(r) lapply(r, function(m) m$hist)), recursive = FALSE)
png("plots/all_models_histograms.png", width = 16, height = 12, units = "in", res = 300)
grid.arrange(grobs = hist_plots, nrow = 4, ncol = length(models), top = "Estimated Probabilities – All Distributions & Models")
dev.off()

pdf("plots/all_models_histograms.pdf", width = 16, height = 12)
grid.arrange(grobs = hist_plots, nrow = 4, ncol = length(models), top = "Estimated Probabilities – All Distributions & Models")
dev.off()

# Combine and save grid of error scatter plots
error_plots <- unlist(lapply(all_plots, function(r) lapply(r, function(m) m$error)), recursive = FALSE)
png("plots/all_models_errors.png", width = 16, height = 12, units = "in", res = 300)
grid.arrange(grobs = error_plots, nrow = 4, ncol = length(models), top = "Prediction Errors – All Distributions & Models")
dev.off()

pdf("plots/all_models_errors.pdf", width = 16, height = 12)
grid.arrange(grobs = error_plots, nrow = 4, ncol = length(models), top = "Prediction Errors – All Distributions & Models")
dev.off()

end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "secs")
cat("Execution time:", 
    if (elapsed > 60) paste(round(as.numeric(elapsed)/60, 2), "minutes") else paste(round(elapsed, 2), "seconds"),
    "\n")
stopCluster(cl)

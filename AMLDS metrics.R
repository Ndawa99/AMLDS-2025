library(dplyr)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(caret)
library(faux)
library(kernlab)
library(naivebayes)
library(moments)
library(xtable)
#install.packages("doParallel")  # Si pas déjà installé
library(doParallel)
source('C:/Users/dieyen/OneDrive - LECNAM/Doctorat au Cnam/Thèse/Calibration coding/calibration/Fichiers simulation de base/auc.R')
source('C:/Users/dieyen/OneDrive - LECNAM/Doctorat au Cnam/Thèse/Calibration coding/calibration/Fichiers simulation de base/brier_score.R')
source('C:/Users/dieyen/OneDrive - LECNAM/Doctorat au Cnam/Thèse/Calibration coding/calibration/Fichiers simulation de base/mce_ece.R')
n <- 5000
g=10

#prob_asym

alpha_u = 0.5  
beta_u = 0.5
alpha_unif = 1
beta_unif = 1
alpha_asym = 0.8 
beta_asym = 2.4
alpha_c = 2.4  
beta_c = 2.4
seeds <- 123:152
all_metrics <- list()
results <- list()
g = 10

#Entraînement du vrai modèle
fitControl <- trainControl(method="none",classProbs = T, savePredictions=T,allowParallel = TRUE)
fitControl_svm <- trainControl(method = "cv", number=10,classProb=T,savePredictions=T,allowParallel = TRUE)
fitControl_cv <- trainControl(method = 'cv', number = 10,classProbs = T,savePredictions=T,allowParallel = TRUE)
#Optimisation des paramètres 
tuneGrid_svm <- expand.grid(cost = c(0.001, 0.01, 0.1, 1))    #Choix du paramètre à optimiser pour ce modèle
tuneGrid_rf <- expand.grid(mtry = c(2, 4, 6)) #Choix du paramètre à optimiser pour ce modèle
tuneGrid_nb <- expand.grid(usekernel = T,
                           laplace = c(0, 0.5, 1), 
                           adjust = c(0.75, 1, 1.25, 1.5))
tuneGrid_nn <- expand.grid(size = seq(from = 3, to = 10, by = 1),
                           decay = seq(from = 0.1, to = 0.5, by = 0.1))


#Liste des modèles à entraîner
models <- list(
  lr = list(method = "glm", control = fitControl, tuneGrid=NULL),
  svm = list(method = "svmLinear2", control = fitControl_svm,tuneGrid=tuneGrid_svm),
  random_forest = list(method = "rf", control = fitControl_cv, tuneGrid=tuneGrid_rf),
  naivebayes= list(method = "naive_bayes", control = fitControl_cv, tuneGrid=tuneGrid_nb),
  neural_net= list(method = "nnet", control = fitControl_cv, tuneGrid=tuneGrid_nn)
)

#Apprentissage de tous les modèles que l'on stocke pour garder les résultats 
log_loss <- function(actual, predicted) {
  eps <- 0.000000001  # Pour éviter log(0)
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  loss <- - mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
  
  return(loss)
}
# Liste des types de probabilités à tester
prob_types <- list(
  #list(name = "uniforme", alpha = alpha_unif, beta = beta_unif),
  #list(name = "asymetrique", alpha = alpha_asym, beta = beta_asym),
 # list(name = "cloche", alpha = alpha_c, beta = beta_c),
  list(name = "u_shape", alpha = alpha_u, beta = beta_u)
)

# Initialisation des résultats finaux
final_results <- list()
# Détecte le nombre de cœurs disponibles
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)
clusterEvalQ(cl, {
  library(dplyr)
  library(randomForest)
  library(ggplot2)
  library(gridExtra)
  library(caret)
  library(faux)
  library(kernlab)
  library(naivebayes)
  library(moments)
  library(xtable)
})
all_metrics <- list()


# Boucle sur les types de probabilité
for(prob_type in prob_types) {
  cat("\n=== Processing probability type:", prob_type$name, "===\n")
  # Boucle sur les seeds
  for (s in seeds) {
    set.seed(s)
    cat("\n--- Processing seed", s, "---\n")
    
    # Generate data with current probability type
    prob <- rbeta(n, shape1 = prob_type$alpha, shape2 = prob_type$beta)
    lin_pred <- qlogis(prob)
    y <- rbinom(n, 1, prob)
    x = matrix(data=rnorm(n*9),nrow=n)
    x[,1]=rnorm_pre(lin_pred,r=0.3)
    x[,2]= rnorm_pre(lin_pred,r=-0.3)
    x[,3]= rnorm_pre(lin_pred,r=-0.2)
    x[,4]= rnorm_pre(lin_pred,r=0.4)
    x[,5]=runif(n,min=-3,max=3)
    x[,6]=runif(n,min=-3,max=3)
    coef=c(2,-1,-1,0.5,3,-4,1,-0.5,-3)
    x=cbind(x,lin_pred-(x%*%coef))
    data=as.data.frame(cbind(x,y))
    
    # Split data
    train_ind <- createDataPartition(data$y, p=0.5, list=FALSE)
    train <- data[train_ind, ]
    test <- data[-train_ind, ]
    
    # Initialize results for this seed
    results <- list()
    
    # Model training loop
    for (model in names(models)) {
      cat("  Training model", model, "...")
      
      tryCatch({
        model_info <- models[[model]]
        
        fit <- train(factor(y, labels=c("zero","un")) ~ ., 
                     data = train, 
                     preProc = c("center", "scale"),
                     method = model_info$method, 
                     trControl = model_info$control, 
                     tuneGrid = model_info$tuneGrid)
        
        # Predictions
        p <- predict(fit, test[,1:ncol(x)], type = "prob")[, 2]
        
        # Calculate metrics
        mtx <- data.frame(y = test$y, prob = p, id=rownames(test))
        mtx <- mtx[order(mtx$prob), ]
        split_mtx <- split(mtx, rep(1:ceiling(nrow(test)/g), each = nrow(test)/g, length.out = nrow(test)))
        moy_y <- sapply(split_mtx, function(group) mean(group$y))
        moy_y_vector <- unlist(lapply(split_mtx, function(group) rep(mean(group$y), nrow(group))))
        mtx$moy_y <- moy_y_vector
        mtx <- mtx[order(as.numeric(mtx$id)), ]
        test$C <- mtx$moy_y
        
        results[[model]] <- list(
          fit = fit,
          kolmogrov_test = ks.test(prob[-train_ind], p)$p.value,
          true_mse = mean((p - prob[-train_ind])^2),
          ll_p = log_loss(prob[-train_ind], p),
          true_ll_epistemic_loss = log_loss(prob[-train_ind], p) - log_loss(prob[-train_ind], prob[-train_ind]),
          irreducible_loss_ll = log_loss(test$y, prob[-train_ind]),
          calibration_loss_ll = log_loss(test$C, p) - log_loss(test$C, test$C),
          refinement_loss_ll = log_loss(test$y, test$C),
          ll_y = log_loss(test$y, p),
          cab_large = mean(test$y)/mean(p),
          auc = auc(p, test$y),
          brier_score = brier_score(test$y, p),
          bs_epistemic_loss = brier_score(prob[-train_ind], p) - brier_score(prob[-train_ind], prob[-train_ind]),
          irreducible_loss_bs = brier_score(test$y, prob[-train_ind]),
          calibration_loss_bs = brier_score(test$C, p) - brier_score(test$C, test$C),
          refinement_loss_bs = brier_score(test$y, test$C),
          ece = ece_mce(test$y, p, g, 'C')$ece
        )
        
        cat(" success\n")
      }, error = function(e) {
        cat(" failed:", e$message, "\n")
      })
    }
    
    # Store results for this seed
    if (length(results) > 0) {
      indicateurs_df <- do.call(rbind, lapply(names(results), function(model_name) {
        res <- results[[model_name]]
        data.frame(
          seed = s,
          model = model_name,
          p_valueks = res$kolmogrov_test,
          True_MSE = res$true_mse,
          LL_p = res$ll_p,
          Epistemic_Loss_LL = res$true_ll_epistemic_loss,
          Irreducible_Loss_LL = res$irreducible_loss_ll,
          Calibration_Loss_LL = res$calibration_loss_ll,
          Refinement_Loss_LL = res$refinement_loss_ll,
          LL_y = res$ll_y,
          ECE = res$ece,
          AUC = res$auc,
          Brier = res$brier_score,
          Epistemic_Loss_BS = res$bs_epistemic_loss,
          Calibration_Loss_BS = res$calibration_loss_bs,
          Refinement_Loss_BS = res$refinement_loss_bs
        )
      }))
      
      all_metrics[[as.character(s)]] <- indicateurs_df
    }
  }
  
  # Clean up
  stopCluster(cl)
  
  # Stocker les résultats pour ce type de probabilité
  if (length(all_metrics) > 0) {
    final_df <- do.call(rbind, all_metrics)
    final_results[[prob_type$name]] <- final_df
    
    # Calculer le nombre de p-values > 0.05 pour chaque modèle
    pvalue_counts <- final_df %>%
      group_by(model) %>%
      summarise(count_pval_gt_005 = sum(p_valueks > 0.05),
                total = n(),
                proportion = count_pval_gt_005 / total)
    
    cat("\n=== Results for", prob_type$name, "===\n")
    print(pvalue_counts)
    
    # Sauvegarder les résultats dans un fichier
    write.csv(pvalue_counts, paste0("pvalue_counts_", prob_type$name, ".csv"), row.names = FALSE)
  } else {
    cat("No results for probability type:", prob_type$name, "\n")
  }
}

# Afficher les résultats finaux
for(prob_name in names(final_results)) {
  cat("\n=== Summary for", prob_name, "===\n")
  
  summary_metrics <- final_results[[prob_name]] %>%
    group_by(model) %>%
    summarise(across(where(is.numeric), list(mean = mean, sd = sd), .names = "{.col}_{.fn}"))
  
  print(summary_metrics)
  
  # Formater le tableau pour LaTeX
  formatted_table <- summary_metrics %>%
    select(
      model,
      ends_with("_mean") | ends_with("_sd")
    ) %>%
    rename_with(~gsub("_mean", "", .), ends_with("_mean"))
  
  for(metric in setdiff(colnames(formatted_table), "model")) {
    if(endsWith(metric, "_sd")) next
    sd_col <- paste0(metric, "_sd")
    formatted_table[[metric]] <- sprintf(
      "%.3f (%.3f)", 
      formatted_table[[metric]], 
      formatted_table[[sd_col]]
    )
  }
  
  formatted_table <- formatted_table %>%
    select(!ends_with("_sd"))
  
  # Générer la table LaTeX
  latex_table <- xtable(t(formatted_table))
  print(latex_table)
  
  # Sauvegarder dans un fichier
  assign(paste0("latex_table_", prob_name), latex_table)
  save(list = paste0("latex_table_", prob_name), 
       file = paste0("latex_table_", prob_name, ".RData"))
}

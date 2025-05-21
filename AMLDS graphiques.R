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

source('C:/Users/dieyen/OneDrive - LECNAM/Doctorat au Cnam/Thèse/Calibration coding/calibration/Fichiers simulation de base/auc.R')
source('C:/Users/dieyen/OneDrive - LECNAM/Doctorat au Cnam/Thèse/Calibration coding/calibration/Fichiers simulation de base/brier_score.R')
source('C:/Users/dieyen/OneDrive - LECNAM/Doctorat au Cnam/Thèse/Calibration coding/calibration/Fichiers simulation de base/mce_ece.R')
set.seed(123)
n <- 10000
g=10

#Prob_en_u

alpha_u = 0.5  
beta_u = 0.5

#***********************Vrai modèle*************************************#
prob_en_u <- rbeta(n, shape1 = alpha_u, shape2 = beta_u)
lin_pred <- qlogis(prob_en_u)

y <- rbinom(n, 1, prob_en_u)

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

#Séparation en train et test
train_ind <- createDataPartition(data$y,p=0.5,list=F)
notrain_ind <- setdiff(seq_len(n),train_ind)
valid_test_ind <- createDataPartition(data$y[notrain_ind], p=0.5, list=F)
valid_ind <- notrain_ind[valid_test_ind]
test_ind <- setdiff(notrain_ind, valid_ind)
train <- data[train_ind, ]
validation <- data[valid_ind,]
test <- data[test_ind, ]
nb_test <-n-length(valid_ind)-length(train_ind)


#Entraînement du vrai modèle
fitControl <- trainControl(method="none",classProbs = T, savePredictions=T)
fitControl_svm <- trainControl(method = "cv", number=10,classProb=T,savePredictions=T)
fitControl_cv <- trainControl(method = 'cv', number = 10,classProbs = T,savePredictions=T)
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
  neural_net= list(method = "nnet", control = fitControl_cv, tuneGrid=tuneGrid_nn, trace = FALSE)
)

#Apprentissage de tous les modèles que l'on stocke pour garder les résultats 
results_u <- list()
results_a <- list()
results_c <- list()
results_unif <- list()

# Entraînement des modèles avec caret
for (model in names(models)) {
  set.seed(123)
  model_info <- models[[model]]
  
  
  # Entraînement du modèle avec la fonction train de caret
  fit <- train(factor(y,labels=c("zero","un"))~ ., 
               data = train, 
               preProc = c("center", "scale"),
               method = model_info$method, 
               trControl = model_info$control, 
               tuneGrid=model_info$tuneGrid
  )
  # Prédiction sur le test set
  p <- predict(fit, test[,1:ncol(x)], type = "prob")[, 2]
  error <- prob_en_u[test_ind]-p
    #Histogramme des vraies probabilités
  df=data.frame(prob=prob_en_u[test_ind], p=p, error=error)
  
  # Histogram of estimated probabilities
  plot_pred_u <- ggplot(df, aes(x = p)) +
    geom_histogram(aes(y = after_stat(density)), bins = 100, fill = "blue", color = "black") +
    geom_density(aes(x = prob), color = "red", size = 1) +
    labs(
      title = bquote("Estimated Probabilities –" ~ .(model)),
      x = expression(hat(pi)),
      y = "U-shaped estimation density"
    ) +
    coord_cartesian(ylim = c(0, 6)) +
    theme_minimal()
  plot_pred_error_u <- ggplot(df, aes(x = prob, y = error)) +
    geom_point(color = "blue", alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    scale_y_continuous(limits = c(-0.6,0.6)) +
    labs(
      title = bquote("Scatter plot of" ~ hat(pi) ~ "vs" ~ "|"~ pi~"-"~hat(pi)~"|" ~ "–" ~ .(model)),
      x = expression(hat(pi)),
      y = "u-error"
    ) +
    theme_minimal()
  
  results_u[[model]]$plot_pred_u <- plot_pred_u
  results_u[[model]]$plot_pred_error_u <- plot_pred_error_u
  
  
  
  

}



#Prob_en_cloche
set.seed(123)
alpha_c = 2.4  
beta_c = 2.4
#***********************Vrai modèle*************************************#
prob_cloche <- rbeta(n, shape1 = alpha_c, shape2 = beta_c)
lin_pred <- qlogis(prob_cloche)
# Ici on choisit le y qu'on va utiliser dans la suite
  y <- rbinom(n, 1, prob_cloche)  #y issue de nos vraies probabilités

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

#Séparation en train et test
train_ind <- createDataPartition(data$y,p=0.5,list=F)
notrain_ind <- setdiff(seq_len(n),train_ind)
valid_test_ind <- createDataPartition(data$y[notrain_ind], p=0.5, list=F)
valid_ind <- notrain_ind[valid_test_ind]
test_ind <- setdiff(notrain_ind, valid_ind)
train <- data[train_ind, ]
validation <- data[valid_ind,]
test <- data[test_ind, ]
nb_test <-n-length(valid_ind)-length(train_ind)


#Entraînement du vrai modèle
fitControl <- trainControl(method="none",classProbs = T, savePredictions=T)
fitControl_svm <- trainControl(method = "cv", number=10,classProb=T,savePredictions=T)
fitControl_cv <- trainControl(method = 'cv', number = 10,classProbs = T,savePredictions=T)
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
  neural_net= list(method = "nnet", control = fitControl_cv, tuneGrid=tuneGrid_nn,trace=F)
)



# Entraînement des modèles avec caret
for (model in names(models)) {
  set.seed(123)
  model_info <- models[[model]]
  
  
  # Entraînement du modèle avec la fonction train de caret
  fit <- train(make.names(y)~ ., 
               data = train, 
               preProc = c("center", "scale"),
               method = model_info$method, 
               trControl = model_info$control, 
               tuneGrid=model_info$tuneGrid
  )
  # Prédiction sur le test set
  p <- predict(fit, test[,1:ncol(x)], type = "prob")[, 2]
  error <- prob_cloche[test_ind]-p
  #Histogramme des vraies probabilités
  df=data.frame(prob=prob_cloche[test_ind], p=p, error=error)
  
  # Histogram of estimated probabilities
  plot_pred_c <- ggplot(df, aes(x = p)) +
    geom_histogram(aes(y = after_stat(density)),bins = 100, color = "black", fill = "blue") +
    geom_density(aes(x = prob), color = "red", size = 1) +
    labs(
      title = bquote("Estimated Probabilities –" ~ .(model)),
      x = expression(hat(pi)),
      y = "bell-shaped estimation density"
    ) +
    coord_cartesian(ylim = c(0, 6)) +
    theme_minimal()
  plot_pred_error_c <- ggplot(df, aes(x = prob, y = error)) +
    geom_point(color = "blue", alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") + 
    scale_y_continuous(limits = c(-0.6,0.6)) +
    labs(
      title = bquote("Scatter plot of" ~ hat(pi) ~ "vs" ~ "|"~ pi~"-"~hat(pi)~"|" ~ "–" ~ .(model)),
      x = expression(hat(pi)),
      y = "bell-error"
    ) +
    theme_minimal()
  
  results_c[[model]]$plot_pred_c <- plot_pred_c
  results_c[[model]]$plot_pred_error_c <- plot_pred_error_c
  
  
}

#---------------------------------------------prob asym-------------------------------------------------------------------
set.seed(123)
alpha_asym = 0.8 #SD 0.2113 odds 1/3
beta_asym = 2.4

prob_asym <- rbeta(n, shape1 = alpha_asym, shape2 = beta_asym)
lin_pred <- qlogis(prob_asym)
# Ici on choisit le y qu'on va utiliser dans la suite
  y <- rbinom(n, 1, prob_asym)  #y issue de nos vraies probabilités

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

#Séparation en train et test
train_ind <- createDataPartition(data$y,p=0.5,list=F)
notrain_ind <- setdiff(seq_len(n),train_ind)
valid_test_ind <- createDataPartition(data$y[notrain_ind], p=0.5, list=F)
valid_ind <- notrain_ind[valid_test_ind]
test_ind <- setdiff(notrain_ind, valid_ind)
train <- data[train_ind, ]
validation <- data[valid_ind,]
test <- data[test_ind, ]
nb_test <-n-length(valid_ind)-length(train_ind)


#Entraînement du vrai modèle
fitControl <- trainControl(method="none",classProbs = T, savePredictions=T)
fitControl_svm <- trainControl(method = "cv", number=10,classProb=T,savePredictions=T)
fitControl_cv <- trainControl(method = 'cv', number = 10,classProbs = T,savePredictions=T)
#Optimisation des paramètres 
tuneGrid_svm <- expand.grid(cost = c(0.001, 0.01, 0.1, 1))
tuneGrid_rf <- expand.grid(mtry = c(2, 4, 6)) #Choix du paramètre à optimiser pour ce modèle
tuneGrid_nb <- expand.grid(usekernel = T,
                           laplace = c(0, 0.5, 1), 
                           adjust = c(0.75, 1, 1.25, 1.5))
tuneGrid_nn <- expand.grid(size = seq(from = 3, to = 10, by = 1),
                           decay = seq(from = 0.1, to = 0.5, by = 0.1))

#Liste des modèles à entraîner
models <- list(
  glm = list(method = "glm", control = fitControl, tuneGrid=NULL),
  svm = list(method = "svmLinear2", control = fitControl_svm,tuneGrid=tuneGrid_svm),
  random_forest = list(method = "rf", control = fitControl_cv, tuneGrid=tuneGrid_rf),
  naivebayes= list(method = "naive_bayes", control = fitControl_cv, tuneGrid=tuneGrid_nb),
  neural_net= list(method = "nnet", control = fitControl_cv, tuneGrid=tuneGrid_nn,trace=F)
)

#Apprentissage de tous les modèles que l'on stocke pour garder les résultats 


# Entraînement des modèles avec caret
for (model in names(models)) {
  set.seed(123)
  model_info <- models[[model]]
  
  
  # Entraînement du modèle avec la fonction train de caret
  fit <- train(make.names(y)~ ., 
               data = train, 
               preProc = c("center", "scale"),
               method = model_info$method, 
               trControl = model_info$control, 
               tuneGrid=model_info$tuneGrid
  )
  # Prédiction sur le test set
  p <- predict(fit, test[,1:ncol(x)], type = "prob")[, 2]
  
  error <- prob_asym[test_ind]-p
  #Histogramme des vraies probabilités
  df=data.frame(prob=prob_asym[test_ind], p=p, error=error)
  
  #Histogramme des probabilités estimées par la modèle
  plot_pred_a <- ggplot(df, aes(x = p)) +
    geom_histogram(aes(y = after_stat(density)),bins = 100, color = "black", fill = "blue") +
    geom_density(aes(x = prob), color = "red", size = 1) +
    labs(
      title = bquote("Estimated Probabilities –" ~ .(model)),
      x = expression(hat(pi)),
      y = "asymmetric-shaped estimation density"
    ) +
    coord_cartesian(ylim = c(0, 6)) +
    theme_minimal()
  plot_pred_error_a <- ggplot(df, aes(x = prob, y = error)) +
    geom_point(color = "blue", alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") + 
    scale_y_continuous(limits = c(-0.6,0.6)) +
    labs(
      title = bquote("Scatter plot of" ~ hat(pi) ~ "vs" ~ "|"~ pi~"-"~hat(pi)~"|" ~ "–" ~ .(model)),
      x = expression(hat(pi)),
      y = "asymmetric-error"
    ) +
    theme_minimal()
  results_a[[model]]$plot_pred_a <- plot_pred_a
  results_a[[model]]$plot_pred_error_a <- plot_pred_error_a
  
  
}

#-----------------------uniforme--------------------------------------------------------------------------------#
set.seed(123)
alpha_unif = 1
beta_unif = 1

prob_unif=rbeta(n, shape1 = alpha_unif, shape2 = beta_unif)
lin_pred <- qlogis(prob_unif)
y <- rbinom(n, 1, prob_unif)  #y issue de nos vraies probabilités

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

#Séparation en train et test
train_ind <- createDataPartition(data$y,p=0.5,list=F)
notrain_ind <- setdiff(seq_len(n),train_ind)
valid_test_ind <- createDataPartition(data$y[notrain_ind], p=0.5, list=F)
valid_ind <- notrain_ind[valid_test_ind]
test_ind <- setdiff(notrain_ind, valid_ind)
train <- data[train_ind, ]
validation <- data[valid_ind,]
test <- data[test_ind, ]
nb_test <-n-length(valid_ind)-length(train_ind)


#Entraînement du vrai modèle
fitControl <- trainControl(method="none",classProbs = T, savePredictions=T)
fitControl_svm <- trainControl(method = "cv", number=10,classProb=T,savePredictions=T)
fitControl_cv <- trainControl(method = 'cv', number = 10,classProbs = T,savePredictions=T)
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
  glm = list(method = "glm", control = fitControl, tuneGrid=NULL),
  svm = list(method = "svmLinear2", control = fitControl_svm,tuneGrid=tuneGrid_svm),
  random_forest = list(method = "rf", control = fitControl_cv, tuneGrid=tuneGrid_rf),
  naivebayes= list(method = "naive_bayes", control = fitControl_cv, tuneGrid=tuneGrid_nb),
  neural_net= list(method = "nnet", control = fitControl_cv, tuneGrid=tuneGrid_nn,trace=F)
)

#Apprentissage de tous les modèles que l'on stocke pour garder les résultats 
results <- list()


# Entraînement des modèles avec caret
for (model in names(models)) {
  set.seed(123)
  model_info <- models[[model]]
  
  
  # Entraînement du modèle avec la fonction train de caret
  fit <- train(make.names(y)~ ., 
               data = train, 
               preProc = c("center", "scale"),
               method = model_info$method, 
               trControl = model_info$control, 
               tuneGrid=model_info$tuneGrid
  )
  # Prédiction sur le test set
  p <- predict(fit, test[,1:ncol(x)], type = "prob")[, 2]
  error <- prob_unif[test_ind]-p
  #Histogramme des vraies probabilités
  df=data.frame(prob=prob_unif[test_ind], p=p, error=error)
  
  #Histogramme des probabilités estimées par la modèle
  plot_pred_unif <- ggplot(df, aes(x = p)) +
    geom_histogram(aes(y = after_stat(density)),bins = 100, color = "black", fill = "blue") +
    geom_density(aes(x = prob), color = "red", size = 1) +
    labs(
      title = bquote("Estimated Probabilities – Model" ~ .(model)),
      x = expression(hat(pi)),
      y = "uniform-shaped estimation density"
    ) +
    coord_cartesian(ylim = c(0, 6)) +
    theme_minimal()
  plot_pred_error_unif <- ggplot(df, aes(x = prob, y = error)) +
    geom_point(color = "blue", alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") + 
    scale_y_continuous(limits = c(-0.6,0.6)) +
    labs(
      title = bquote("Scatter plot of" ~ hat(pi) ~ "vs" ~ "|"~ pi~"-"~hat(pi)~"|" ~ "–" ~ .(model)),
      x = expression(hat(pi)),
      y = "uniform-error"
    ) +
    theme_minimal()
  results_unif[[model]]$plot_pred_unif <- plot_pred_unif
  results_unif[[model]]$plot_pred_error_unif <- plot_pred_error_unif
  
  
}

hist_u<- lapply(names(results_u), function(model_name) results_u[[model_name]]$plot_pred_u)
hist_a<- lapply(names(results_a), function(model_name) results_a[[model_name]]$plot_pred_a)
hist_c<- lapply(names(results_c), function(model_name) results_c[[model_name]]$plot_pred_c)
hist_unif<- lapply(names(results_unif), function(model_name) results_unif[[model_name]]$plot_pred_unif)
grid.arrange(
  grobs = c(hist_u,hist_a, hist_c,hist_unif),
  nrow = 4,
  ncol = 5,
  top = ""
)

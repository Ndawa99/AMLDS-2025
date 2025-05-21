auc <- function(predictions, true_labels) {
  # Séparer les prédictions en fonction des vrais labels
  neg_preds <- predictions[true_labels == 0]
  pos_preds <- predictions[true_labels == 1]
  
  # Calculer la somme des indicateurs pour chaque paire de prédictions
  sum_indicators <- sum(outer(neg_preds, pos_preds, function(x, y) as.integer(x < y)))+(1/2)*sum(outer(neg_preds, pos_preds, function(x, y) as.integer(x == y)))
  auc <- sum_indicators / (length(neg_preds) * length(pos_preds))

  # Retourner l'AUC
  return(auc)
}


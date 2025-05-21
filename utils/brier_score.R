brier_score <- function(predictions, actual) {
  return(mean((predictions - actual)^2))
}

brier_reliability <- function (predictions, actual){
  decomp_matrix = BrierDecomp(predictions, actual, bins = 10, bias.corrected = FALSE)
  Reliability= decomp_matrix[1,1]
  return(Reliability)
}

brier_resolution <- function (predictions, actual){
  decomp_matrix = BrierDecomp(predictions, actual, bins = 10, bias.corrected = FALSE)
  Resolution= decomp_matrix[1,2]
  return(Resolution)
}

brier_uncertainty <- function (predictions, actual){
  decomp_matrix = BrierDecomp(predictions, actual, bins = 10, bias.corrected = FALSE)
  uncertainty= decomp_matrix[1,3]
  return(uncertainty)
}

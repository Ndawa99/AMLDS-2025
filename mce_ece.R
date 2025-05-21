ece_mce = function(y, prob, g, stat_type){
  mtx = cbind(y, y_not = 1- y, prob, prob_not = 1-prob)
  mtx = as.data.frame(mtx)
  mtx = mtx[order(mtx$prob),]
  n <- length(prob)/g
  nr <- nrow(mtx)
  
  if (stat_type == 'C'){
    split_mtx = split(mtx, rep(1:ceiling(nr/n), each=n, length.out=nr))
    taille=n
  }else{ ### H statistics, equal intervals
    split_mtx = split(mtx, cut(mtx$prob, seq(0,1,1/g), include.lowest=TRUE))
    split_mtx = split_mtx[sapply(split_mtx, nrow)>0]

  }
  

  H_stat = c()
  taux=c()
  estimee=c()
  for (i in 1:length(split_mtx)){
    obs = mean(split_mtx[[i]]$y == 1)
    #exp = sum(split_mtx[[i]][split_mtx[[i]]$y == 1,]$prob)
    exp = mean(split_mtx[[i]]$prob)

    
    
    
    
    H_stat = c(H_stat, abs(obs - exp))
    taux=rbind(taux,obs)
    estimee=c(estimee,exp)
  }
  return(list(taille=taille, ece = sum(H_stat)/length(split_mtx), mce = max(H_stat), taux=taux,estimee=estimee,split=split_mtx))
}
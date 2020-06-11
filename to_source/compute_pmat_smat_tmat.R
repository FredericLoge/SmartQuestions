# parsing the state vector X (partial or full) into a one-dim value
foo_id <- function(x){
  paste0(x, collapse = '')
}

#' Preparing *pmat*, expecting binary X ! 
#' @param p dimension of X
#' @param x_parsed observed (X_1, X_2, ..., X_p), parsed in single element
compute_pmat <- function(p, x_parsed, use_ucb = FALSE){
  
  # probability matrix of each state
  cmd <- paste0('pmat <- expand.grid(', paste0(rep('c(0,1)', p), collapse = ', '), ', KEEP.OUT.ATTRS = FALSE)')
  eval(parse(text = cmd))
  colnames(pmat) <- paste0('X', 1:p)
  str(pmat)
  
  # add column with parsed vector X, which will serve as an ID
  pmat$PARSED_X <- apply(X = pmat[,1:p], MARGIN = 1, FUN = foo_id)
  
  # compute probability based on the data observed
  pmat$p <- NA
  for(i in 1:nrow(pmat)){
    pmat$p[i] <- mean(pmat$PARSED_X[i] == x_parsed)
    # if(use_ucb){
    #   pmat$p[i] <- pmat$p[i] + 2 * sqrt(pmat$p[i] * (1 - pmat$p[i]) / min(1, sum(pmat$PARSED_X[i] == x_parsed)))
    # }
  }
  if(use_ucb){
    pmat$p <- pmax(pmat$p, min(pmat$p[pmat$p > 0]))
  }
  
  # normalize
  pmat$p <- pmat$p / sum(pmat$p)
  
  # add unique ID
  pmat$ID <- 1:nrow(pmat)
  
  return(pmat)
  
}


#' Preparing *smat*, expecting binary X ! 
#' @param p dimension of X
compute_smat <- function(p, pmat, y_pred, x, y){
  
  # small checks
  stopifnot(is.matrix(y_pred))
  stopifnot(nrow(y_pred) == nrow(pmat))
  stopifnot(ncol(pmat) >= p)
  
  # state matrix
  cmd <- paste0('smat <- expand.grid(', paste0(rep('c(-1, 0, 1)', p), collapse = ', '), ', KEEP.OUT.ATTRS = FALSE)')
  eval(parse(text = cmd))
  colnames(smat) <- paste0('X', 1:p)

  # keep q first layers of tree (this will change depending on the stopping criterion)
  smat$depth_in_tree <- apply(X = (smat[,1:p] != -1), MARGIN = 1, FUN = sum, na.rm = TRUE)
  smat <- smat[smat$depth_in_tree <= q,]
  
  # check if node is terminal (this will also change depending on the stopping criterion, and will 
  # actually require us to compute the accuracy in order to now if we are yet to stop)
  smat$is_terminal <- (smat$depth_in_tree == q)
  
  # parse X
  smat$PARSED_X <- apply(X = smat[,1:p], MARGIN = 1, FUN = foo_id)
  
  # add ID
  smat$ID <- 1:nrow(smat)

  # add probability, predicted Y and Reward
  smat$p <- NA
  smat$Y_PRED <- NA
  smat$PRED_ERROR <- NA
  smat$REWARD <- 0
  
  for(i in 1:nrow(smat)){
    
    ###
    ### find x which are neighbours of state line i
    ###
    
    # extract state information
    si <- as.numeric(smat[i,1:p])
    
    # identify X candidates
    cond <- rep(TRUE, nrow(pmat))
    for(j in 1:p){
      if(si[j] != -1){
        cond <- cond & (pmat[,j] == si[j])
      }
    }
    index <- pmat$ID[cond]
    
    ###
    ### state probability : sum of candidate x
    ###
    
    smat$p[i] <- sum(pmat$p[pmat$ID %in% index])
    
    ###
    ### if state is terminal : compute prediction for each x candidate and compute reward
    ###
    ### /!\ Need to look at the prediction error : not done here !!
    
    #
    my_pred <- y_pred[pmat$ID %in% index,]
    my_prob <- pmat$p[pmat$ID %in% index]
    proba_pred <- apply(X = my_pred, MARGIN = 2, FUN = function(x){ sum(x * my_prob)} )
    proba_pred <- as.numeric(proba_pred / sum(my_prob) )
    smat$Y_PRED[i] <- which.max(proba_pred)
    
    # identify Xi candidates
    xi_cond <- rep(TRUE, nrow(x))
    for(j in 1:p){
      if(si[j] != -1){
        xi_cond <- xi_cond & (x[,j] == as.numeric(si[j]))
      }
    }
    smat$PRED_ERROR[i] <- sum( (smat$Y_PRED[i] != y) & xi_cond ) / sum( xi_cond )
    
  }
  
  smat$REWARD[smat$is_terminal] <- (- smat$PRED_ERROR[smat$is_terminal])
  
  return(smat)
  
}

compute_tmat <- function(p, smat){
  
  # expand grid
  tmat <- expand.grid('s0' = smat$ID[smat$is_terminal == FALSE], 'a0' = 1:p, 'answer' = c(0,1))
  
  # add 's1' and check if 'a0' is not already asked
  tmat$s1 <- -1
  tmat$r <- NA
  a0_already_asked <- rep(TRUE, nrow(tmat))
  for(i in 1:nrow(tmat)){
    a0_already_asked[i] <- (smat[smat$ID == tmat$s0[i], tmat$a0[i]] != -1)
    if(a0_already_asked[i] == FALSE){
      ref <- smat[smat$ID == tmat$s0[i],1:p]
      ref[tmat$a0[i]] <- tmat$answer[i]
      cond <- (smat$PARSED_X == foo_id(ref))
      if(sum(cond) > 1) stop('More than one match is not expected. Error generated in compute_tmat()')
      tmat$s1[i] <- smat$ID[cond]
      tmat$r[i] <- smat$REWARD[cond]
      tmat$p[i] <- smat$p[cond] / smat$p[smat$ID == tmat$s0[i]]
    }
  }
  
  # erase cases where questions where already asked
  tmat <- tmat[a0_already_asked == FALSE,]
  
  return(tmat)
  
}

#' Preparing link matrix between IDs of pmat and smat where X is neighbour of S
#' @param p dimension of X
#' @param pmat Matrix pmat previously computed
#' @param smat Matrix smat previously computed
deprecated_compute_link_smat_pmat <- function(smat, pmat, p){
  
  # find link between partial and full information
  index_smat_pmat <- NULL
  for(i in 1:nrow(smat)){
    # extract state information
    si <- as.numeric(smat[i,1:p])
    # identify coordinates where 
    si_known <- which(si != -1)
    if(length(si_known) == 0){
      v <- cbind(smat$ID[i], pmat$ID)
    }else if(length(si_known) == 1){
      cond <- (pmat[,si_known] == si[si_known])
      v <- cbind(smat$ID[i], pmat$ID[cond])  
    }else{
      cond <- (apply(X = pmat[,si_known], MARGIN = 1, FUN = foo_id) == foo_id(x = si[si_known]))
      v <- cbind(smat$ID[i], pmat$ID[cond])  
    }
    index_smat_pmat <- rbind(index_smat_pmat, v)
  }
  colnames(index_smat_pmat) <- c('smat_ID', 'pmat_ID')
  
  return(index_smat_pmat)
  
}



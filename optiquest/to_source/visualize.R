build_grViz_cmd <- function(p, mdp, states){
  
  # create nice label for the state information
  LABEL_QUESTIONS <- paste0('Q', 1:p)
  states$label <- ""
  for(i in 1:nrow(states)){
    si <- states[i,1:p]  
    si_known <- (states[i,1:p] != - 1)
    if(sum(si_known) == 0) states$label[i] <- 'No information.'
    else states$label[i] <- paste0('(', paste0(LABEL_QUESTIONS[si_known], ':', si[si_known], collapse = ', '), ')')
  }
  
  # construct state and action labels, must be contained within quotes !
  ## s0_ids <- paste0('\'', states$label, '\'')
  s0_ids <- (paste0('\'', states$label, '\n Y^ = ', states$Y_PRED, '\n Err = ', round(states$PRED_ERROR, 2), '\''))
  s0_ids[states$depth_in_tree == 0] <- '\'No information.\''
  a0_ids <- paste0('Q', 1:p)
  
  # create (s0, a0, s1)
  s0_a0_s1 <- lapply(X = states$ID[states$depth_in_tree < q], FUN = function(s){
    a0 <- states$recommended_action[states$ID == s]
    cbind('s0' = s, 'a0' = a0, tmat[tmat$s0 == s & tmat$a0 == a0, c('s1', 'p')])
  })
  s0_a0_s1 <- do.call(rbind.data.frame, s0_a0_s1)
  
  # find s0 which are on optimal path, we assume the root is 1
  acceptable_s0 <- 1 
  for(i in 1:(q-1)){
    acceptable_s0 <- c(acceptable_s0, s0_a0_s1$s1[s0_a0_s1$s0 %in% acceptable_s0])
  }
  acceptable_s0 <- sort(unique(acceptable_s0))
  
  # filter from the initial list (s0, a0, s1)
  s0_a0_s1 <- s0_a0_s1[s0_a0_s1$s0 %in% acceptable_s0,]
  str(s0_a0_s1)
  
  # create unique (s0,a0) ids
  s0_a0_ids <- paste0("\'", gsub("'", "", s0_ids[s0_a0_s1$s0]), ':', a0_ids[s0_a0_s1$a0], "\'")
  unique_s0_a0_ids <- s0_a0_ids[seq(1, nrow(s0_a0_s1)-1, by = 2)]
  unique_s0_a0_labels <- a0_ids[s0_a0_s1$a0][seq(1, nrow(s0_a0_s1)-1, by = 2)]
  
  # create unique ids linking s0 to (s0,a0)
  s0_to_s0_a0_ids <- unique(paste0(s0_ids[s0_a0_s1$s0], '->', s0_a0_ids))
  
  # create ids linking (s0,a0) to s1
  s0_a0_to_s1_ids <- paste0(s0_a0_ids, '->', s0_ids[s0_a0_s1$s1])
  s0_a0_to_s1_col <- rgb(maxColorValue = 1, red = s0_a0_s1$p, green = 0, blue = 0)
  
  #
  cmd <- paste0("
                digraph boxes_and_circles {
                
                # a 'graph' statement
                graph [overlap = false, fontsize = 14]
                
                # several 'node' statements (STATES)
                node [shape = circle,
                fontname = Helvetica]
                ", paste0(unique_s0_a0_ids, collapse = '; '), "
                
                # ACTIONS !! 
                node [shape = box,
                fixedsize = false,
                width = 0.9] // sets as circles
                ", paste0(unique_s0_a0_ids, '[label = ', unique_s0_a0_labels, ']', collapse = '; '), "
                
                # several 'edge' statements
                edge [color = grey]
                ", paste0(s0_to_s0_a0_ids, '[color = "', "blue" , '"]', collapse = ' '), "
                ", paste0(s0_a0_to_s1_ids, '[color = "', s0_a0_to_s1_col, '"]', collapse = ' '), "
                }
                ")
  return(cmd)
  
}

#' 
#' @param qsa_mat
reframe_qsa_mat <- function(qsa, smat, row_normalized){
  
  qsa_mat <- qsa[rowSums(!(is.na(qsa) & is.nan(qsa))) > 1,]
  if(row_normalized){
    for(i in 1:nrow(qsa_mat)){
      qsa_mat[i,] <- qsa_mat[i,] / sum(abs(qsa_mat[i,]), na.rm = TRUE)
    }
  }
  
  # transform to dataframe
  temp <- data.frame(qsa_mat)
  n <- nrow(temp)
  p <- ncol(temp)
  
  # give name to columns
  colnames(temp) <- paste0('A', 1:p)
  
  # add column with state index
  LABEL_QUESTIONS <- paste0('Q', 1:p)
  temp$state <- ""
  for(i in 1:nrow(smat)){
    si <- smat[i,1:p]  
    si_known <- (smat[i,1:p] != - 1)
    if(sum(si_known) == 0) temp$state[i] <- 'No information.'
    else temp$state[i] <- paste0('(', paste0(LABEL_QUESTIONS[si_known], ':', si[si_known], collapse = ', '), ')')
  }
  
  # add depth of state
  temp$depth_in_tree <- smat$depth_in_tree
    
  # keep rows where at least one element is not NA
  temp <- temp[rowSums(is.na(temp[,1:p]))<p,]
  
  # melt dataframe
  temp <- data.table::melt(temp, id.vars = c('state', 'depth_in_tree'))
  
  # erase missing data (questions already asked)
  temp <- temp[!is.na(temp$value),]
  
  # return it
  return(temp)
  
}




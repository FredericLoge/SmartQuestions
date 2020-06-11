
#' computing the state-action value functions for some policy pi in input
#' @param pi_mat
#' @param df
#' @param adj
#' @param states
#' @param gamma
#' @param qsa_diff_tol
compute_qsa_based_on_pi <- function(pi_mat, df, adj, states, gamma, qsa_diff_tol = 1e-10 ){
  max_nb_runs <- 1e04
  n <- nrow(pi_mat)
  p <- ncol(pi_mat)
  # initialize qsa and vs for each moment order
  qsa <- array(data = 0, dim = c(n,p))
  for(j in 1:p){
    qsa[adj[,j]==0,j] <- NA
  }
  
  vs <- array(data = 0, dim = n)
  # boolean controlling the stopping of algo
  has_not_converged <- TRUE
  ##
  k <- 0
  while(has_not_converged){
    qsa_old <- qsa
    k = k + 1
    for(i in 1:n){
      ## why is it not allowed for terminal states ?
      if(states$is_terminal[i] == FALSE){
        j_is_adj <- which(adj[i,] == 1)
        vs[i] <- sum( pi_mat[i,j_is_adj] * qsa[i,j_is_adj] )
      }
    }
    for(i in 1:n){
      if(states$is_terminal[i] == FALSE){
        for(a in 1:p){
          if(adj[i,a] == 1){
            j <- which(df$s0 == i & df$a0 == a)
            ptr <- df$p[j]
            i_prim <- df$s1[j]
            qsa[i,a] <- sum(ptr * (df$r[j] + gamma * vs[i_prim])) / sum(ptr)
          }
        }
      }
    }
    has_not_converged <- ( max(abs(qsa - qsa_old), na.rm = TRUE) > qsa_diff_tol )
    if(k > max_nb_runs) has_not_converged <- FALSE
  }
  # return result
  return(qsa)
}

#' initialize with uniformly random policy
#' @param adj
init_pi <- function(adj){
  #
  pi_mat <- 0*adj
  #
  for(i in 1:nrow(pi_mat)){
    a_possible <- which(adj[i,] == 1)
    pi_mat[i, a_possible] <- 1/length(a_possible)
  }
  return(pi_mat)
}

#' get greedy
#' @param x
greedy <- function(x){
  v <- which(x == max(x))
  if(length(v)>1) v <- base::sample(x = v, size = 1)
  return(v)
}
# not run :
# greedy(x = c(0, 6, 2))
# greedy(c(4,3,6))

#' @param df : formatted dataframe column names :
#' - s0 : integer index for state S_0
#' - a0 : integer index for action A_0
#' - s1 : integer index for state S_1
#' - p  : transition probability P(S_1 | S_0, A_0)
#' - r  : expected reward associated to transition (S_0, A_0, S_1)
#' some remarks : ideally, the integer indexes should start from 1 to the number of states / action
#' the set of indexes used for s0 and s1 should match obviously !
#' @param gamma
#' @param qsa_diff_tol
dp <- function(df, gamma = 1, qsa_diff_tol = 1e-4, max_nb_iter = 1e4){
  
  # get number of states and actions
  nb_states <- max(c(df$s0, df$s1))
  nb_actions <- max(df$a0)

  # compute adjacency matrix between states and actions, based on df  
  couples_allowed <- unique(df[,c('s0', 'a0')])
  adj = array(0, c(nb_states, nb_actions))
  for(i in 1:nb_states){
    aa <- couples_allowed$a0[couples_allowed$s0 == i]
    if(length(aa) > 0){
      adj[i,aa] <- 1
    }
  }
  
  # initialize state-action value matrix Q(s,a)
  # with NA reward for unauthorized pairs and 0 otherwise
  qsa_mat = array(0, dim = c(nb_states, nb_actions))
  for(j in 1:nb_actions){
    qsa_mat[adj[,j]==0,j] <- NA
  }
  
  # create a dataframe indicating for each state whether or not it is terminal
  states <- data.frame('s0' = 1:nb_states)
  states$is_terminal <- (!states$s0 %in% df$s0)
  
  # while loop booleans
  max_nb_iter_not_reached <- TRUE
  has_not_converged <- TRUE
  
  # iteration counter
  count <- 0
  
  ## 
  while(has_not_converged & max_nb_iter_not_reached){
    
    # copy old estimate Q(s,a)
    old_qsa_mat <- qsa_mat
    
    # establish \pi(s,a) based on Q(s,a)
    if(count == 0){
      # initialize decision matrix by method init_pi
      pi_mat <- init_pi(adj)
    }else{
      # otherwise, act greedily
      pi_mat <- 0*adj
      #
      for(i in 1:nrow(pi_mat)){
        temp <- qsa_mat[i,]
        if(any(!is.na(temp))){
          a_optimal <- which(temp == max(temp, na.rm = TRUE))
          pi_mat[i, a_optimal] <- 1/length(a_optimal)
        }
      }
    }
    
    # compute new estimate of Q(s,a) based on \pi(s,a)
    qsa_mat <- compute_qsa_based_on_pi(pi_mat, df, adj, states, gamma = gamma)
    
    # check for loop exit
    has_not_converged <- ( max(abs(qsa_mat - old_qsa_mat), na.rm = TRUE) > qsa_diff_tol )
    max_nb_iter_not_reached <- ( count < max_nb_iter )
    
    # iterate counter
    count <- count + 1
    
  }
  
  return(qsa_mat)
  
}

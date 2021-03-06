##
## TOY MODEL #3
##


rm(list = ls())

##
## LOAD LIBRARIES
##

library(pROC)
library(DiagrammeR)
library(rpart)
library(rpart.plot)
library(randomForest)
library(keras)
library(tensorflow)
library(mlbench)
library(tidyverse)
library(mvtnorm)

##
## READ DATA: boston housing dataset, split in four samples, with similar Y distribution 
##

list_of_seeds <- c(1234, 1734, 5932, 2953, 7437, 4964, 2631, 1953, 3843, 2632)
global_results <- array(data = NA, dim = c(length(list_of_seeds), 4))

for(iter in 1:length(list_of_seeds)){
  
  cat('\n', iter)
  set.seed(list_of_seeds[iter])
  
  n <- 6000
  p <- 8
  # u <- matrix(runif(p^2)*2-1, ncol=p) 
  # SIGMA <- t(u) %*% u
  SIGMA <- diag(rep(1,p))
  x <- rmvnorm(n = n, mean = rep(0,p), sigma = SIGMA)
  image(cor(x))
  library(corrplot)
  corrplot(cor(x))
  set.seed(143253)
  beta1 <- rep(0, p-1) # rnorm(n = p-1, mean = 0, sd = 5) * (2 * (runif(n = p-1) > 0.5) - 1)
  beta1[1:2] <- 1
  beta2 <- rep(0, p-1) # rnorm(n = p-1, mean = 0, sd = 5) * (2 * (runif(n = p-1) > 0.5) - 1)
  beta2[3:4] <- 1
  s <- x[,-1] %*% beta1
  s[x[,1] > 0] <- x[x[,1] > 0,-1] %*% beta2
  SNR <- 1 # 10
  e <- rnorm(n = n, mean = 0, sd = sqrt(2 / SNR))
  y <- s + e
  n_train <- n*2/3
  colnames(x) <- paste0('X', 1:p)
  complete_df <- data.frame(y, x)
  
  ##
  ## SOURCE IMPORTANT FUNCTIONS
  ##
  
  #
  source('foo_dqn_specifics.R')
  
  #
  random_mask <- function(x, l, p, w = rep(1, ncol(x))){
    if(l == 0){
      res <- cbind(array(data = 0, dim = dim(x)), array(data = 0, dim = dim(x)))
      return(res)
    }
    x_mask <- array(data = 0, dim = dim(x))
    for(i in 1:nrow(x)){
      sel <- sample(x = 1:p, size = l, replace = FALSE, prob = w)
      x[i,-sel] <- 0
      x_mask[i,sel] <- 1
    }
    colnames(x_mask) <- paste0("is_masked_", colnames(x))
    return(cbind(x, x_mask))
  }
  
  # 
  get_reward <- function(prediction, true_value){
    r <- - (prediction - true_value)^2
    return(r)
  }
  
  # ...
  which_max_otherwise_runif <- function(x){
    max_x <- max(x, na.rm = FALSE)
    index_max_x <- which(x == max_x)
    if(length(index_max_x) == 1) return(index_max_x)
    return(sample(x = index_max_x, size = 1))
  }
  
  # sample validation indexes (specific to neural network section)
  n_valid <- floor(n_train / 2)
  validation_indexes <- 1:n_valid
  training_indexes <- (n_valid+1):n_train
  length(training_indexes)
  
  ##
  ## DEFINE GENERIC PREDICTOR: \tilde{\mathcal{X}} \rightarrow \mathcal{Y} 
  ##
  
  # random forest
  rf0 <- randomForest(y ~ ., data = complete_df[training_indexes,])
  varImpPlot(rf0)
  as.numeric(rf0$importance)
  
  #
  testing_indexes <- (n_train+1):nrow(x)
  rf1 <- randomForest(y ~ ., data = complete_df[training_indexes,], mtry = p)
  rf1_errors <- (complete_df$y[testing_indexes] - predict(rf1, complete_df[testing_indexes,]))^2
  summary(rf1_errors)
  paste0(round(summary(rf1_errors), 2), collapse = ' & ')
  
  # # interesting to compare to this:
  # beta_named <- beta
  # names(beta_named) <- paste0('X', 1:p)
  # beta_named[order(abs(beta_named), decreasing = TRUE)]
  
  ##
  ## DEFINE GENERIC PREDICTOR #2: \tilde{\mathcal{X}} \rightarrow \mathcal{Y} 
  ##
  
  # 
  potential_questions <- expand.grid(1:p, 1:p, 1:p)
  potential_questions <- potential_questions %>% filter(Var1 < Var2 & Var2 < Var3)
  nrow(potential_questions) == (factorial(p) / (factorial(3) * factorial(p-3)))
  
  #
  rf_training_indexes <- training_indexes
  set_of_rf <- list()
  pb <- txtProgressBar(min = 0, max = nrow(potential_questions))
  for(i in 1:nrow(potential_questions)){
    setTxtProgressBar(pb = pb, value = i)
    colindexes <- c(1, 1 + as.numeric(potential_questions[i,1:3]))
    tmp_rf <- randomForest(y ~ ., data = complete_df[rf_training_indexes, colindexes], ntree = 100, mtry = 3, sampsize = 1500)
    set_of_rf[[i]] <- tmp_rf 
  }
  nb_trees <- 100
  summary( sapply(set_of_rf, function(x){ x$rsq[nb_trees] }) )
  hist(sapply(set_of_rf, function(x){ x$rsq[nb_trees] }), breaks = 20)
  potential_questions$rsq <- sapply(set_of_rf, function(x){ x$rsq[nb_trees] }) 
  potential_questions$set_of_rf_index <- 1:nrow(potential_questions)
  potential_questions <- potential_questions %>% arrange(desc(rsq)) 
  potential_questions_summary <- sapply(X = 1:p, FUN = function(j){
    summary(which(potential_questions$Var1 == j | potential_questions$Var2 == j | potential_questions$Var3 == j))
  })
  colnames(potential_questions_summary) <- colnames(x)
  potential_questions_summary <- potential_questions_summary[,order(potential_questions_summary['Median',])]
  boxplot(potential_questions_summary)
  head(potential_questions)
  colnames(x)[as.numeric(potential_questions[1,1:3])]
  
  # TO SAVE
  # potential_questions_summary, 
  # potential_questions
  
  #
  get_prediction <- function(si_prime){
    pr <- rep(NA, nrow(si_prime))
    for(it in 1:nrow(si_prime)){
      index_var <- which(si_prime[it,p + 1:p] == 1)
      if(length(index_var) != 3){
        pr[it] <- 0
      }else{
        si_prime_it_ <- si_prime[it, index_var]
        index <- with(potential_questions,
                      which(Var1 == index_var[1] & 
                              Var2 == index_var[2] &
                              Var3 == index_var[3]))
        index <- potential_questions$set_of_rf_index[index]
        pr[it] <- as.numeric(predict(set_of_rf[[index]], t(si_prime_it_)))
      }
    }
    return(pr)
  }
  
  ##
  ## DATA SAMPLES
  ##
  
  # this set should be definetely different from the one used to build random forest,
  # especially for the last layer, where we look at the performance, so overfitting is
  # not permitted ... 
  indexes_to_train_mask <- validation_indexes
  
  # hello :)
  masked_by_layer <- list()
  for(layer in 0:2){
    row_indexes <- x_unmasked <- x_masked <- y_masked <- NULL
    if(layer == 0){
      row_indexes <- indexes_to_train_mask
      y_masked <- y[indexes_to_train_mask]
      x_unmasked <- x[indexes_to_train_mask,]
      x_masked <- array(data = 0, dim = c(length(indexes_to_train_mask), 2*p))
    }
    if(layer == 1){
      row_indexes <- x_unmasked <- x_masked <- y_masked <- NULL
      for(first_action_picked in 1:p){
        row_indexes <- c(row_indexes, indexes_to_train_mask)
        y_masked <- c(y_masked, y[indexes_to_train_mask])
        x_unmasked <- rbind(x_unmasked, x[indexes_to_train_mask,])
        x_masked_tmp <- array(data = 0, dim = c(length(indexes_to_train_mask), 2*p))
        x_masked_tmp[, first_action_picked] <- x[indexes_to_train_mask, first_action_picked]
        x_masked_tmp[, first_action_picked+p] <- 1
        x_masked <- rbind(x_masked, x_masked_tmp)
      }
    }
    if(layer == 2){
      row_indexes <- x_unmasked <- x_masked <- y_masked <- NULL
      action_sequence <- expand.grid("first_action_picked" = 1:p, "second_action_picked" = 1:p)
      action_sequence <- action_sequence[action_sequence$first_action_picked < action_sequence$second_action_picked,]
      for(action_sequence_row_index in 1:nrow(action_sequence)){
        actions_picked <- unique(as.numeric(action_sequence[action_sequence_row_index,]))
        row_indexes <- c(row_indexes, indexes_to_train_mask)
        y_masked <- c(y_masked, y[indexes_to_train_mask])
        x_unmasked <- rbind(x_unmasked, x[indexes_to_train_mask,])
        x_masked_tmp <- array(data = 0, dim = c(length(indexes_to_train_mask), 2*p))
        x_masked_tmp[, actions_picked] <- x[indexes_to_train_mask, actions_picked]
        x_masked_tmp[, p + actions_picked] <- 1
        x_masked <- rbind(x_masked, x_masked_tmp)
      }
    }
    colnames(x_unmasked) <- colnames(x)
    colnames(x_masked) <- c(colnames(x), paste0('is_unmasked_', colnames(x)))
    masked_by_layer[[length(masked_by_layer) + 1]] <- list('row' = row_indexes, 'x_unmasked' = x_unmasked, 'x_masked' = x_masked, 'y' = y_masked)
  }
  ### str(masked_by_layer, 2)
  
  # penalty of re-asking a question
  penalty_question_already_asked <- (0)
  
  # for each possible action, evaluate long-term reward
  layer <- 2
  si <- masked_by_layer[[layer + 1]]$x_masked
  colMeans(si == 0)
  xi <- masked_by_layer[[layer + 1]]$x_unmasked
  yi <- masked_by_layer[[layer + 1]]$y
  reward_last_layer <- array(data = NA, dim = c(nrow(si), p))
  for(action_picked in 1:p){
    
    # generate following state
    si_prime <- si
    si_prime[,action_picked] <- xi[,action_picked] # reveal value
    si_prime[,p + action_picked] <- 1 # indicate it is now known
    
    # generate reward, penalty if question was already asked
    reward <- rep(0, nrow(xi))
    cond_question_already_asked <- (si[,action_picked+p] == 1)
    reward[cond_question_already_asked] <- penalty_question_already_asked * 4000
    
    # for the last transition, add to reward the final accuracy metric ; and for others add the Q-values
    
    # get prediction from following state
    pred_si_prime <- rep(0, nrow(si_prime))
    masked_pasted <- apply(si_prime[,p + 1:p], 1, paste0, collapse = '')
    u_masked_pasted <- unique(masked_pasted)
    pb <- txtProgressBar(min = 0, max = length(u_masked_pasted))
    for(it in 1:length(u_masked_pasted)){
      setTxtProgressBar(pb = pb, value = it)
      index_row <- which(masked_pasted == u_masked_pasted[it])
      index_var <- which(si_prime[index_row[1],p + 1:p] == 1)
      if(length(index_var) != 3){
        pred_si_prime[index_row] <- 0
        reward[index_row] <- penalty_question_already_asked * 4000
      }else{
        si_prime_it_ <- si_prime[index_row, index_var]
        index <- with(potential_questions,
                      which(Var1 == index_var[1] & 
                              Var2 == index_var[2] &
                              Var3 == index_var[3]))
        index <- potential_questions$set_of_rf_index[index]
        pred_si_prime[index_row] <- as.numeric(predict(set_of_rf[[index]], si_prime_it_))
      }
    }
    
    # get reward associated
    pred_reward <- get_reward(prediction = pred_si_prime, true_value = yi)
    
    # add to reward vector
    reward_last_layer[,action_picked] <- reward + pred_reward
    
  }
  hist(reward_last_layer, breaks = 200)
  
  head(si[2501:2550,])
  head(reward_last_layer[2501:2550,])
  
  #
  hist(c(reward_last_layer)/4000, breaks = 200)
  reward_last_layer_truncated <- reward_last_layer / 4000
  hist(c(reward_last_layer_truncated), breaks = 200)
  
  ##
  ## DEFINE NEURAL NETWORKS
  ##
  
  # dimensions
  input_size <- 3*p
  output_size <- 1
  
  # from state of layer 2 + action to final reward || architecture + network instanciation + compilation
  network2 <- keras_model_sequential() %>%
    layer_dense(units = 3*input_size, kernel_initializer = "uniform", input_shape = input_size, activation = "relu") %>%
    layer_dense(units = 3*input_size, kernel_initializer = "uniform", activation = "relu") %>%
    layer_dense(units = input_size, kernel_initializer = "uniform", activation = "relu") %>%
    layer_dense(units = input_size, kernel_initializer = "uniform", activation = "relu") %>%
    layer_dense(units = output_size, kernel_initializer = "uniform", activation = "relu")
  network2 %>% compile(optimizer = optimizer_rmsprop(lr = 1e-4), loss = "mse")
  network2$optimizer$get_config()
  
  # from state of layer 1 + action to best reward from layer 2 || architecture + network instanciation + compilation
  network1 <- keras_model_sequential() %>%
    layer_dense(units = 3*input_size, kernel_initializer = "uniform", input_shape = input_size, activation = "relu") %>%
    layer_dense(units = 3*input_size, kernel_initializer = "uniform", activation = "relu") %>%
    layer_dense(units = input_size, kernel_initializer = "uniform", activation = "relu") %>%
    layer_dense(units = input_size, kernel_initializer = "uniform", activation = "relu") %>%
    layer_dense(units = output_size, kernel_initializer = "uniform", activation = "relu")
  network1 %>% compile(optimizer = optimizer_rmsprop(lr = 1e-4), loss = "mse")
  network1$optimizer$get_config()
  
  # callbacks
  fit_callbacks <- list(
    callback_early_stopping(monitor = "val_loss", min_delta = 1e-4, patience = 25, verbose = 0, mode = "auto"),
    callback_model_checkpoint(filepath = ".mdl_wts.hdf5", monitor = "val_loss", verbose = 0, save_best_only = TRUE, mode = "min"),
    callback_reduce_lr_on_plateau(monitor = "val_loss", min_delta = 1e-4, factor = 0.9, patience = 10, mode = "auto")
  )
  
  ##
  ## TRAIN NEURAL NETWORK 2
  ##
  
  # take masked x from training set and reward data
  initial_state <- masked_by_layer[[3]]$x_masked
  target <- (+1 * sqrt( - reward_last_layer))
  hist(target)
  
  # reframe it by action
  initial_state_complete <- NULL
  target_complete <- NULL
  for(j in 1:p){
    action_mat <- array(data = 0, dim = c(nrow(initial_state), p))
    action_mat[,j] <- 1
    colnames(action_mat) <- paste0('A',1:p)
    initial_state_complete <- rbind(initial_state_complete, cbind(initial_state, action_mat))
    target_complete <- c(target_complete, target[,j])
  }
  
  # getting rid of cases which don't have three questions asked, gives us less stuff to learn ...
  count_by_coordinate <- rowSums(initial_state_complete[,p+ 1:p] + initial_state_complete[,2*p+ 1:p] >= 1)
  initial_state_complete <- initial_state_complete[count_by_coordinate == 3,]
  target_complete <- target_complete[count_by_coordinate == 3]
  
  # checking out progress in view panel
  options(keras.view_metrics = TRUE)
  
  # getting rid of cases which don't have three questions asked, gives us less stuff to learn ...
  shuffled_indexes <- sample(nrow(initial_state_complete))
  initial_state_complete_shuffled <- initial_state_complete[shuffled_indexes,]
  target_complete_shuffled <- target_complete[shuffled_indexes]
  
  # proper_selection <- initial_state_complete[,p+5]==1 & initial_state_complete[,p+6]==1
  
  # repeat until convergence
  network2 %>% fit(x = initial_state_complete_shuffled, y = target_complete_shuffled, epochs = 500, batch_size = 256, validation_split = 0.5, verbose = 0, callbacks = fit_callbacks)
  
  
  ##
  ## TRAIN NEURAL NETWORK 1
  ##
  
  #
  s0 <- masked_by_layer[[2]]$x_masked
  x0 <- masked_by_layer[[2]]$x_unmasked
  state_compiled <- target_compiled <- NULL
  for(action_picked in 1:p){
    s1 <- s0
    s1[,action_picked] <- x0[,action_picked]
    s1[,action_picked+p] <- 1
    pre <- sapply(X = 1:p, FUN = function(j){
      action_mat <- array(data = 0, dim = c(nrow(s1), p))
      action_mat[,j] <- 1
      pmax( predict(object = network2, x = cbind(s1, action_mat)), 0)
    })
    target <- apply(X = pre, MARGIN = 1, min)
    action_mat <- array(data = 0, dim = c(nrow(s0), p))
    action_mat[,action_picked] <- 1
    state_compiled <- rbind(state_compiled, cbind(s0, action_mat))
    target_compiled <- c(target_compiled, target)  
  }
  
  # getting rid of cases which don't have three questions asked, gives us less stuff to learn ...
  count_by_coordinate <- rowSums(state_compiled[,p+ 1:p] + state_compiled[,2*p+ 1:p] >= 1)
  table(count_by_coordinate)
  state_compiled <- state_compiled[count_by_coordinate == 2,]
  target_compiled <- target_compiled[count_by_coordinate == 2]
  
  # getting rid of cases which don't have three questions asked, gives us less stuff to learn ...
  shuffled_indexes <- sample(nrow(state_compiled))
  state_compiled_shuffled <- state_compiled[shuffled_indexes,]
  target_compiled_shuffled <- target_compiled[shuffled_indexes]
  
  
  #
  network1 %>% fit(x = state_compiled_shuffled, y = target_compiled_shuffled, epochs = 250, batch_size = 128, validation_split = 0.5, verbose = 0, callbacks = fit_callbacks)
  
  ##
  ## FINALIZING
  ##
  
  #
  s0 <- masked_by_layer[[1]]$x_masked
  x0 <- masked_by_layer[[1]]$x_unmasked
  target_compiled_0 <- NULL
  for(action_picked in 1:p){
    s1 <- s0
    s1[,action_picked] <- x0[,action_picked]
    s1[,action_picked+p] <- 1
    pre <- sapply(X = 1:p, FUN = function(j){
      action_mat <- array(data = 0, dim = c(nrow(s1), p))
      action_mat[,j] <- 1
      predict(object = network1, x = cbind(s1, action_mat))
    })
    target <- apply(X = pre, MARGIN = 1, min)
    target_compiled_0 <- cbind(target_compiled_0, target)  
  }
  colnames(target_compiled_0) <- paste0('A', 1:p)
  summary(target_compiled_0)
  sort(colMeans(target_compiled_0))
  
  ##
  ## TRYING OUT ON VALIDATION DATA
  ##
  
  #
  testing_indexes <- (n_train+1):nrow(x)
  x_valid <- x[testing_indexes,]
  states <- array(data = 0, dim = c(nrow(x_valid), 2*p))
  colnames(states) <- c(colnames(x), paste0('is_unmasked_', colnames(x)))
  
  # fill in first answer 
  first_action <- as.numeric(which.min(colMeans(target_compiled_0)))
  states[,first_action] <- x_valid[,first_action]
  states[,p+first_action] <- 1
  head(states)
  
  # then select next action!
  pr <- sapply(X = 1:p, FUN = function(j){
    action_mat <- array(data = 0, dim = c(nrow(states), p))
    action_mat[,j] <- 1
    predict(object = network1, x = cbind(states, action_mat))
  })
  pr[,first_action] <- pr[,first_action] + 10
  pr_sel <- apply(X = pr, MARGIN = 1, FUN = which.min)
  table(pr_sel)
  # pr_sel <- rep(6, nrow(states))
  table(pr_sel[states[,1]<0])
  table(pr_sel[states[,1]>0])
  
  # fill in required answers
  for(j in 1:p){
    cond <- (pr_sel == j)
    states[cond,j] <- x_valid[cond,j]
    states[cond,j+p] <- 1
  }
  head(states)
  
  # then select last action!
  pr2 <- sapply(X = 1:p, FUN = function(j){
    action_mat <- array(data = 0, dim = c(nrow(states), p))
    action_mat[,j] <- 1
    predict(object = network2, x = cbind(states, action_mat))
  })
  # pr2 <- (-1) * pr2
  pr2_sel <- rep(NA, nrow(states))
  for(it in 1:nrow(states)){
    xi <- pr2[it,]
    xi <- (xi == min(xi[- c(first_action, pr_sel[it])]))
    index <- which(xi)
    index <- index[!index %in% c(first_action, pr_sel[it])]
    if(length(index) > 1) index <- sample(x = index, size = 1)
    pr2_sel[it] <- index
  }
  table(pr2_sel)
  ## pr2_sel <- rep(3, nrow(states))
  
  # fill in required answers
  for(j in 1:p){
    cond <- (pr2_sel == j)
    states[cond,j] <- x_valid[cond,j]
    states[cond,p+j] <- 1
  }
  
  # hello there.
  selected_questions <- data.frame(
    'Q1' = as.numeric(first_action),
    'Q2' = pr_sel,
    'Q3' = pr2_sel
  )
  sort(table(apply(selected_questions, 1, function(x){ paste0(sort(x), collapse = ' - ') })))
  final_pred <- get_prediction(si_prime = states)
  
  rmse <- function(y, yhat){
    sqrt(mean( (y - yhat)^2 ))
  }
  
  #
  baseline_pred <- predict(set_of_rf[[potential_questions$set_of_rf_index[1]]], x[testing_indexes,])
  
  #
  library(rpart)
  rp_control <- rpart.control(maxdepth = 3)
  rp0 <- rpart(formula = y ~., data = complete_df[training_indexes,], control = rp_control)
  rpart_pred <- predict(rp0, complete_df[testing_indexes,])
  
  rf0_pred <- predict(rf0, complete_df[testing_indexes,])
  
  # save results :)
  global_results[iter,] <- c(
    rmse(y = y[testing_indexes], yhat = rf0_pred),
    rmse(y = y[testing_indexes], yhat = final_pred),
    rmse(y = y[testing_indexes], yhat = baseline_pred),
    rmse(y = y[testing_indexes], yhat = rpart_pred)
  )
  
  
}

#
colnames(global_results) <- c('rf_oracle', 'our_method', 'best_subset', 'tree')
resres <- apply(global_results, 2, function(x){ c(mean(x), sd(x)) })
paste0(round(resres[1,],2), ' (', round(resres[2,],2), ')', collapse = ' & ')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                         LOAD LIBRARIES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# rpart, random forests
library(rpart)
library(rpart.plot)
library(randomForest)

# neural networks
library(keras)
library(tensorflow)

# tidyverse operations
library(tidyverse)

# 
library(gridExtra)
library(ggplot2)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                               FEW FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#' @title Prediction error, computed for couple (prediction value, true value)
#' @param predicted_value
#' @param true_value
prediction_error <- function(predicted_value, true_value){
  return( (predicted_value - true_value)^2 )
}

#' @title Get prediction for a state variable (V2, works faster)
#' @param si_prime
get_prediction <- function(si_prime){
  pred_si_prime <- rep(0, nrow(si_prime))
  masked_pasted <- apply(si_prime[,p + 1:p], 1, paste0, collapse = '')
  u_masked_pasted <- unique(masked_pasted)
  pb <- txtProgressBar(min = 0, max = length(u_masked_pasted))
  for(it in 1:length(u_masked_pasted)){
    setTxtProgressBar(pb = pb, value = it)
    index_row <- which(masked_pasted == u_masked_pasted[it])
    index_var <- which(si_prime[index_row[1],p + 1:p] == 1)
    if(length(index_var) != 3){
      pred_si_prime[index_row] <- NA
    }else{
      si_prime_it_ <- si_prime[index_row, index_var]
      index <- with(potential_questions,
                    which(Var1 == index_var[1] & 
                            Var2 == index_var[2] &
                            Var3 == index_var[3]))
      index <- potential_questions$set_of_rf_index[index]
      # if(length(index_row) == 1){
      #   si_prime_it_ <- t(si_prime_it_)
      # }
      pred_si_prime[index_row] <- predict(set_of_rf[[index]], si_prime_it_)
    }
  }
  return(pred_si_prime)
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                           LOAD AND PREP DATA
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# read training dataset
library(mlbench)
data("BostonHousing")
wine <- BostonHousing

# generate x
x <- wine[,1:13]
x$chas <- as.numeric(as.character(x$chas))
x <- scale(x)

# generate y
y <- wine$medv
hist(y)

# re-compile (y,x) and stratify three batches by y variable
complete_df <- data.frame('y' = y, x)

list_of_seeds <- c(9010, 2026, 2052, 6511, 9993, 1796, 6606, 7547, 9617, 6751)
global_results <- array(data = NA, dim = c(length(list_of_seeds), 5))

rmse <- function(y, yhat){
  sqrt(mean( (y  - yhat)^2 ))  
}

for(iter in (1:10)){
  
  cat('\n', iter)
  set.seed(list_of_seeds[iter])
  
# complete_df <- complete_df[order(complete_df$y),]
complete_df <- complete_df[sample(x = 1:nrow(complete_df)),]
nn <- nrow(complete_df)
new_line_order <- c(seq(from = 1, to = nn, by = 3), seq(from = 2, to = nn, by = 3), seq(from = 3, to = nn, by = 3))
length(unique(new_line_order))
complete_df <- complete_df[new_line_order,]

# re-copy (y,x)
y <- complete_df$y
x <- as.matrix(complete_df[,-1])

# number of potential questions
p <- ncol(x)

# tree depth (do not change!)
q <- 3

# split between train/valid/test
n <- nrow(complete_df)
n_train <- ceiling(n/3)
training_indexes <- 1:n_train
validation_indexes <- (n_train+1):(2*n_train)
testing_indexes <- (2*n_train+1):(3*n_train - 1)
stopifnot(n == length(unique(c(training_indexes, validation_indexes, testing_indexes))))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                       ESTABLISHING BASELINES RPART
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # full tree, oracle

# build
rpFull <- rpart(formula = y ~ ., data = complete_df[1:n_train,])

# plot
plot(rpFull) ; text(rpFull)
rpart.plot(rpFull, roundint = F)

# predict + eval
rpFull_pred <- predict(rpFull, complete_df)
mean( prediction_error(predicted_value = rpFull_pred[testing_indexes], true_value = y[testing_indexes]) )

# # # depth-3 tree, baseline

# build
rpart_control <- rpart.control(maxdepth = 3)
rp0 <- rpart(formula = y ~ ., data = complete_df[training_indexes,], control = rpart_control)

# plot
plot(rp0) ; text(rp0)
rpart.plot(x = rp0, roundint = F)

# predict + eval
rp0_pred <- predict(rp0, complete_df)
mean( prediction_error(predicted_value = rp0_pred[testing_indexes], true_value = y[testing_indexes]) )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                       ...
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

rmse <- function(y, yhat) sqrt(mean( (y-yhat)^2 ))

# ...
which_max_otherwise_runif <- function(x){
  max_x <- max(x, na.rm = FALSE)
  index_max_x <- which(x == max_x)
  if(length(index_max_x) == 1) return(index_max_x)
  return(sample(x = index_max_x, size = 1))
}

# random forest
lm0 <- lm(y ~ ., data = complete_df[training_indexes,])
rf0 <- randomForest(y ~ ., data = complete_df[training_indexes,])
varImpPlot(rf0)
as.numeric(rf0$importance)
rf0_pred <- predict(rf0, complete_df)
mean( prediction_error(predicted_value = rf0_pred[testing_indexes], true_value = y[testing_indexes]) )
rmse( (rf0_pred[testing_indexes]), (y[testing_indexes]) )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   COMPUTE PREDICTOR, \hat{m} : \mathcal{\tilde{X}} \rightarrow \mathcal{Y}
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# establish list of potential questions one could ask with a three-layer questionnaire
potential_questions <- expand.grid(1:p, 1:p, 1:p)
potential_questions <- potential_questions %>% filter(Var1 < Var2 & Var2 < Var3)
nrow(potential_questions) == (factorial(p) / (factorial(3) * factorial(p-3)))

# compute random forest for each triplet considered
rf_training_indexes <- training_indexes
set_of_rf <- list()
pb <- txtProgressBar(min = 0, max = nrow(potential_questions))
for(i in 1:nrow(potential_questions)){
  setTxtProgressBar(pb = pb, value = i)
  colindexes <- c(1, 1 + as.numeric(potential_questions[i,1:3]))
  ##  tmp_rf <- randomForest(y ~ ., data = complete_df[rf_training_indexes, colindexes], ntree = 100, mtry = 3)
  tmp_rf <- lm(y ~ ., data = complete_df[rf_training_indexes, colindexes])
  set_of_rf[[i]] <- tmp_rf 
}

# # add performance indicator to potential_questions
# nb_trees <- 100
# potential_questions$rsq <- sapply(set_of_rf, function(x){ x$rsq[nb_trees] }) 
# hist(potential_questions$rsq, breaks = 20)
# potential_questions$set_of_rf_index <- 1:nrow(potential_questions)
# potential_questions <- potential_questions %>% arrange(desc(rsq)) 

# add performance indicator to potential_questions
potential_questions$rsq <- sapply(set_of_rf, function(x){ 1 - var(x$residuals)/var(x$residuals + x$fitted.values) }) 
hist(potential_questions$rsq, breaks = 20)
potential_questions$set_of_rf_index <- 1:nrow(potential_questions)
potential_questions <- potential_questions %>% arrange(desc(rsq)) 

# present performance summary
potential_questions_summary <- sapply(X = 1:p, FUN = function(j){
  summary(which(potential_questions$Var1 == j | potential_questions$Var2 == j | potential_questions$Var3 == j))
})
colnames(potential_questions_summary) <- colnames(x)
potential_questions_summary <- potential_questions_summary[,order(potential_questions_summary['Median',])]
boxplot(potential_questions_summary)

# three columns retained
colnames(x)[as.numeric(potential_questions[1,1:3])]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                 BUILDING (MASKED DATA SAMPLES, REWARD)
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# this set should be definetely different from the one used to build random forest,
# especially for the last layer, where we look at the performance, so overfitting is
# not permitted ...

# row indexes
indexes_to_train_mask <- c(validation_indexes)

# for each layer, expand masks
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
      for(a in actions_picked){
        x_masked_tmp[, a] <- x[indexes_to_train_mask, a]
        x_masked_tmp[, p + a] <- 1
      }
      x_masked <- rbind(x_masked, x_masked_tmp)
    }
  }
  colnames(x_unmasked) <- colnames(x)
  colnames(x_masked) <- c(colnames(x), paste0('is_unmasked_', colnames(x)))
  masked_by_layer[[length(masked_by_layer) + 1]] <- list('row' = row_indexes, 'x_unmasked' = x_unmasked, 'x_masked' = x_masked, 'y' = y_masked)
}

# for each possible action, evaluate long-term reward
si <- masked_by_layer[[3]]$x_masked
xi <- masked_by_layer[[3]]$x_unmasked
yi <- masked_by_layer[[3]]$y
colMeans(si == 0)
error_last_layer <- array(data = NA, dim = c(nrow(si), p))
for(action_picked in 1:p){
  
  # generate following state
  si_prime <- si
  si_prime[,action_picked] <- xi[,action_picked] # reveal value
  si_prime[,p + action_picked] <- 1 # indicate it is now known
  
  # get prediction from following state
  si_prime <- as.data.frame(si_prime)
  pred_si_prime <- get_prediction(si_prime = si_prime)
  
  # get reward associated
  error_last_layer[,action_picked] <- prediction_error(predicted_value = (pred_si_prime), true_value = (yi))
  
}

# summary
boxplot(error_last_layer)
boxplot(pmin(error_last_layer, 0.1))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                       TRAIN NEURAL NETWORK, LAYER 2
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# then: for each state, for each action, consider following states and 
# prediction from network1 pick the action returning highest reward
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# dimensions
input_size <- 3*p
output_size <- 1

# from state of layer 2 + action to final reward || architecture + network instanciation + compilation
fl <- 5 * p # 25*p
# network2 <- keras_model_sequential() %>%
#   layer_dense(units = fl, kernel_initializer = "uniform", input_shape = input_size, activation = "relu") %>%
#   layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
#   layer_dropout(rate = 0.4) %>%
#   layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
#   layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
#   layer_dropout(rate = 0.4) %>%
#   layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
#   layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
#   layer_dense(units = output_size, kernel_initializer = "uniform", activation = "relu")
network2 <- keras_model_sequential() %>%
  layer_dense(units = fl, kernel_initializer = "uniform", input_shape = input_size, activation = "relu") %>%
  layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
  layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
  layer_dense(units = output_size, kernel_initializer = "uniform", activation = "relu")
network2 %>% compile(optimizer = optimizer_rmsprop(lr = 1e-4), loss = "mse")
network2$optimizer$get_config()
summary(network2)

# ...
head(error_last_layer)
stats::var(c(error_last_layer), na.rm = T)

# take masked x from training set and reward data
initial_state <- masked_by_layer[[3]]$x_masked
str(masked_by_layer[[3]], 1)
head(initial_state)
target <- error_last_layer
cond <- (masked_by_layer[[3]]$row %in% validation_indexes)
initial_state <- initial_state[cond,]
target <- target[cond,]

# ...
hist(target, breaks = 50)
hist(log(1 + target*1e5), breaks = 50)
hist(log(1 + sqrt(target) * 1e2), breaks = 50)
hist(sqrt(target), breaks = 50)
hist(sqrt(sqrt(target)), breaks = 50)
## target <- 10 * sqrt(sqrt(target)) 
### target <- 10 * sqrt(sqrt((target)))
target <- sqrt(sqrt(target))

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
table(count_by_coordinate)
initial_state_complete <- initial_state_complete[count_by_coordinate == 3,]
target_complete <- target_complete[count_by_coordinate == 3]

#
validation_size <- 0.3 * length(target_complete)
encoded_state <- apply(X = initial_state_complete[,p + (1:(2*p))], MARGIN = 1, FUN = paste0, collapse = "")
u_encoded_state <- unique(encoded_state)
validation_size_per_state <- ceiling(validation_size / length(u_encoded_state))
nnet_validation_indexes <- NULL
for(u in u_encoded_state){
  tmp <- sample(x = which(encoded_state == u), size = validation_size_per_state)
  nnet_validation_indexes <- c(nnet_validation_indexes, tmp)
}
length(nnet_validation_indexes) / length(target_complete)
nnet_nonvalidation_indexes <- 1:length(target_complete)
nnet_nonvalidation_indexes <- nnet_nonvalidation_indexes[- nnet_validation_indexes]
new_indexes <- c(nnet_nonvalidation_indexes, nnet_validation_indexes)
length(unique(new_indexes)) == length(target_complete)
initial_state_complete <- initial_state_complete[new_indexes,]
target_complete <- target_complete[new_indexes]

# ...
summary(target_complete)
var(target_complete)
length(target_complete)

# checking out progress in view panel
options(keras.view_metrics = FALSE)

# callbacks
fit_callbacks <- list(
  callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 25, verbose = 0, mode = "auto"),
  callback_model_checkpoint(filepath = ".mdl_wts.hdf5", monitor = "val_loss", verbose = 0, save_best_only = TRUE, mode = "min"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", min_delta = 1e-5, factor = 0.99, patience = 10, mode = "auto")
)

# repeat until convergence
network2 %>% fit(x = initial_state_complete, y = target_complete, epochs = 500, batch_size = 256, validation_split = 0.3, verbose = 0, callbacks = fit_callbacks)
network2$optimizer$get_config()

# # take masked x from training set and reward data
# initial_state <- masked_by_layer[[3]]$x_masked
# target <- log(1 + error_last_layer*1e5) 
# action_vec <- rep(0,p)
# action_vec[3] <- 1
# action_mat <- t(replicate(10, action_vec))
# cbind(
#   predict(network2, cbind( initial_state[1:10,], action_mat) ),
#   error_last_layer[1:10,3]
# )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                       TRAIN NEURAL NETWORK, LAYER 1
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# dimensions
input_size <- 3*p
output_size <- 1

# from state of layer 1 + action to best reward from layer 2 || architecture + network instanciation + compilation
fl <- 50 # 25*p
network1 <- keras_model_sequential() %>%
  layer_dense(units = fl, kernel_initializer = "uniform", input_shape = input_size, activation = "relu") %>%
  layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
  #   layer_dropout(rate = 0.4) %>%
  # layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
  # layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
  #   layer_dropout(rate = 0.4) %>%
  layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
  layer_dense(units = output_size, kernel_initializer = "uniform", activation = "relu")
network1 %>% compile(optimizer = optimizer_rmsprop(lr = 1e-4), loss = "mse")
network1$optimizer$get_config()

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

# shuffle
mysample2 <- sample(x = 1:nrow(state_compiled))
state_compiled <- state_compiled[mysample2,]
target_compiled <- target_compiled[mysample2]

var(target_compiled)

# fit neural network
fit_callbacks <- list(
  callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 25, verbose = 0, mode = "auto"),
  callback_model_checkpoint(filepath = ".mdl_wts.hdf5", monitor = "val_loss", verbose = 0, save_best_only = TRUE, mode = "min"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", min_delta = 1e-5, factor = 0.99, patience = 10, mode = "auto")
)
summary(target_compiled)
network1 %>% fit(x = state_compiled, y = target_compiled, epochs = 500, batch_size = 128, validation_split = 0.3, verbose = 0, callbacks = fit_callbacks)
network1$optimizer$get_config()$learning_rate

# # ...
# pred_compiled <- predict(network1, state_compiled)
# cbind(pred_compiled, target_compiled)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                       LAYER 0, NO NEED FOR A NNET
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

#
boxplot(target_compiled_0)
sort(apply(target_compiled_0, 2, median))
sort(colMeans(target_compiled_0))
colnames(x)[order(colMeans(target_compiled_0))]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                       TRYING OUT ON EVALUATION DATA
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#
worst_error_possible <- 100

#
x_valid <- x[testing_indexes,]
states <- array(data = 0, dim = c(nrow(x_valid), 2*p))
colnames(states) <- c(colnames(x), paste0('is_unmasked_', colnames(x)))

# fill in first answer 
first_action <- as.numeric(which.min(colMeans(target_compiled_0)))
colnames(x)[first_action]
states[,first_action] <- x_valid[,first_action]
states[,p+first_action] <- 1
head(states)

# then select next action!
pred_network1 <- sapply(X = 1:p, FUN = function(j){
  action_mat <- array(data = 0, dim = c(nrow(states), p))
  action_mat[,j] <- 1
  predict(object = network1, x = cbind(states, action_mat))
})
pred_network1[,first_action] <- worst_error_possible
second_action <- apply(X = pred_network1, MARGIN = 1, FUN = which.min)
sort(table(colnames(x)[second_action]))

# fill in second answer
for(j in 1:p){
  cond <- (second_action == j)
  states[cond,j] <- x_valid[cond,j]
  states[cond,j+p] <- 1
}
head(states)

# then select last action!
pred_network2 <- sapply(X = 1:p, FUN = function(j){
  action_mat <- array(data = 0, dim = c(nrow(states), p))
  action_mat[,j] <- 1
  predict(object = network2, x = cbind(states, action_mat))
})
pred_network2[,first_action] <- worst_error_possible
for(j in 1:p){
  pred_network2[second_action==j,j] <- worst_error_possible
}
third_action <- apply(X = pred_network2, MARGIN = 1, FUN = which.min)
sort(table(colnames(x)[third_action]))

# fill in third answer
for(j in 1:p){
  cond <- (third_action == j)
  states[cond,j] <- x_valid[cond,j]
  states[cond,p+j] <- 1
}

# hello there.
selected_questions <- data.frame(
  'Q1' = first_action,
  'Q2' = second_action,
  'Q3' = third_action
)
sort(table( colnames(x)[unlist(selected_questions)] ))
colnames(x)[as.numeric( potential_questions[1,1:q] )]

# check that state does not contain more than q variables !
stopifnot( unique(rowSums(states[,p+1:p])) == 3 )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#     PREDICTIONS, PERFORMANCE / VARIABLE IMPORTANCE / QUESTION SEQUENCES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# states_check <- 0 * states
# states_check[,1:3] <- x[testing_indexes,1:3]
# states_check[,p+1:3] <- 1
# baseline_pred_2 <- get_prediction(si_prime = states_check)
# hist(baseline_pred_2)
# plot(baseline_pred_2, baseline_pred)

y_valid <- y[testing_indexes]

# get predictions
final_pred <- get_prediction(si_prime = as.data.frame(states))
rp0_pred <- predict(rp0, complete_df[testing_indexes,])
baseline_pred <- predict(set_of_rf[[potential_questions$set_of_rf_index[1]]], as.data.frame(x_valid))

# RMSE
rf0_pred <- predict(rf0, complete_df[testing_indexes,])
lm0_pred <- predict(lm0, complete_df[testing_indexes,])

# final_pred <- get_prediction(si_prime = as.data.frame(states))
# rf0_pred <- predict(rf0, complete_df[testing_indexes,])
# rp0_pred <- predict(rp0, complete_df[testing_indexes,])
# baseline_pred <- predict(set_of_rf[[potential_questions$set_of_rf_index[1]]], complete_df[testing_indexes,])

global_results[iter,] <- c(
  rmse((y_valid), (rf0_pred)),
  rmse((y_valid), (lm0_pred)),
  rmse((y_valid), (final_pred)),
  rmse((y_valid), (baseline_pred)),
  rmse((y_valid), (rp0_pred))
)

}

# 
tmpvec <- c('rf_oracle', 'lm_oracle', 'our_method', 'best_subset', 'tree')
colnames(global_results) <- c(paste0('rmse_', tmpvec)) #, paste0('rmsle_', tmpvec))
boxplot(global_results[,1:5])
colMeans(global_results[,1:5], na.rm = T)

resres <- apply(global_results[1:10,], 2, function(x){ c(mean(x), sd(x)) })
paste0(round(resres[1,],2), ' (', round(resres[2,],2), ')', collapse = ' & ')
# 3.89 (0.93) & 4.95 (0.82) & 4.82 (0.54) & 5.29 (0.7) & 5.13 (0.85)
# 3.83 (0.65) & 4.99 (0.57) & 4.92 (0.54) & 5.33 (0.54) & 5.16 (0.64)

# variable importance, global random forest / our method
TMP <- data.frame(
  "var" = colnames(x),
  "rf_importance" = as.numeric( importance(rf0) ),
  "nb_times" = as.numeric( table(factor(x = unlist(selected_questions), levels = 1:p )) )
)
g0 <- ggplot(data = TMP, mapping = aes(x = var, y = rf_importance)) + geom_bar(position = position_dodge(), stat = "identity") 
g1 <- ggplot(data = TMP, mapping = aes(x = var, y = nb_times)) + geom_bar(position = position_dodge(), stat = "identity") 
grid.arrange(g0, g1)

#
TMP2 <- TMP %>% arrange(desc(nb_times))
TMP2$var <- factor(x = as.character(TMP2$var), levels = as.character(TMP2$var))
TMP2$rf_importance <- TMP2$rf_importance / sum(TMP2$rf_importance)
TMP2$nb_times <- TMP2$nb_times / sum(TMP2$nb_times)
TMP2 <- TMP2 %>% gather(key = score_key, value = score_value, - var)
TMP2$score_key_label <- factor(x = TMP2$score_key, levels = c('rf_importance', 'nb_times'), labels = c('RF importance', 'Pct selection'))
ggplot(data = TMP2, mapping = aes(x = var, y = score_value, fill = score_key_label)) +
  geom_bar(stat = 'identity', position = 'dodge') +
  theme(text = element_text(size = 25), legend.position = 'bottom') +
  xlab(label = '') + ylab(label = '') + labs(fill = 'Score') +
  ggtitle(label = 'Random Forest importance (normalized) versus Pct selection using our method')

# sequence of questions
question_sequences <- apply(X = selected_questions, MARGIN = 1, FUN = function(xi){ 
  paste0(colnames(x)[xi], collapse = ' / ') 
})
sort(table(question_sequences))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#     DiagrammeR for the proportion of questions asked
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

library(DiagrammeR)

foo_code <- function(i, layer){ paste0('layer', layer, '', colnames(x)[as.numeric(i)]) }
foo_label <- function(i, layer){ colnames(x)[i] } 

u_nodes_layer1 <- unique(selected_questions[,1])
u_nodes_layer2 <- unique(selected_questions[,2])
u_nodes_layer3 <- unique(selected_questions[,3])

node_cmd <- paste0(c(
  paste0("node [fillcolor = white, label = '", foo_label(i = u_nodes_layer1, layer = 1), "'] \n", foo_code(i = u_nodes_layer1, layer = 1), "\n"),
  paste0("node [fillcolor = white, label = '", foo_label(i = u_nodes_layer2, layer = 2), "'] \n", foo_code(i = u_nodes_layer2, layer = 2), "\n"),
  paste0("node [fillcolor = white, label = '", foo_label(i = u_nodes_layer3, layer = 3), "'] \n", foo_code(i = u_nodes_layer3, layer = 3), "\n")
), collapse = '\n')

edges_layer_1_to_2 <- data.frame(table(selected_questions[,1:2]) / nrow(x[testing_indexes,]))
edges_layer_1_to_2$Q1 <- as.numeric(as.character(edges_layer_1_to_2$Q1))
edges_layer_1_to_2$Q2 <- as.numeric(as.character(edges_layer_1_to_2$Q2))
edges_layer_2_to_3 <- data.frame(table(selected_questions[,2:3]) / nrow(x[testing_indexes,]))
edges_layer_2_to_3 <- edges_layer_2_to_3[edges_layer_2_to_3$Freq > 1e-7,]
edges_layer_2_to_3$Q2 <- as.numeric(as.character(edges_layer_2_to_3$Q2))
edges_layer_2_to_3$Q3 <- as.numeric(as.character(edges_layer_2_to_3$Q3))

scaleUP <- 20
edges_layer_1_to_2_cmd <- paste0(
  "edge [color = black, penwidth = ", scaleUP*round(edges_layer_1_to_2$Freq, 2), "] \n",
  foo_code(i = edges_layer_1_to_2$Q1, layer = 1),
  " -> ", foo_code(i = edges_layer_1_to_2$Q2, layer = 2), "\n"
)
edges_layer_2_to_3_cmd <- paste0(
  "edge [color = black, penwidth = ", scaleUP*round(edges_layer_2_to_3$Freq, 2), "] \n",
  foo_code(i = edges_layer_2_to_3$Q2, layer = 2),
  " -> ", foo_code(i = edges_layer_2_to_3$Q3, layer = 3), "\n"
)

cmd <- paste0(
  "digraph blabla{

  # a 'graph' statement
  graph [layout = dot, overlap = false, fontsize = 10]
  
  node [shape = circle,
      style = filled,
      color = black,
      label = '']  
",
  node_cmd,
  paste0(c(edges_layer_1_to_2_cmd, edges_layer_2_to_3_cmd), collapse = "\n"),
  "}")
cat(cmd)
grViz(cmd)

colnames(x)[as.numeric( potential_questions[1,1:3] )]


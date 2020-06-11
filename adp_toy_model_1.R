## empty environment
rm(list = ls())

# load some libraries
## models
library(randomForest)
library(rpart)
## viz
library(ggplot2)
library(data.table)
library(rpart.plot)
library(DiagrammeR)
library(pROC)

# source functions
source('to_source/checks_on_y_and_x.R')
source('to_source/compute_pmat_smat_tmat.R')
source('to_source/dynamic_programming.R')
source('to_source/visualize.R')
source('to_source/tree_depth_pruning.R')

paste0(floor(runif(n = 10, min = 1000, max = 9999)), collapse = ', ')
list_of_seeds <- c(5031, 7402, 7639, 7217, 9219, 2931, 7134, 3740, 9767, 6779)
global_results <- array(data = NA, dim = c(length(list_of_seeds), 4))

for(iter in 1:length(list_of_seeds)){
  
  cat('\n', iter)
  set.seed(list_of_seeds[iter])
  
# read dataframe and seperate into (x, y)
n <- 6000

# number of variates
p <- 8

# generate X 
x <- rbinom(n = n*p, size = 1, prob = 0.5)
x <- matrix(data = x, nrow = n, ncol = p)

# generate Y based on rules
y <- rep(1, n)
y[x[,1] == 0 & x[,2] == 0 & x[,8] == 0] <- 2
y[x[,6] == 0 & x[,2] == 1 & x[,3] == 1] <- 3
y[x[,3] == 0 & x[,5] == 0 & x[,7] == 0] <- 4
y[x[,8] == 1 & x[,1] == 1 & x[,3] == 1] <- 5
y[x[,4] == 0 & x[,5] == 0 & x[,6] == 0] <- 6
y[x[,3] == 1 & x[,4] == 1 & x[,2] == 1] <- 7
y[x[,4] == 1 & x[,8] == 1 & x[,1] == 1] <- 8
y <- 1 + 1*(y == 1)

# bind (x,y) and write it in .csv file
colnames(x) <- paste0('X', 1:p)

# choose percentage of train/test
pct_test <- 1/3

# seperate into train/test
set.seed(0)
index_test <- sample(x = 1:nrow(x), size = floor(nrow(x)*pct_test)) 
y_test <- y[index_test]
x_test <- x[index_test,]
y <- y[-index_test]
x <- x[-index_test,]

# check class balance
table(y)
table(y_test)

# choose depth of tree
q <- 3

#############################################################################################
###
###                   Checking procedure + get (n,p) dimensions
###
#############################################################################################

# checking individual validity of x and y
checks_x(x = x)
checks_y(y = y)

table(rowSums(x))
# 89201/115922
table(y)

# number of observations and number of variates
n <- nrow(x)
p <- ncol(x)

# are dimensions compatible between x and y ?
nb_rows_x_equals_length_y <- (n == length(y))
stopifnot(nb_rows_x_equals_length_y)

# is the tree depth required q small enough w.r.t p ?
q_smaller_than_p <- (q < p)
stopifnot(q_smaller_than_p)

# parsing matrix "x"
x_parsed <- as.character(apply(X = x, MARGIN = 1, FUN = foo_id))

# compute "pmat"
pmat <- compute_pmat(p = p, x_parsed = x_parsed, use_ucb = FALSE)
str(pmat)
hist(pmat$p)

#############################################################################################
###
###   Compute \hat{m}^* = \argmin \hat{E}_{X,Y}(Y \neq \argmax_{y} m(X)_[y]]
###   based on random forest method and full information X
###
#############################################################################################

# create predictor based on full-information
rf <- randomForest(y = factor(y), x = x, mtry = 8, ntree = 500)
varImpPlot(rf)
rf

# apply a comparison with a single CART tree
df_rpart <- data.frame('y' = y, x)
rpart_control <- rpart.control(maxdepth = p, minsplit = 5, cp = 0)
rp0 <- rpart(formula = y ~., data = df_rpart, method = 'class', control = rpart_control)
## un-pruned tree
prp(rp0, extra = 8)
plotcp(rp0)
rp0 <- my.prune.rpart(tree = rp0, L = q-1)
prp(rp0, extra = 8)
str(x)
rp0_pred <- predict(rp0, data.frame(x))

# extract predictions for each full-information
x_new <- as.matrix(pmat[,1:p])
y_pred_rf <- predict(object = rf, newdata = x_new, type = 'prob')

# build smat
smat <- compute_smat(p = p, pmat = pmat, y_pred = y_pred_rf, x = x, y = y)
str(smat)

# inspection on predicted Y 
table(smat$Y_PRED[smat$is_terminal], useNA = 'always')

# # reward distribution
# ggplot(data = smat) +
#   geom_boxplot(mapping = aes(x = is_terminal, y = REWARD))
# hist(smat$REWARD[smat$is_terminal])

#############################################################################################
###
###   Apply Dynamic Programming technique on "tmat" and present direct results.
###
#############################################################################################

# compute tmat
tmat <- compute_tmat(p = p, smat = smat)

# use dp() function
qsa <- dp(df = tmat)

# add recommendation to smat
smat$recommended_action <- apply(qsa, 1, function(x){
  l <- which.max(x)
  if(length(l)==0) l <- NA
  return(l)
})

# plot recommendations
qsa_reframed <- reframe_qsa_mat(qsa = qsa, smat = smat, row_normalized = TRUE)
# ggplot(data = qsa_reframed[qsa_reframed$depth_in_tree == 0,]) +
#   geom_tile(mapping = aes(x = variable, y = state, fill = value))
# ggplot(data = qsa_reframed[qsa_reframed$depth_in_tree == 1,]) +
#   geom_tile(mapping = aes(x = variable, y = state, fill = value))

# # represent recommended questionnaire
# cmd <- build_grViz_cmd(p = p, mdp = tmat, states = smat)
# grViz(diagram = cmd) 

#############################################################################################
###
###   Evaluation part. Compute prediction and therefore (TP, FP, TN, FN).
###
#############################################################################################

# compute for decision tree baseline
y_pred_rp0 <- predict(rp0, type = 'prob', newdata = data.frame(x_test))
roc_obj_rp0 <- roc(1*(y_test==2), y_pred_rp0[,2])

# compute for our approach
y_pred_rf0 <- array(data = 0, dim = c(nrow(x_test), 2))
for(i in 1:nrow(x_test)){
  
  # compute partial state reached from the tree
  state <- rep(-1,p)
  for(j in 1:q){
    ij <- which(smat$PARSED_X == foo_id(state))
    aj <- smat$recommended_action[ij]
    state[aj] <- x_test[i,aj]
  }
  
  # identify X candidates
  cond <- rep(TRUE, nrow(pmat))
  for(j in 1:p){
    if(state[j] != -1){
      cond <- cond & (pmat[,j] == state[j])
    }
  }
  
  # compute prediction for each X candidate
  y_pred <- y_pred_rf[cond,]
  
  # compute probability for each X candidate
  x_prob <- pmat$p[cond]
  x_prob <- x_prob / sum(x_prob)
  
  # compute final probability
  y_pred_rf0[i,] <- x_prob %*% y_pred
  
}
#
roc_obj_rf0 <- roc(1*(y_test==2), y_pred_rf0[,2])
#
roc_obj_rf <- roc(1*(y_test==2), predict(rf, x_test, type = 'prob')[,2])
#
tmpp <- expand.grid('Q1' = 1:p, 'Q2' = 1:p, 'Q3' = 1:p)
tmpp <- tmpp[tmpp$Q1 < tmpp$Q2 & tmpp$Q2 < tmpp$Q3,]
rf_df <- data.frame('y' = factor(y), x)
rf_list <- list()
for(row_index in 1:nrow(tmpp)){
  rf_list[[row_index]] <- randomForest(y ~., data = rf_df[,c(1, 1+as.numeric(tmpp[row_index,]))], ntree = 100)
}
rf_best_subset <- which.min(sapply(rf_list, function(x){ as.numeric(x$err.rate[100,1]) }))
roc_obj_q_subset <- roc(1*(y_test==2), predict(rf_list[[rf_best_subset]], x_test, type = 'prob')[,2])

global_results[iter,] <- c(
  as.numeric(auc(roc_obj_rf)),
  as.numeric(auc(roc_obj_rf0)),
  as.numeric(auc(roc_obj_q_subset)),
  as.numeric(auc(roc_obj_rp0))
)

}

#
colnames(global_results) <- c('rf_oracle', 'our_method', 'best_subset', 'tree')
resres <- apply(global_results, 2, function(x){ c(mean(x), sd(x)) })
paste0(round(resres[1,],2), ' (', round(resres[2,],2), ')', collapse = ' & ')


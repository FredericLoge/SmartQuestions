# Rpart tree pruning based on depth parameter L
my.prune.rpart <- function (tree, L, ...){
  ff <- tree$frame
  id <- as.integer(row.names(ff))
  nodes_L <- rpart:::tree.depth(id)
  toss <- id[nodes_L > L & ff$var != "<leaf>"]
  if (length(toss) == 0L) 
    return(tree)
  newx <- snip.rpart(tree, toss)
  # temp <- pmax(tree$cptable[, 1L])
  # keep <- match(unique(temp), temp)
  # newx$cptable <- tree$cptable[keep, , drop = FALSE]
  # newx$cptable[max(keep), 1L] <- cp
  newx$variable.importance <- rpart:::importance(newx)
  newx
}

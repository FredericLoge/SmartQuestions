checks_x <- function(x){
  
  x_is_matrix <- is.matrix(x)
  stopifnot(x_is_matrix)
  
  x_are_binary <- all(c(x) %in% c(0,1))
  stopifnot(x_are_binary)
  
}

checks_y <- function(y){
  
  y_is_vector <- is.vector(y)
  stopifnot(y_is_vector)
  
  y_is_numeric <- is.numeric(y)
  stopifnot(y_is_numeric)
  
  y_has_no_NA <- (any(is.na(y)) == FALSE)
  stopifnot(y_has_no_NA)
  
  y_min_is_1 <- (min(y) == 1)
  stopifnot(y_min_is_1)
  
  y_max_is_not_1 <- (max(y) > 1)
  stopifnot(y_max_is_not_1)
  
  y_took_all_values_from_min_to_max <- (length(table(y)) == max(y))
  stopifnot(y_took_all_values_from_min_to_max)
  
}

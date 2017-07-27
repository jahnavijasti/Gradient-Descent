##Consider the Hitters dataset. Standardize the data.

library(ISLR)
Hitters = na.omit(Hitters)
x = model.matrix(Salary~., Hitters)[,-1]
y = Hitters$Salary
set.seed(1)
ntrain = as.integer(nrow(x)/2)
train_idxs = sample(1:nrow(x), ntrain)
test_idxs = (-train_idxs)
x_train = x[train_idxs,]
x_test = x[test_idxs,]
y_train = y[train_idxs]
y_test = y[test_idxs]
x_train = scale(x_train, center = TRUE, scale = TRUE)
y_train = y_train-mean(y_train)

## Write a function computegrad that computes and returns rF( ) for any beta.

computegrad <- function(beta, lambda=0.1, x=x_train, y=y_train){ 
  grad_beta = -2/nrow(x)*t(x)%*%(y-x%*%beta) + 2*lambda*beta 
  return(as.vector(grad_beta))
}

##Write a function graddescent that implements the gradient descent algorithm described in Algorithm 1. 
##The function graddescent calls the function computegrad as a sub-routine. 
##The function takes as input the initial point, the constant step- size value, and 
##the maximum number of iterations. The stopping criterion is the maximum number of iterations.

graddescent <- function(x_init, eta, max_iter=1000){
  x = x_init
  grad_x = computegrad(x)
  x_vals = x
  iter = 0
  while(iter < max_iter){
    x = x - eta*grad_x
    x_vals = rbind(x_vals, x)
    grad_x = computegrad(x)
    iter = iter + 1
    }
  return(x_vals) 
}

## Set the constant step-size to ⌘ = 0.1 and the maximum number of iterations to 1000. 
##Run graddescent on the training set of the Hitters dataset for  eta = 0.1. 
##Plot the curve of the objective value F( Beta-t) versus the iteration counter t. 
##What do you observe?

f_ridge <- function(beta, lambda=0.1, x=x_train, y=y_train){
  obj = 1/nrow(x)*sum((y-x%*%beta)^2) + lambda*sum(beta*beta) 
  return(obj)
}
convergence_plot <- function(f, beta_vals){ 
  n = nrow(beta_vals)
  fs = numeric(n)
  for (i in 1:n){
    fs[i] = f(beta_vals[i,])
    }
  par(mfrow=c(1,1))
  plot(fs, xlab="Iteration", ylab="Objective value") 
  }
n = nrow(x_train)
d = ncol(x_train)
beta_init = numeric(d)
eta = 0.1
beta_vals = graddescent(beta_init, eta)
convergence_plot(f_ridge, beta_vals) # It converges quite quickly

beta_vals[nrow(beta_vals),]

##Denote  beta-T the final iterate of your gradient descent algorithm. 
##Compare  beta-T to the  beta* found by glmnet. 
##Compare the objective value for  T to the one for  ?. What do you observe?

library(glmnet)

ridge.mod = glmnet(x_train, y_train, alpha=0, lambda=0.1,
                   thresh = 1e-011)
beta_glmnet = as.vector(coef(ridge.mod)[2:length(coef(ridge.mod))])
beta_glmnet

f_ridge(beta_glmnet)

f_ridge(beta_vals[nrow(beta_vals),])

## The coefficient estimates and objective values are different.
## If we scale y first, we get the same result as in glmnet.

y_train = scale(y_train, center = TRUE, scale = TRUE)
beta_vals = graddescent(beta_init, eta)

ridge.mod = glmnet(x_train, y_train, alpha = 0, lambda = 0.1, thresh = 1e-011)
beta_glmnet = as.vector(coef(ridge.mod)[2:length(coef(ridge.mod))])

f_ridge(beta_vals[nrow(beta_vals),])
f_ridge(beta_glmnet)

y_train = y[train_idxs] - mean(y[train_idxs])
ridge.mod = glmnet(x_train, y_train, alpha = 0, lambda = 0.1*sd(y_train), thresh = 1e-011)

beta_glmnet = as.vector(coef(ridge.mod)[2:length(coef(ridge.mod))])


##Run your gradient algorithm for many values of eta on a logarithmic scale. 
##Find the final iterate, across all runs for all the values of eta, that achieves the 
##smallest value of the objective. Compare  beta-T to the beta* found by glmnet. 
##Compare the objective value for  beta-T to the beta*. What conclusion to you draw?

y_train = y[train_idxs]-mean(y[train_idxs]) etas = 10^seq(-10, 3, 1)
obj_vals = numeric(length(etas))
i=1
for (eta in etas){
  beta_vals = graddescent(beta_init, eta) 
  obj_vals[i] = f_ridge(beta_vals[nrow(beta_vals),]) 
  i=i+1
}

argmin = which.min(obj_vals)
best_eta = etas[argmin]
best_obj = obj_vals[argmin]
best_obj


betas_best = graddescent(beta_init, best_eta)
beta_best = betas_best[nrow(betas_best),]
beta_best


#Again, if we standardize y, we get (close to) the same result as glmnet:
y_train = scale(y_train, center=TRUE, scale=TRUE) 
beta_best = graddescent(beta_init, best_eta) 
beta_best[nrow(beta_best),]

ridge.mod = glmnet(x_train, y_train, alpha=0, lambda=0.1,
                       thresh = 1e-011)
beta_glmnet = as.vector(coef(ridge.mod)[2:length(coef(ridge.mod))])
f_ridge(beta_best[nrow(beta_best),])

f_ridge(beta_glmnet)

# Let's try with a step size of 1/L, where L is the Lipschitz  constant

y_train = y[train_idxs]-mean(y[train_idxs])
library(RSpectra)
# Compute the top eigenvalue
eigs_result = eigs_sym(t(x_train)%*%x_train, 1)
lambda_max = eigs_result$values[1]
lambda = 0.1
eta_L = 1/(2/nrow(x_train)*lambda_max + 2*lambda)
betas_L = graddescent(beta_init, eta_L, max_iter=10000)
beta_L = betas_L[nrow(betas_L),]
obj_L = f_ridge(beta_L)
obj_L # Same as the best one we found before





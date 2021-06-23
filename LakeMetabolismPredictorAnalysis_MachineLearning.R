
# Script prepared by Nicole K. Ward.
# Date of Last update: June 22 2021
# R Version: 3.6.3

#### Acknowledgements and Attributions ####

# The vast majority of this script, particularly the individual model fiting routines, the ensemble 
# model stacking approach, and visualization/interpretation techniques are reproduced and adapted 
# from the wonderful resource by Bradley Boehmke & Brandon Greenwell: Hands on Machine Learning in 
# R, a book available at https://bradleyboehmke.github.io/HOML/ with the citation:
# Boehmke B, Greenwell B (2020) Hands-on Machine Learning with R. CRC Press, Taylor & Francis Group, 
#      Boca Raton, FL. 

# Tuning script for XGB hyperparameters from https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret

# This script builds on the foundation of Boehmke and Greenwell by applying a nested cross-validation 
# approach to each model fitting routine, as described in the paper by Vabalas et al. This technique
# is a way to robustly fit machine learning models using small datasets: 
# Vabalas A, Gowen E, Poliakoff E, Casson AJ (2019) Machine learning algorithm validation with 
#      a limited sample size. PLoS One 14(11):e0224365. https://doi.org/10.1371/journal.pone.0224365



#### Purpose of this script ####
# Conduct predictor analysis on Lake Metabolism estimates (GPP and R) using machine learning. 
# Once through the script is a complete predictor analysis for one response variable of interest 
# (either GPP or R) at one site in the study. This script is being published to accompany a 
# manuscript being submitted to the Journal of Aquatic Sciences, with the title "Physical 
# characteristics of the stream - lake transitional zone affect littoral lake metabolism." 
# The manuscript has complete description of the machine learning analysis to accompany this script.

### Required Packages ####
# Clean your space
rm(list=ls())

library(caret)
library(h2o)
library(glmnet)

#### Outline of Script ####
# 1. Data Input: description of data requirements, data citations, read in file to working 
#        environment
# 2. Nested CV assessment of Random Forest model
# 2a. Fit Final Random Forest model
# 3. Nested CV assessment of Regularized Regression model
# 3a. Fit Final Regularized Regression Model
# 4. Nested CV assessment of eXtreme Gradient Boosting (XGB) model
# 4a. Fit Final XGB model
# 5. Build final ensemble model using random forest, regularized regression, and xgb base learners


#### 1. Data Input ####
# Required File: A data table with lake metabolism estimates and associated predictor variables for 
# the site of interest. In this script, the collated data table is referred to as "Metab_Pred"
# All predictor and response variables are collated into one table with daily values.

# Clean (provisional) metbaolism estimates (to be published with DOI upon manuscript acceptance) 
# is available at:
# https://portal-s.edirepository.org/nis/mapbrowse?scope=edi&identifier=214&revision=5

# Predictor data pulls from many different publicly available datasets, see manuscript for details.
# 
# Availability of data and material: All data used in the analysis are archived at:
# Ewing, H.A., B.G. Steele, and K.C. Weathers. 2021. High resolution stream temperature, pressure, 
#     and estimated depth from transducers in streams in the Lake Sunapee watershed, New Hampshire, 
#     USA 2010 – 2018 ver 4. Environmental Data Initiative,
#     https://doi.org/10.6073/pasta/9921cdeb6291b1e43251a3d776942c71.
# LSPA, K.C. Weathers, & B.G. Steele. 2020a. High-frequency Weather Data at Lake Sunapee, 
#     New Hampshire, USA 2007 – 2019. Environmental Data Initiative,
#     https://doi.org/10.6073/pasta/698e9ffb0cdcda81ecf7188bff54445e.
# LSPA, K.C. Weathers, and B.G. Steele. 2020b. Lake Sunapee Instrumented Buoy: High Frequency Water 
#     Temperature and Dissolved Oxygen Data – 2007-2019 ver 1. Environmental Data Initiative, 
#     https://doi.org/10.6073/pasta/70c41711d6199ac2758764ecfcb9815e.
# Ward, N.K., J.A. Brentrup, A.E. Johnson, C.C. Carey, K.C. Weathers, and J.R. Fichter. 2019. 
#     Underwater temperature and light data from 3 mini-buoys in Lake Sunapee, NH, USA from June – 
#     October 2018 ver 1. Environmental Data Initiative,
#     https://doi.org/10.6073/pasta/20a698ec77707dd41c8a0b03a9fb2e34.
  

# Set to your own working directory
setwd("~/Downloads")              

# subfolder in working directory with metabolism data
input_dir<-("./machinelearningupdate/") 

# read in collated data containing metabolism estimates and associated predictor variables at a daily
# time step
Metab_Pred<-read_csv("./machinelearningupdate/regressionTreeGPP_HC.csv") 

# Machine learning analysis requires complete data:
Metab_Pred<-Metab_Pred[complete.cases(Metab_Pred),] 



#### 2. Nested CV assessment of Random Forest model ####

# this section adapted from https://bradleyboehmke.github.io/HOML/random-forest.html
# other potentially useful links:
# https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/#comment-578556
# https://stats.stackexchange.com/questions/65128/nested-cross-validation-for-model-selection
# https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/

# Train & cross-validate a RF model
# random forest is useful when one or two predictor variables may swamp out the signal
# by randomly subsetting the possible predictors at each node, other predictor variables
# can be more "seen" - wind direction and speed was coming out as overly important in
# regression tree analysis, so RF might help assess a wider range of predictors

# main hyperparameters when tuning random forests: 1) number of trees, 2) number of features 
# to consider at any given split (mtry), 3) complexity of each tree, 4) sampling scheme
# and 5) splitting rule to use during tree construction.


# to make reproducible
set.seed(1)

# split dataset
folds<-createFolds(Metab_Pred$GPP_mgO2perLperD,k = 7)

# Please download and install the latest version of h2o from http://h2o.ai/download/
# make sure h2o connection is fresh
h2o.shutdown(prompt = FALSE)
h2o.init() 

# Note: this for loop fails to work if the h2o connection is not shut down and re-established for each
# iteration. Easy workaround is to modify the "for(i in 1:7)" to be "for(i in 1:1)", then 2:2, etc...
# If anyone has a dependable solution to disconnect and reconnect the h2o cluster within a for loop,
# let me know, as that would be quite useful!
for(i in 1:7){

  data_in<-Metab_Pred[-folds[[i]],]
  data_h2o<-as.h2o(data_in)
  
  response<-"GPP_mgO2perLperD"
  predictors<-setdiff(names(data_h2o),response)
  
  # number of features
  n_features <- length(setdiff(names(data_h2o), "GPP_mgO2perLperD"))
  
  hyper_grid <- list(
    mtries = floor(n_features * c(.25, .4, .5, .6, .75)),
    min_rows = c(1, 3, 5, 10),
    max_depth = c(10, 20, 30),
    sample_rate = c(1)
  )
  # random grid search strategy
  search_criteria <- list(
    strategy = "RandomDiscrete",
    stopping_metric = "RMSE",
    stopping_tolerance = 0.001,   # stop if improvement is < 0.1%
    stopping_rounds = 10,         # over the last 10 models
    max_runtime_secs = 60*5      # or stop search after 5 min.
  )
  # perform grid search 
  random_grid <- h2o.grid(
    algorithm = "randomForest",
    grid_id = "rf_random_grid",
    x = predictors,nfolds=3,
    y = response, 
    training_frame = data_h2o,
    hyper_params = hyper_grid,
    ntrees = n_features * 10,
    seed = 123,
    search_criteria = search_criteria
  )
  # collect the results and sort by our model performance metric 
  # of choice
  random_grid_perf <- h2o.getGrid(
    grid_id = "rf_random_grid", 
    sort_by = "RMSE", 
    decreasing = FALSE
  )
  random_grid_perf
  # save the top model, by mse
  best_1<-h2o.getModel(random_grid_perf@model_ids[[1]])
  if(i==1){
    save_1<-(best_1@model[["model_summary"]])
    save_1$min_rows<-as.numeric(random_grid_perf@summary_table[1,2])
    save_1$mtries<-as.numeric(random_grid_perf@summary_table[1,3])
    save_1$inner_rmse<-as.numeric(random_grid_perf@summary_table[1,6])
  }else{
    save_1[i,]<-(best_1@model[["model_summary"]])
    save_1$min_rows[i]<-as.numeric(random_grid_perf@summary_table[1,2])
    save_1$mtries[i]<-as.numeric(random_grid_perf@summary_table[1,3])
    save_1$inner_rmse[i]<-as.numeric(random_grid_perf@summary_table[1,6])
  }

  #then evaluate model on the test set
  data_test<-Metab_Pred[folds[[i]],]
  test_h2o<-as.h2o(data_test)
  best_perf<-h2o.performance(model=best_1,newdata = test_h2o)
  rmse_in<-best_perf@metrics$RMSE
  if(i==1){
    save_1$outer_rmse<-as.numeric(rmse_in)
  }else{
    save_1$outer_rmse[i]<-as.numeric(rmse_in)
  }

}

# the above nested CV provides an assessment of the model fiting routine itself (see Vabalas et al., 2019
# for more details), but does not provide the final model parameters -- see next section for final model.


#### 2a. Fit Final Random Forest model #### 
# With the above assessment of the model fiting routine, we now run the inner loop on the whole dataset 
# to establish final model parameters
h2o.shutdown(prompt = FALSE)
h2o.init()
  
  data_in<-Metab_Pred
  data_h2o<-as.h2o(data_in)
  
  response<-"GPP_mgO2perLperD"
  predictors<-setdiff(names(data_h2o),response)
  
  # number of features
  n_features <- length(setdiff(names(data_h2o), "GPP_mgO2perLperD"))
  
  hyper_grid <- list(
    mtries = floor(n_features * c(.25, .4, .5, .6, .75)),
    min_rows = c(1, 3, 5, 10),
    max_depth = c(10, 20, 30),
    sample_rate = c(0.6,0.75,0.9,1)
  )
  # random grid search strategy
  search_criteria <- list(
    strategy = "RandomDiscrete",
    stopping_metric = "RMSE",
    stopping_tolerance = 0.001,   # stop if improvement is < 0.1%
    stopping_rounds = 10,         # over the last 10 models
    max_runtime_secs = 60*5      # or stop search after 5 min.
  )
  # perform grid search 
  random_grid <- h2o.grid(
    algorithm = "randomForest",
    grid_id = "rf_random_grid",
    x = predictors,nfolds=3,
    y = response, 
    training_frame = data_h2o,
    hyper_params = hyper_grid,
    ntrees = n_features * 10,
    seed = 123,
    search_criteria = search_criteria
  )
  # collect the results and sort by our model performance metric 
  # of choice
  random_grid_perf <- h2o.getGrid(
    grid_id = "rf_random_grid", 
    sort_by = "RMSE", 
    decreasing = FALSE
  )
  random_grid_perf
  # save the top model, by mse
  best_1<-h2o.getModel(random_grid_perf@model_ids[[1]])

  
    #Now, create the final model using parameters from above grid search
    h2o.init()
    
    #split into training and test data
    set.seed(2) 
    data_split<-initial_split(Metab_Pred,strata = "GPP_mgO2perLperD")
    ames_train<-training(data_split)
    ames_test<-testing(data_split)
    train_h2o <- as.h2o(ames_train)
    test_h2o <- as.h2o(ames_test)
    
    #assign response variable
    Y <- "GPP_mgO2perLperD" 
    
    #assign predictor variables
    X <- setdiff(names(train_h2o), Y)
    
    # use parameters from top fitting model:
    best_rf <- h2o.randomForest(
      x = X, y = Y, training_frame = train_h2o, ntrees = 1000, mtries = 12,
      validation_frame = test_h2o,sample_rate = 1,
      max_depth = 10, min_rows = 5,  nfolds = 3,
      fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
      seed = 123, stopping_rounds = 50, stopping_metric = "RMSE",
      stopping_tolerance = 0
    )
    
    #pull final model RMSE w/hold-out test data
    h2o.performance(best_rf, newdata = test_h2o)@metrics$RMSE


#### 3. Nested CV assessment of Regularized Regression model ####

# see https://bradleyboehmke.github.io/HOML/regularized-regression.html:
# why regularized regression for this dataset? useful with multicollinearity
# regularized regression constrains the total size of all coefficient estimates. 
# this reduces the magnitude and fluctuations in coeffients, reducing varience in
# the model. objective function is to minimize SSE + P (penalty term = only way coefficients 
# can increase is if we experience comparable decrease in sum of squared errors)
# 3 types of penalty functions: 1) ridge penalty, lambda = 0 same as regular OLS, 
# lamda = infinity coefficients = 0. less important features get pushed to zero
# correlated features get pushed toward each other, rather than one being wildly
# positive and one being wildly negative. Good for smaller datasets with multicollinearity


# We will use glmnet R package for training regularized regression:
# https://cran.r-project.org/web/packages/glmnet/index.html

#create table to save results
save_GLM<-data.frame(matrix(ncol = 4,nrow=7))
x<-c("alpha","lambda","innerRMSE","outerRMSE")
colnames(save_GLM)<- x

#make reproducible
set.seed(111)

#fold data
folds<-createFolds(Metab_Pred$GPP_mgO2perLperD,k = 7)
h2o.shutdown(prompt = FALSE)
h2o.init()

#conduct nested cv assessment of model fiting routine for regularized regression model:
for(i in 1:7){
  data_in<-Metab_Pred[-folds[[i]],]
  data_in<-data_in[complete.cases(data_in),]
  
  # Create training  feature matrices
  # we use model.matrix(...)[, -1] to discard the intercept
  X <- model.matrix(GPP_mgO2perLperD ~ ., data_in)[, -1]
  
  # transform y with log transformation
  Y <- log(data_in$GPP_mgO2perLperD)
  
  # grid search across 
  cv_glmnet <- train(
    x = X,
    y = Y,
    method = "glmnet",
    preProc = c("zv", "center", "scale"),
    trControl = trainControl(method = "cv", number = 3),
    tuneLength = 10
  )
  
  # results for model with lowest RMSE
  cv_glmnet$results %>%
    filter(alpha == cv_glmnet$bestTune$alpha, lambda == cv_glmnet$bestTune$lambda)
  # Note:  RMSE in auto-results is not comparable to other variables, because we log transformed Y
  
  # save top model parameters
  save_GLM$alpha[i]<-  cv_glmnet$bestTune$alpha
  save_GLM$lambda[i]<- cv_glmnet$bestTune$lambda

  # predict sales price on training data
  pred <- predict(cv_glmnet, X)
  
  # compute RMSE of transformed predicted (to create interpretable RMSE)
  rmse<-RMSE(exp(pred), exp(Y))
  save_GLM$innerRMSE[i]<-rmse
  
  #start validation assessment
  val_in<-Metab_Pred[folds[[i]],]
  val_in<-val_in[complete.cases(val_in),]
  
  # Create training  feature matrices
  # we use model.matrix(...)[, -1] to discard the intercept
  X.t <- model.matrix(GPP_mgO2perLperD ~ ., val_in)[, -1]
 
  # when test data does not have observations for categorical variables, need to add columns with zeros for
  # those categories:
  t<- !(colnames(X) %in% colnames(X.t))
  colsnamesneed<-colnames((X[,c(t),drop=FALSE]))
  if(!is.null(colsnamesneed)){
    add<-matrix(0, ncol=length(colsnamesneed),nrow = nrow(X.t))
    colnames(add) <- c(colsnamesneed)
    X.t.2<-cbind(X.t,add)
  }
  if(is.null(colsnamesneed)){
    X.t.2<-X.t
  }

  #fit test data:
  pred.t <- predict(cv_glmnet, X.t.2)
  
  # transform y with log transformation
  Y.t <- log(val_in$GPP_mgO2perLperD)
  rmse.t<-RMSE(exp(pred.t), exp(Y.t))
  save_GLM$outerRMSE[i]<-rmse.t
}
# 'save_GLM' generated in above for loop has the nested CV assessment of the model fitting routine


#### 3a. Fit Final Regularized Regression Model ####

# assign predictor variables
 X <- model.matrix(GPP_mgO2perLperD ~ ., Metab_Pred)[, -1]

# transform y (response variable) with log transformation
Y <- log(Metab_Pred$GPP_mgO2perLperD)

# settings for model fiting procedure
cv_glmnet <- train(
  x = X,
  y = Y,
  method = "glmnet",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 3),
  tuneLength = 10
)

# results for model with lowest RMSE (this RMSE is not comparable, due to log transformation)
cv_glmnet$results %>%
  filter(alpha == cv_glmnet$bestTune$alpha, lambda == cv_glmnet$bestTune$lambda)
pred <- predict(cv_glmnet, X)

# compute RMSE of transformed predicted, for comparable RMSE
RMSE(exp(pred), exp(Y))

# make reproducible
set.seed(2)

# split training and test data
data_split<-initial_split(Metab_Pred,strata = "GPP_mgO2perLperD")
ames_train<-training(data_split)
ames_test<-testing(data_split)
# create H2O objects
train_h2o <- as.h2o(ames_train)
test_h2o <- as.h2o(ames_test)
# assign predictor (X) and response (Y) variables
Y <- "GPP_mgO2perLperD"
X <- setdiff(names(train_h2o), Y)

# build final model using parameters from model fitting procedure
best_glm <- h2o.glm(
  y = Y, x = X, training_frame = train_h2o, alpha = 0.75,
  lambda = 0.01,validation_frame = test_h2o,
  standardize = TRUE,
  remove_collinear_columns = TRUE, nfolds = 3, fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE, seed = 123
)

# RMSE for final model on hold-out test data
h2o.performance(best_glm, newdata = test_h2o)@metrics$RMSE
#for more information, see 
# https://bradleyboehmke.github.io/HOML/regularized-regression.html#implementation


#### 4. Nested CV Assessment of eXtreme Gradient Boosting (XGB) Model ####

#BACKGROUND from https://bradleyboehmke.github.io/HOML/gbm.html:
# -- Where random forests build ensemble of deep independent trees, gradient boosting models
# build an ensemble of shallow trees in sequence with each tree learning and improving on the
# previous one. shallow trees are weak predictive models, but they can be "boosted" to make a 
# powerful "committee" 
# -- Boosting is a general algorithm for building an ensemble out of simpler models (typically 
# decision trees), it is more effectively applied to models with high bias and low variability
# -- A sequential ensemble approach: boosting addresses the bias-variance tradeoff by starting
# with a weak model (decision tree with only a few splits), and sequentially boosts its 
# performance by building new trees that try to fix up where the previous one made the biggest
# mistakes (each new tree focuses on training rows where the previous tree had the largest
# prediction errors).
# -- Boosting is a general framework that iteratively improves any weak learning model (you can
# use any base learner), but they almost always use decision trees
# the name GRADIENT boosting machine comes from the use of generalized loss functions other than SSE
# it is considered a gradient descent algorithm - tweak parameters iteratively in order to minimize
# a cost function. 
# -- Boosting Hyperparameters: 1) number of trees -- too many and you overfit; 2) learning rate
# determines contribution of each tree on the final outcome (too small = never reach minimum loss 
# function and high computing time; too big = skip over minimum)
# Tree Hyperparameters: 1) tree depth, (small trees, 1 is not uncommon) 2) minimum number of obs
# -- Stochastic gradient boosting helps reduce the change of getting stuck in local minimas, plateaus, etc
# A high error of test data may indicate overfitting --> XGBoost may be the way to go. 
# XGBoost has additional hyperparameters that can help reduce the chances of overfitting, less prediction
# variability, and improved accuracy
# -- Hyperparameters: 1) gamma = control complexity of a tree, specifies a min loss reduction required to
# make another partition. 
#  alpha and lamda regularization parameters limits how extreme the weights (or influence) of
# the leaves in a tree can become

# XGBoost Packages
library(recipes)
library(xgboost)
library(vtreat)
library(caret)

#initiate h2o cluster
h2o.init(max_mem_size = "10g")

#assign training data, response, and predictor variables
train_h2o <- as.h2o(ames_train)
response <- "GPP_mgO2perLperD"
predictors <- setdiff(colnames(ames_train), response)

# function to set up random seeds from http://jaehyeon-kim.github.io/2015/05/Setup-Random-Seeds-on-Caret-Package.html
setSeeds <- function(method = "cv", numbers = 1, repeats = 1, tunes = NULL, seed = 1237) {
  #B is the number of resamples and integer vector of M (numbers + tune length if any)
  B <- if (method == "cv") numbers
  else if(method == "repeatedcv") numbers * repeats
  else NULL
  if(is.null(length)) {
    seeds <- NULL
  } else {
    set.seed(seed = seed)
    seeds <- vector(mode = "list", length = B)
    seeds <- lapply(seeds, function(x) sample.int(n = 1000000, size = numbers + ifelse(is.null(tunes), 0, tunes)))
    seeds[[length(seeds) + 1]] <- sample.int(n = 1000000, size = 1)
  }
  # return seeds
  seeds
}

# helper function for the plots from https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, 
                                      probs = probs), 
                             min(x$results$RMSE))) +
    theme_bw()
}

# make reproducible
set.seed(1)
# fold data
folds<-createFolds(Metab_Pred$GPP_mgO2perLperD,k = 7)

# create dataframe to save nested cv results
save_XGB<-data.frame(matrix(ncol = 9,nrow=7))
x<-c("gamma","eta","maxdepth","column_sample","row_sample","nrounds","minchild","innerRMSE","outerRMSE")
colnames(save_XGB)<- x

#nested cv assessment of model fitting procedure for XGB model
for(i in 1:7){
  ames_train<-Metab_Pred[-folds[[i]],]
  ames_test<-Metab_Pred[folds[[i]],]
  xgb_prep <- recipe(GPP_mgO2perLperD ~ ., data = ames_train) %>%
    step_integer(all_nominal()) %>%
    prep(training = ames_train, retain = TRUE) %>%
    juice()
  
  X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "GPP_mgO2perLperD")])
  Y <- xgb_prep$GPP_mgO2perLperD
  
  ## tune for XGB hyperparameters from https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret
  # XGBoost will only work with numeric vectors, i.e. we have to convert categorical variables into a 
  # set of indicator variables, use vtreat package

  treat_plan <- vtreat::designTreatmentsZ(
    dframe = ames_train, # training data
    varlist = colnames(ames_train), # input variables = all training data columns
    codeRestriction = c("clean", "isBAD", "lev"), # derived variables types (drop cat_P)
    verbose = FALSE) # suppress messages
  #We can examine the derived variables is from the scoreFrame component of the created treatment plan:
  score_frame <- treat_plan$scoreFrame %>% 
    select(varName, origName, code)
  head(score_frame)
  unique(score_frame$code) # clean stands for cleaned numerical variable; lev is a binary indicator whether
  # a particular value of that categorical variable was present
  # list of variables without the target variable
  
  tr_treated <- vtreat::prepare(treat_plan, ames_train)
  te_treated <- vtreat::prepare(treat_plan, ames_test)
  
  tr_treated$GPP_mgO2perLperD <- log(tr_treated$GPP_mgO2perLperD)
  te_treated$GPP_mgO2perLperD <- log(te_treated$GPP_mgO2perLperD)
  dim(tr_treated)
  
  #Next, we'll ensure that the y-variables follow the same distribution:
  ggplot2::qplot(te_treated$GPP_mgO2perLperD, main="Hold-out Set") + 
    geom_histogram(colour="black", fill="grey",binwidth = 0.1) + theme_bw() 
  ggplot2::qplot(tr_treated$GPP_mgO2perLperD, main="Training Set") + 
    geom_histogram(colour="black", fill="grey",binwidth = 0.1) + theme_bw() 
  
  input_x <- as.matrix(select(tr_treated, -GPP_mgO2perLperD))
  input_y <- tr_treated$GPP_mgO2perLperD
  
  # tune for XGB hyperparameters from https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret
  # cross validation
  cvSeeds <- setSeeds(method = "cv", numbers = 3, tunes = 330, seed = 1237)
  c('B + 1' = length(cvSeeds), M = length(cvSeeds[[1]]))
  nrounds <- 350
  # note to start nrounds from 200, as smaller learning rates result in errors so
  # big with lower starting points that they'll mess the scales
  # Next, as the maximum tree depth is also depending on the number of iterations and the learning rate,
  # we want to experiment with it at this point to narrow down the possible hyperparameters. 
  tune_grid <- expand.grid(
    nrounds = seq(from = 20, to = nrounds, by = 10),
    eta = c(0.025,0.05, 0.1),
    max_depth = c(3, 5, 7),
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  )
  
  tune_control <- caret::trainControl(
    method = "cv", # cross-validation
    number = 3, # with n folds 
    seeds = cvSeeds, #set seeds for reproducibility
    #index = createFolds(tr_treated$Id_clean), # fix the folds
    verboseIter = FALSE, # no training log
    allowParallel = FALSE # FALSE for reproducible results 
  )
  
  set.seed(1)
  xgb_tune <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid,
    method = "xgbTree",
    verbose = TRUE
  )

  tuneplot(xgb_tune)
  xgb_tune$bestTune
  best.fit<-xgb_tune$results[which.min(xgb_tune$results$RMSE),] 
  ind.innerRMSE<-best.fit$RMSE
  
  
  ind.nrounds<-best.fit$nrounds
  ind.maxdepth<-best.fit$max_depth
  ind.learnrate<-best.fit$eta

  #Then, fix maximum depth and minimum child weight:
  tune_grid2 <- expand.grid(
    nrounds = seq(from = 20, to = ind.nrounds, by = 10),
    eta = ind.learnrate,
    max_depth = c(1, 2, 3),
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = c(1, 2, 3),
    subsample = 1
  )
  set.seed(1)
  xgb_tune2 <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid2,
    method = "xgbTree",
    verbose = TRUE
  )
  
  tuneplot(xgb_tune2)

  best_fit2 <-xgb_tune2$results[which.min(xgb_tune2$results$RMSE),] 
  ind.maxdepth<-best_fit2$max_depth
  ind.minchild<-best_fit2$min_child_weight

  tune_grid3 <- expand.grid(
    nrounds = seq(from = 20, to = ind.nrounds, by = 10),
    eta = ind.learnrate,
    max_depth = xgb_tune2$bestTune$max_depth,
    gamma = 0,
    colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
    min_child_weight = ind.minchild,
    subsample = c(0.5, 0.75, 1.0)
  )
  
  set.seed(1)
  xgb_tune3 <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid3,
    method = "xgbTree",
    verbose = TRUE
  )
  
  tuneplot(xgb_tune3, probs = .95)
  
  best.fit3<-xgb_tune3$results[which.min(xgb_tune3$results$RMSE),] 
  ind.colsamplebytree<-best.fit3$colsample_bytree
  ind.subsample<-best.fit3$subsample
  # Next, we again pick the best values from previous step, and now will see whether changing the gamma 
  # has any effect on the model fit:
  tune_grid4 <- expand.grid(
    nrounds = seq(from = 20, to = 200, by = 10),
    eta = ind.learnrate,
    max_depth = ind.maxdepth,
    gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
    colsample_bytree = ind.colsamplebytree,
    min_child_weight = ind.minchild,
    subsample = ind.subsample
  )
  set.seed(1)
  xgb_tune4 <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid4,
    method = "xgbTree",
    verbose = TRUE
  )
  
  tuneplot(xgb_tune4)
 
  best.fit4<-xgb_tune4$results[which.min(xgb_tune4$results$RMSE),] 
  ind.gamma<-best.fit4$gamma
  
  #Now, we have tuned the hyperparameters and can start reducing the learning rate to get to the final model:
  tune_grid5 <- expand.grid(
    nrounds = seq(from = 10, to = ind.nrounds, by = 10),
    eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
    max_depth = ind.maxdepth,
    gamma = ind.gamma,
    colsample_bytree = ind.colsamplebytree,
    min_child_weight = ind.minchild,
    subsample = ind.subsample
  )
  
  set.seed(1)
  xgb_tune5 <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid5,
    method = "xgbTree",
    verbose = TRUE
  )
  
  tuneplot(xgb_tune5)


  best.fit5<-xgb_tune5$results[which.min(xgb_tune5$results$RMSE),] 
  ind.nrounds<-best.fit5$nrounds
  save_XGB$nrounds[i]<-ind.nrounds
  
  ind.learnrate<-best.fit5$eta
  save_XGB$eta[i]<-ind.learnrate
  
  ind.maxdepth<-best.fit5$max_depth
  save_XGB$maxdepth[i]<-ind.maxdepth
  
  ind.gamma<-best.fit5$gamma
  save_XGB$gamma[i]<-ind.gamma
  
  ind.colsamplebytree<-best.fit5$colsample_bytree
  save_XGB$column_sample[i]<-ind.colsamplebytree
  
  ind.minchild<-best.fit5$min_child_weight
  save_XGB$minchild[i]<-ind.minchild
  
  ind.subsample<-best.fit5$subsample
  save_XGB$row_sample[i]<-ind.subsample
  
  #Now that we have determined the parameters we want to use, we will use the training 
  # data (excluding the hold-out set which we will soon use to measure the model performance) 
  # without resampling to fit the model:
  (final_grid <- expand.grid(
    nrounds = xgb_tune5$bestTune$nrounds,
    eta = xgb_tune5$bestTune$eta,
    max_depth = xgb_tune5$bestTune$max_depth,
    gamma = xgb_tune5$bestTune$gamma,
    colsample_bytree = xgb_tune5$bestTune$colsample_bytree,
    min_child_weight = xgb_tune5$bestTune$min_child_weight,
    subsample = xgb_tune5$bestTune$subsample
  ))
   
  train_control <- caret::trainControl(
    method = "none",
    seeds = cvSeeds, #set seeds for reproducibility
    verboseIter = TRUE, # no training log
    allowParallel = FALSE # FALSE for reproducible results 
  )
  set.seed(1)
  (xgb_model <- caret::train(
    x = input_x,
    y = input_y,
    trControl = train_control,
    tuneGrid = final_grid,
    method = "xgbTree",
    verbose = TRUE
  ))
  
  #  RMSE in auto-results is not comparable to other variables, because we log transformed Y:
  pred <- predict(xgb_model, input_x)
  
  # compute RMSE of transformed predicted
  rmse<-RMSE(exp(pred), exp(input_y))
  save_XGB$innerRMSE[i]<-rmse
  
  #By testing the performance with the hold-out set, we can see the effects that the tuning had over the two baseline models:
  
  holdout_x <- as.matrix(select(te_treated, -GPP_mgO2perLperD))
  holdout_y <- te_treated$GPP_mgO2perLperD
  
  pred.t <- predict(xgb_model, holdout_x)

  rmse.t<-RMSE(exp(pred.t), exp(holdout_y))
  save_XGB$outerRMSE[i]<-rmse.t
  
}
# 'save_XGB' generated in the for loop above saves all of the nested cv assessment values for
# the XGB model fitting procedure

#### 4a. Fit Final eXtreme Gradient Boosting (XGB) Model ######

# then we run the above algorithm on the full dataset to get hyperparameters for final h2o.xgb model

# prep data
xgb_prep <- recipe(GPP_mgO2perLperD ~ ., data = Metab_Pred) %>%
  step_integer(all_nominal()) %>%
  prep(training = Metab_Pred, retain = TRUE) %>%
  juice()

#assign predictor (X) and response (Y) variables
X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "GPP_mgO2perLperD")])
Y <- xgb_prep$GPP_mgO2perLperD

## tune for XGB hyperparameters from https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret
# (AS ABOVE IN SECTION 4): XGBoost will only work with numeric vectors, i.e. we have to convert categorical variables into a 
# set of indicator variables, use vtreat package
treat_plan <- vtreat::designTreatmentsZ(
  dframe = Metab_Pred, # training data
  varlist = colnames(Metab_Pred), # input variables = all training data columns
  codeRestriction = c("clean", "isBAD", "lev"), # derived variables types (drop cat_P)
  verbose = FALSE) # suppress messages
#We can examine the derived variables is from the scoreFrame component of the created treatment plan:
score_frame <- treat_plan$scoreFrame %>% 
  select(varName, origName, code)
head(score_frame)
unique(score_frame$code) # clean stands for cleaned numerical variable; lev is a binary indicator whether
# a particular value of that categorical variable was present
# list of variables without the target variable

# assign treated data
tr_treated <- vtreat::prepare(treat_plan, Metab_Pred)

#log transform y
tr_treated$GPP_mgO2perLperD <- log(tr_treated$GPP_mgO2perLperD)

# assign x and y
input_x <- as.matrix(select(tr_treated, -GPP_mgO2perLperD))
input_y <- tr_treated$GPP_mgO2perLperD

# tune for XGB hyperparameters from https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret
# cross validation
cvSeeds <- setSeeds(method = "cv", numbers = 3, tunes = 330, seed = 1237)
c('B + 1' = length(cvSeeds), M = length(cvSeeds[[1]]))
nrounds <- 350
# note to start nrounds from 200, as smaller learning rates result in errors so
# big with lower starting points that they'll mess the scales
# Next, as the maximum tree depth is also depending on the number of iterations and the learning rate,
# we want to experiment with it at this point to narrow down the possible hyperparameters. 
tune_grid <- expand.grid(
  nrounds = seq(from = 20, to = nrounds, by = 10),
  eta = c(0.025,0.05, 0.1),
  max_depth = c(3, 5, 7),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 3, # with n folds 
  seeds = cvSeeds, #set seeds for reproducibility
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = FALSE # FALSE for reproducible results 
)

set.seed(1)
xgb_tune <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  tuneGrid = tune_grid,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune)
xgb_tune$bestTune
best.fit<-xgb_tune$results[which.min(xgb_tune$results$RMSE),] 
Find.innerRMSE<-best.fit$RMSE
Find.nrounds<-best.fit$nrounds
Find.maxdepth<-best.fit$max_depth
Find.learnrate<-best.fit$eta

#Then, fix maximum depth and minimum child weight:
tune_grid2 <- expand.grid(
  nrounds = seq(from = 20, to = 200, by = 10),
  eta = Find.learnrate,
  max_depth = c(1, 2, 3),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3),
  subsample = 1
)
set.seed(1)
xgb_tune2 <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  tuneGrid = tune_grid2,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune2)

best_fit2 <-xgb_tune2$results[which.min(xgb_tune2$results$RMSE),] 
Find.maxdepth<-best_fit2$max_depth
Find.minchild<-best_fit2$min_child_weight

tune_grid3 <- expand.grid(
  nrounds = seq(from = 20, to = 200, by = 10),
  eta = Find.learnrate,
  max_depth = Find.maxdepth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = Find.minchild,
  subsample = c(0.5, 0.75, 1.0)
)

set.seed(1)
xgb_tune3 <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  tuneGrid = tune_grid3,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune3, probs = .95)

best.fit3<-xgb_tune3$results[which.min(xgb_tune3$results$RMSE),] 
Find.colsamplebytree<-best.fit3$colsample_bytree
Find.subsample<-best.fit3$subsample
# Next, we again pick the best values from previous step, and now will see whether changing the gamma 
# has any effect on the model fit:
tune_grid4 <- expand.grid(
  nrounds = seq(from = 20, to = 200, by = 10),
  eta = Find.learnrate,
  max_depth = Find.maxdepth,
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  colsample_bytree = Find.colsamplebytree,
  min_child_weight = Find.minchild,
  subsample = Find.subsample
)
set.seed(1)
xgb_tune4 <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  tuneGrid = tune_grid4,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune4)

best.fit4<-xgb_tune4$results[which.min(xgb_tune4$results$RMSE),] 
Find.gamma<-best.fit4$gamma

#Now, we have tuned the hyperparameters and can start reducing the learning rate to get to the final model:
tune_grid5 <- expand.grid(
  nrounds = seq(from = 10, to = 200, by = 10),
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = Find.maxdepth,
  gamma = Find.gamma,
  colsample_bytree = Find.colsamplebytree,
  min_child_weight = Find.minchild,
  subsample = Find.subsample
)

set.seed(1)
xgb_tune5 <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  tuneGrid = tune_grid5,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune5)

best.fit5<-xgb_tune5$results[which.min(xgb_tune5$results$RMSE),] 

# refresh h2o cluster
h2o.shutdown(prompt = FALSE)
h2o.init()

# for reproducibility
set.seed(2)

#split training and test data
data_split<-initial_split(Metab_Pred,strata = "GPP_mgO2perLperD")
ames_train<-training(data_split)
ames_test<-testing(data_split)
train_h2o <- as.h2o(ames_train)
test_h2o <- as.h2o(ames_test)

#assign predictor and response variables
Y <- "GPP_mgO2perLperD"
X <- setdiff(names(train_h2o), Y)
# Train & cross-validate final XGBoost model using parameters from model fitting procedure
best_xgb <- h2o.xgboost(
  x = X, y = Y, training_frame = train_h2o,  learn_rate = ind.learnrate,
  min_child_weight = ind.minchild,validation_frame = test_h2o,
  col_sample_rate_per_tree = ind.colsamplebytree,min_split_improvement = ind.gamma,
  max_depth = ind.maxdepth, sample_rate = ind.subsample, 
  nfolds = 3, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 123, stopping_rounds = ind.nrounds,
  stopping_metric = "RMSE",stopping_tolerance = 0
)

# extract RMSE
h2o.performance(best_xgb, newdata = test_h2o)@metrics$RMSE


###### 5. Build final ensemble model with rf, glm, xgb ########
# details/information from https://bradleyboehmke.github.io/HOML/stacking.html
# Train a stacked tree ensemble

# make reproducible
set.seed(2)

# split training and test data
data_split<-initial_split(Metab_Pred,strata = "GPP_mgO2perLperD")
ames_train<-training(data_split)
ames_test<-testing(data_split)
train_h2o <- as.h2o(ames_train)
test_h2o <- as.h2o(ames_test)

# assign predictor and response variables
Y <- "GPP_mgO2perLperD"
X <- setdiff(names(train_h2o), Y)

###!!! Before you run this, you need to make sure each base learner is saved in the h2o environment
# (if not, re-run the final model code for each base learner in same h2o session)
ensemble_tree <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o, model_id = "my_tree_ensemble",
  validation_frame = test_h2o,
  base_models = list( best_rf, best_xgb, best_glm),
  metalearner_algorithm = 'AUTO'
)

# Get results from base learners
get_rmse <- function(model) {
  results <- h2o.performance(model, newdata = test_h2o)
  results@metrics$RMSE
}
list(best_rf, best_glm, best_xgb) %>% #best_glm, 
  purrr::map_dbl(get_rmse)
# Stacked results
h2o.performance(ensemble_tree, newdata = test_h2o)@metrics$RMSE
h2o.performance(ensemble_tree, newdata = test_h2o)@metrics$r2

## make sure baselearners are not too correlated w/each other (no more than 0.9)
mod.cor <-data.frame(
  GLM_pred = as.vector(h2o.getFrame(best_glm@model$cross_validation_holdout_predictions_frame_id$name)),
  RF_pred = as.vector(h2o.getFrame(best_rf@model$cross_validation_holdout_predictions_frame_id$name)),
  XGB_pred = as.vector(h2o.getFrame(best_xgb@model$cross_validation_holdout_predictions_frame_id$name))
) %>% cor()
mod.cor
# https://bradleyboehmke.github.io/HOML/stacking.html#stacking-existing

# See Boehmke and Greenwell book for examples on visualizing models!

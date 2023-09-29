#Load packages and data
library(tidyverse)
#The tidyverse package is an "umbrella-package" that installs several packages useful for data analysis which work together well such as tidyr(swiftly convert between different data formats), dplyr(data manipulation), ggplot2, tibble, etc. 
library(dlookr) # **Data Diagnosis, Exploration, transform. compare before and after results  
library(gridExtra)
library(corrplot)

library(caTools)
library(caret)
library(mlbench)
library("gbm")

library(Metrics)

# 1. Problem Definition
# 1.1 Load the Dataset
load("D:/work/job/prepare/Movie/movies.RData")

# take a peek at the first 5 rows of the data
head(movies,5)

# 1.2 Data Preparation
# Data Cleaning
## adding new features
# Column for if movie was released during oscar season
movies <- movies %>% 
  mutate(oscar_season = as.factor(ifelse(thtr_rel_month %in% c('10', '11', '12'), 'yes', 'no')))

# Column for if movie was released during summer season
movies <- movies %>% 
  mutate(summer_season = as.factor(ifelse(thtr_rel_month %in% c('6', '7', '8'), 'yes', 'no')))


# extracting meaningful features
movies_new <- movies %>% select(title_type, genre, runtime, mpaa_rating, imdb_rating, imdb_num_votes, critics_rating, critics_score, audience_rating, audience_score, best_pic_win, best_actor_win, best_actress_win, best_dir_win, top200_box, oscar_season, summer_season)


# Let’s check and deal with missing values.
cat("Number of missing value:", sum(is.na(movies_new)), "\n")
plot_na_pareto(movies_new)
# plot percentage of missing values per feature {library(naniar)}
# gg_miss_var(movies_new,show_pct=TRUE)

# Delete rows where CustomerID is missing
movies_new<-movies_new %>%
  na.omit(runtime)

## We have added a couple new columns of interest,like oscar_season, summer_season, extracted columns and removed missing values. 

## 1.3 Split the data sets into a training Dataset and a testing Dataset, 
# library(caTools)
set.seed(101) 
sample = sample.split(movies_new$audience_score, SplitRatio = .8)
train = subset(movies_new, sample == TRUE)
test  = subset(movies_new, sample == FALSE)


#2. Analyze Data
# 2.1 Descriptive Statistics
dim(train)
# summarize feature distributions
summary(train)
# Let's also look at the data types of each feature
sapply(train, class)

# Correlation between categorical feature and audience score
names <- names(Filter(is.factor,train))
plot_list <- list()

for (name in names) {
  plot <- ggplot(data = train, aes_string(x = name, y = train$audience_score, fill = name)) + 
    geom_boxplot(show.legend=FALSE) + xlab(name) + ylab('Audience Score')  # +ggtitle(paste("Audience Score in", name, sep=" "))
  # plot + theme(plot.title = element_text(size = 10), axis.text.x = element_text(angle = 20))    
  plot_list[[name]] <- plot
}
plot_grob <- arrangeGrob(grobs=plot_list, ncol=3)
grid.arrange(plot_grob)

by(train$audience_score, train$top200_box, summary)

## As we know, if median value lines from two different boxplots do not overlap, then there is a statistically significant difference between the medians.
## "best_actor_win" "best_actress_win" "best_dir_win"     "top200_box"       "oscar_season"     "summer_season" does not seem to have much impact on audience_score.

# Bar plot of categorical features  
plot_list2 <- list()
for (name in names[1:6]) {
  plot <- ggplot(aes_string(x=name), data=train) + geom_bar(aes(y=100*(..count..)/sum(..count..))) + ylab('percentage')  +
    ggtitle(name) + coord_flip()
  plot_list2[[name]] <- plot
}
plot_grob2 <- arrangeGrob(grobs=plot_list2, ncol=2)
grid.arrange(plot_grob2)
##  "critics_rating"  "audience_rating"  have observations spread out fairly evenly over all categories shows high variability, while  "title_type", "genre",  "mpaa_rating" and "best_pic_win"  where most observations are only in one or a handful of categories displays low variability.

# Histogram of Numeric attributes
names_n <- names(Filter(is.numeric,train))

hisplot_n <- list()
for (name in names_n) {
  plot <- ggplot(data = train, aes_string(x = name)) + 
    geom_histogram(aes(y=100*(..count..)/sum(..count..)), color='black', fill='white') + ylab('percentage') + ggtitle(name) 
  hisplot_n[[name]] <- plot
}
hisplot_grobn <- arrangeGrob(grobs=hisplot_n, ncol=2)
grid.arrange(hisplot_grobn)


# The distribution of attribute imdb_num_votes is right skewed, will be shifted by using The BoxCox transform to reduce the skew and make it more Gaussian 

# Correlation between numerical attributes
corr.matrix <- cor(train[names_n])
corrplot(corr.matrix, main="\n\nCorrelation Plot of numerical attributes", method="number")

# Summary of Ideas
# There is a lot of structure in this dataset. We need to think about transforms that we could use later to better expose the structure which in turn may improve modeling accuracy. So far it would be worth trying:
# 1. Feature selection and removing the most correlated attributes.
# 2. standardizing the dataset to reduce the effect of differing scales and distributions.
# 3. Box-Cox transform to see if flattening out some of the distributions improves accuracy.


# 3. Data transformation
# library(caret)
# library(mlbench)
sapply(train[names_n],class)
train$imdb_num_votes <- as.numeric(train$imdb_num_votes)
train_num <- as.data.frame(train[names_n])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(train_num, method=c("center","scale","BoxCox")) # Standardize and power transform
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, train_num)
# summarize the transformed dataset
summary(transformed)

train_cat <- as.data.frame(train[names[1:6]])

# combine two datasets in r
train_trans <-cbind(train_cat, transformed)

# 4: Modeling
# 4.1 Statistical regression modeling
full_model <- lm(audience_score~ title_type + genre + mpaa_rating + critics_rating + audience_rating + best_pic_win + 
                   runtime + imdb_rating + imdb_num_votes, data=train_trans)
# full_model <- lm(audience_score~. , data=train_trans)
summary(full_model)

# Backward Stepwise 
newmodel <- step(full_model, direction = "backward", trace=FALSE) 
summary(newmodel)

# Model diagnostics
p1<-ggplot(data = newmodel, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  xlab("Fitted values") +
  ylab("Residuals")

p2<-ggplot(data = newmodel, aes(x = .resid)) +
  geom_histogram(binwidth = 0.05, fill='white', color='black') +
  xlab("Residuals")

p3<-ggplot(data = newmodel, aes(sample = .resid)) +
  stat_qq()

grid.arrange(p1,p2,p3, ncol=2)


# 4.2 Machine learning modeling

# Run algorithms using 10-fold cross validation
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"


# 4.2.1 Evaluate Algorithms: Baseline

# lm
set.seed(7)
fit.lm <- train(audience_score~., data=train_trans, method="lm", metric=metric, trControl=trainControl)
# GLM
set.seed(7)
fit.glm <- train(audience_score~., data=train_trans, method="glm", metric=metric,trControl=trainControl)
# GLMNET
set.seed(7)
fit.glmnet <- train(audience_score~., data=train_trans, method="glmnet", metric=metric,trControl=trainControl)
# SVM
set.seed(7)
fit.svm <- train(audience_score~., data=train_trans, method="svmRadial", metric=metric,trControl=trainControl)
# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(audience_score~., data=train_trans, method="rpart", metric=metric,trControl=trainControl)
# KNN
set.seed(7)
fit.knn <- train(audience_score~., data=train_trans, method="knn", metric=metric, trControl=trainControl)
# Compare algorithms
results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm,
                                  CART=fit.cart, KNN=fit.knn))
summary(results)
dotplot(results)

# 4.2.2 Evaluate Algorithms: Feature Selection
# Find and drop attributes that are highly corrected
set.seed(7)
cutoff <- 0.70
correlations <- cor(train_trans[,7:10])
highlyCorrelated <- findCorrelation(correlations, cutoff=cutoff)
for (value in highlyCorrelated) {
  print(names(train_trans[,7:10])[value])
}

# We can see that we have dropped 1 attributes: imdb_rating.

# create a new dataset without highly corrected features
datasetFeatures <- train_trans[,-highlyCorrelated]
dim(datasetFeatures)



# lm
set.seed(7)
fit.lm <- train(audience_score~., data=datasetFeatures, method="lm", metric=metric, trControl=trainControl)
# GLM Generalized Linear Regression
set.seed(7)
fit.glm <- train(audience_score~., data=datasetFeatures, method="glm", metric=metric,trControl=trainControl)
# GLMNET Penalized Linear Regression
set.seed(7)
fit.glmnet <- train(audience_score~., data=datasetFeatures, method="glmnet", metric=metric,trControl=trainControl)
# SVM
set.seed(7)
fit.svm <- train(audience_score~., data=datasetFeatures, method="svmRadial", metric=metric,trControl=trainControl)
# CART Classification and Regression Trees
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(audience_score~., data=datasetFeatures, method="rpart", metric=metric,trControl=trainControl)
# KNN
set.seed(7)
fit.knn <- train(audience_score~., data=datasetFeatures, method="knn", metric=metric, trControl=trainControl)
# Compare algorithms
feature_results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm,
                                  CART=fit.cart, KNN=fit.knn))
summary(feature_results)
dotplot(feature_results)

# It looks like GLMNET has the lowest RMSE and high Rsquared, followed closely by the other linear algorithms, SVM CART and KNN appear to be in the same ball park and slightly worse error and Rsquared.

# Improve Results With Tuning
# Let's look at the default parameters already adopted.
print(fit.glmnet)

# Make a custom tuning grid.  glmnet is capable of fitting 2 different kinds of penalized models, and it has 2 tuning parameters:
# alpha
# Ridge regression (or alpha = 0)
# Lasso regression (or alpha = 1)
#lambda
# the strength of the penalty on the coefficients

grid <- expand.grid(alpha = 0:1, lambda = seq(0.0001, 1, length = 10))

# Fit a model
tune.glmnet <- train(audience_score~., data=train_trans, method = "glmnet", metric=metric,
               tuneGrid = grid, trControl = trainControl)
print(tune.glmnet)
plot(tune.glmnet)


# Ensemble Methods
# Random Forest
set.seed(7)
fit.rf <- train(audience_score~., data=train_trans,  method="rf", metric=metric, trControl=trainControl)
# Stochastic Gradient Boosting
set.seed(7)
fit.gbm <- train(audience_score~., data=train_trans,  method="gbm", metric=metric, trControl=trainControl, verbose=FALSE)

# Compare algorithms
ensembleResults <- resamples(list(RF=fit.rf, GBM=fit.gbm))
summary(ensembleResults)
dotplot(ensembleResults)

# We can see that Gradient Boosting was the most accurate with an RMSE that was lower than that achieved by tuning glmnet.
# look at parameters used for Gradient Boosting
print(fit.gbm)

grid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                        n.trees = seq(50, 500, by = 50),
                        n.minobsinnode = 10,
                        shrinkage = c(0.01, 0.1))
tune.gbm <- train(audience_score~., data=train_trans,  method="gbm", metric=metric, tuneGrid = grid, trControl=trainControl, verbose=FALSE)
print(tune.gbm)
plot(tune.gbm)

# Finalize Model
# train the final model
# library("gbm")
finalModel <- gbm(audience_score~., data=train_trans, n.trees = 400, interaction.depth = 7, shrinkage = 0.01, n.minobsinnode = 10)
summary(finalModel)

# use final model to make predictions on the testing dataset
# testing Data transformation
names_n
train$imdb_num_votes <- as.numeric(train$imdb_num_votes)
test_num <- as.data.frame(test[names_n])
# transform the testing dataset using the parameters
transformed_test <- predict(preprocessParams, test_num)
# summarize the transformed dataset
summary(transformed_test)

test_cat <- as.data.frame(test[names[1:6]])

# combine two datasets in r
test_trans <-cbind(test_cat, transformed_test)

predictions <- predict(finalModel, newdata=test_trans)
# calculate RMSE
test_Y <- test_trans[,11]
rmse <- rmse(predictions, test_Y)

#R SQUARED error metric -- Coefficient of Determination
RSQUARE = function(y_actual,y_predict){
  cor(y_actual,y_predict)^2
}
r2 <- RSQUARE(test_Y, predictions)
# print(paste("RMSE:", rmse))

sprintf("RMSE: %#.2f", rmse)
sprintf("R-squared:  %#.2f", r2)


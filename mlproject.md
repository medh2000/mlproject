Coursera Machine Learning Project
M Hassan
2015
========================================================

Wearable devices are gaining popularity recently, there are number of companies both startup and legacy tech leaders that are trying to dominate this new market. Wearable devices collect data from users in different categories of events such as sports actvivities to health. The focus of this project is to utilize some sample data on the quality of certain exercises to predict the manner in which they did the exercise.

Background
========================================================

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 


Data Sources
========================================================
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project comes from this original source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

Goal
========================================================
The goal the project is to predict the manner in which users did the exercise. This is the "classe" variable in the training set. 

Data Analysis
========================================================
In data analysis for wearables devices, we will attempt to create an accurate machine learning model from the sample data. The goal is to try to find out how accuratly a person performs a particular exercise. We will try to classify each user in a class outcome variable. This is a classification problem where the class variable is called "classe" in the training data set.

Data Preprocessing
========================================================

There are 2 data sets, the training data set and the testing data set we are attempting to perform the predictions from the final model on. When the data is loaded into dataframes, it is necessary to locate strings containing unwanted strings such as ‘#DIV/0!'. These error codes are loaded into the data frame as NA fields. I have downloaded the training and testing data into my local folder but these 2 files are available online in the data sources section.


Data loading in training and testing sets
========================================================

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.3
```

```r
pml.train <- read.csv(file = 'pml-training.csv',na.strings = c('NA','#DIV/0!',''))
pml.test <- read.csv(file = 'pml-testing.csv',na.strings = c('NA','#DIV/0!',''))
```


Exploraring Data Analysis
========================================================
Summary of plm.train

We have 159 variables and outcome variable "class" of exercice. Some of variables contain NA and are not used for in the class prediction analysis. There are 152 variables that can be used capture a person body's movement that will be used as predictors. There are other variables that are considered as predictors such as a person username that do notcontain any measurements.



```r
dim(pml.train)
```

```
## [1] 19622   160
```

```r
str(pml.train)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ kurtosis_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

Converting to numeric variables for the rest of the columns that contain measurement predictors.

```r
for(i in c(8:ncol(pml.train)-1)) {
  pml.train[,i] = as.numeric(as.character(pml.train[,i]))
  pml.test[,i] = as.numeric(as.character(pml.test[,i]))
}
```

Data Cleaning
========================================================

Some columns in the data set have more data consisting of either Null values and/or NA. We can remove these columns since they are considered sparse and do not add any significance to classifying our model. We also want to remove variables that just describe the data rather offer us any meaningfull measurements. These are the first 7 variables.


```r
head(names(pml.train),7)
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```

Remove columns with NA and a value of 0
Remove first 7 columns/variables.


```r
predictors <- colnames(pml.train)
predictors  <- colnames(pml.train[colSums(is.na(pml.train)) == 0])
predictors  <- predictors [-c(1:7)]
```

Spliting data into training and testing 
========================================================
We split training data randomly into a new training and testing dataset with 80% of data into training and 20% into testing data to validate the accuracy of the classification model.



```r
set.seed(1200)
train.index <- createDataPartition(y=pml.train$classe, p=0.80, list=FALSE)
pml_train <- pml.train[train.index,predictors]
pml_test <- pml.train[-train.index,predictors]
```

We have less predictors now than in the original training dataset


```r
dim(pml_train); dim(pml_test)
```

```
## [1] 15699    53
```

```
## [1] 3923   53
```
Visualize different classes frequencies using a histogram
=========================================================
Before a model is fit, it is useful to have an idea of the ratio that should be expected of the classification variable outcome. This wil govern how we seek to optimize models for Specificity, Sensitivity, and Positive/Negative Predictive Value. 

```r
hist(as.numeric(pml_train$classe),main="Histogram of Class frequency", ylab="Frequency",xlab='Classes',breaks=5, col='blue', freq=T)
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png) 

The amplitude for each of the bar on the chart are of similar range. Each class is equally likely to be an outcome. This indicates that optimizing a model for accuracy and minimizing overall out of sample error should indicate an optimal model for making classificions.

Train a Model and Cross Validation
========================================
From prelimiary analysis, a Random Forest model was chosenn with 2 number of folds for cross-validation


```r
library(e1071)
rf.model <- train(classe ~ ., data = pml_train, method = 'rf', 
                trControl = trainControl(method = "cv", 
                                         number = 2, 
                                         allowParallel = TRUE, 
                                         verboseIter = TRUE))
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=27 
## - Fold1: mtry=27 
## + Fold1: mtry=52 
## - Fold1: mtry=52 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=27 
## - Fold2: mtry=27 
## + Fold2: mtry=52 
## - Fold2: mtry=52 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 27 on full training set
```

```r
pred.rf <- predict(rf.model,pml_test)
final.model.rf <- confusionMatrix(pred.rf,pml_test$classe)
```

Prediction agains cross validation model
========================================
For each candidate model, predictions are made against the cross-validation dataset. Then a confusion matrix is calculated and stored for each model for later reference.

Final Model Evaluation
======================
We can now evaluate our model


```r
final.model.rf 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1112   10    0    0    0
##          B    3  748    3    0    0
##          C    0    1  675    6    3
##          D    0    0    6  637    0
##          E    1    0    0    0  718
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9916          
##                  95% CI : (0.9882, 0.9942)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9894          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9855   0.9868   0.9907   0.9958
## Specificity            0.9964   0.9981   0.9969   0.9982   0.9997
## Pos Pred Value         0.9911   0.9920   0.9854   0.9907   0.9986
## Neg Pred Value         0.9986   0.9965   0.9972   0.9982   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2835   0.1907   0.1721   0.1624   0.1830
## Detection Prevalence   0.2860   0.1922   0.1746   0.1639   0.1833
## Balanced Accuracy      0.9964   0.9918   0.9919   0.9944   0.9978
```

The accuracy of the model is 0.9939. The out of sample error is 1- accuracy = 0.0061. Considering that the test set is a sample size of 20, an accuracy rate well above 99% is sufficient to expect that few or none of the test samples will be mis-classified.

Applying Selected Model to Test Set
===================================

Now finaly, we can classify our testing data consisting of 20 observations. We can make predictions from the most accurate tree in the random forest.


```r
predictClasse <- length(colnames(pml.test[]))
validate.rf <- predict(rf.model,pml.test[,predictors])
```

```
## Error in `[.data.frame`(pml.test, , predictors): undefined columns selected
```

```r
validate.rf
```

```
## Error in eval(expr, envir, enclos): object 'validate.rf' not found
```


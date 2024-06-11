#The libraries used

library(psych) #desc
library(DescTools) #mode
library(caret) #collinearity
library(ggplot2) #ggplot
library(rsample) #Split
library(ROSE) #Balancing dataset
library(FSelector) #cfs
library(Boruta) #Boruta
library(RWeka)
library(C50) #C5
library(rpart) #RPart
library(rpart.plot) #RPart
library(e1071)

#Reading the csv file with the data into R

data = read.csv("/Users/amruguru/Downloads/project_dataset.csv",header = T)

#Converting the data into a dataframe

df = data.frame(data)

--------------------------------------------------------------------------------

##DATA EXPLORATION##

##To find the basic statistical values of the data##

#To find the variance,mean,standard deviation,min,max and range for each column in the data

describe(df)

#To find the mode for each column in the data

for (i in 1:(ncol(df)-1))
{
  # calculating mode of ith column
  
  mod_value <- Mode(df[,i])
  cat(i, ": ",mod_value,"\n")
}

#Check the quality of the data 

#To see the first four rows of the data

head(df)

#To see the last four rows of the data

tail(df)

#To check the datatypes of the attributes

str(df)
ncol(df)

#Visual representation of the class attribute

#Barplot

ggplot(data = df) + geom_bar(mapping = aes(x = o_bullied))

#To find missing values in the data

sum(is.na(data))

#--------------------------------------------------------------------------------

##DATA REDUCTION##

df1 = df
df2 = df1

#Loop for deducting the columns with the same value for all rows

for (i in 1:(ncol(df)-1))
{
  temp = table(df[i])
  if(length(temp)==1)
  {df2 = df2[,-i]}
}

#Number of columns after running the loop 

ncol(df2)

#Loop for deducting the duplicated rows

for (i in 1:(nrow(df)))
{
  df2 = df2[!duplicated(df2[i,])]
}

#Number of rows and columns after running the loop 

ncol(df2)
nrow(df2)

#To find the columns with zero variability and deleting them from our data

cols = nearZeroVar(df2)
df2 = df2[,-cols]

#Number columns after running the loop 

ncol(df2)

#Reducing columns based on collinearity 

corr1 <- cor(df2)
highCorr <- findCorrelation(corr1, cutoff = 0.9,verbose = TRUE)
for(i in highCorr)
  df2 = df2[,-i]

#To find the number of columns in df2

ncol(df2)

#To normalize the given data

df_scaled = df2
df_scaled = scale(df_scaled)

#To find the quartiles and IQR and get rid of columns with no variability and missing values

df_scaled1=df_scaled
for (i in 1:ncol(df_scaled))
{
  summ = summary(df_scaled[,i])
  q1 = summ[2]
  q3 = summ[4]
  IQR= q3 - q1
  cat("df column",i,"q1:",q1,"q3:",q3,"IQR:",IQR, "\n")
  if((q1==0 && q3==0)|(IQR==0)|(is.na(q1)|is.na(q3)))
  {df_scaled1 = df_scaled1[,-i]}
  
}

#Number of columns after removing the columns near zero variability and scaling

ncol(df_scaled1)
nrow(df_scaled1)

#Copy the pre-processed data set into a csv

write.csv(df2,file = "/Users/amruguru/Downloads/preprocessed_data.csv")

#Taking the preprocessed data

data_prep = read.csv("/Users/amruguru/Downloads/preprocessed_data.csv")
data_prep = subset(data_prep, select = -1)

#Splitting the pre-processed dataset

set.seed(31)
split <- initial_split(data_prep, prop = 0.70, strata = o_bullied)
train <- training(split)
test <- testing(split)

#To check if the given training and testing dataset is balanced

hist(train$o_bullied)
table(train$o_bullied)
hist(test$o_bullied)
table(test$o_bullied)

#To balance the given dataset
#Sampling Technique 

attach(data_prep)

#Train
bal_train = ovun.sample(o_bullied~., data = train, method = "over", N = 5342)$data
table(bal_train$o_bullied)
hist(bal_train$o_bullied)

#Test
bal_test = ovun.sample(o_bullied~., data = test, method = "over", N = 2292)$data
table(bal_test$o_bullied)
hist(bal_test$o_bullied)

#--------------------------------------------------------------------------------
  
#Feature selection 

# CFS

subset <- cfs(o_bullied ~., bal_train)
dataset.cfs <- as.simple.formula(subset, "o_bullied")
dataset.cfs
important.features1 = c("V2025A","V2038","V2043","V2050","V2125","V2132","V3012","V3018","V3019","V3062","V3064","V3071","V3072","VS0007","VS0042","VS0046","VS0051","VS0053","VS0124")

#Training and testing dataset based on feature selection (cfs)

trcfs = bal_train[,c(important.features1, "o_bullied")]
tscfs = bal_test[,c(important.features1, "o_bullied")]

##Naive Bayes on CFS

# build a Naïve Bayes model from training dataset
data_nb <- naiveBayes(o_bullied ~ ., data=trcfs)

# test on test dataset
pred <- predict(data_nb, newdata = tscfs, type = "class")

# produce performance measures
performance_measures  <- confusionMatrix(data=pred, reference = as.factor(tscfs$o_bullied))
performance_measures

# calculate performance measures assuming YES is positive
cm <- performance_measures$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa', 'Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k, accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures = calculate_measures(tp, fp, tn, fn)
performance_measures

## J48 on CFS

model <- J48(as.factor(o_bullied)~., trcfs)
predicted <- data.frame(predict(model, tscfs))
colnames(predicted)[1] <- "predicted"
cm <- table(predicted$predicted, tscfs$o_bullied)
cm

sens = 363/(363+98)
sens
spec = 169/(169+170)
spec

# calculate performance measures assuming YES is positive

tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa', 'Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k, accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures = calculate_measures(tp, fp, tn, fn)
performance_measures

## C5.0 on CFS

library(C50)
C5.tree <- C5.0(as.factor(o_bullied)~., data = trcfs)

# test 
pred <- predict(C5.tree, newdata = tscfs, type = "class")
performance_measures  <- confusionMatrix(data=pred, reference = as.factor(tscfs$o_bullied))
performance_measures

# calculate performance measures assuming YES is positive
cm <- performance_measures$table

tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa', 'Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k, accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures = calculate_measures(tp, fp, tn, fn)
performance_measures

## rpart on CFS

library(rpart)
library(rpart.plot)
rpart.tree <- rpart(o_bullied~ ., data = trcfs, method = "class", parms = list(split = "information"))

# plot the tree
prp(rpart.tree, type = 1, extra = 1, under = TRUE, 
    split.font = 1, varlen = -10)

# test 
pred <- predict(rpart.tree, newdata = tscfs, type = "class")
performance_measures  <- confusionMatrix(data=pred, reference = as.factor(tscfs$o_bullied))
performance_measures

# calculate performance measures assuming YES is positive
cm <- performance_measures$table

tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa', 'Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k, accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures = calculate_measures(tp, fp, tn, fn)
performance_measures

#KNN on CFS

train_control <- trainControl(method = "CV",summaryFunction = defaultSummary)
knnGrid <-  expand.grid(k = seq(1, 100, 2))
knnModel <- train(o_bullied~., data = trcfs, method = "knn",
                  trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneGrid = knnGrid)

knnModel
plot(knnModel)

test_pred <- predict(knnModel, newdata = tscfs)
test_pred = as.numeric(test_pred)
str(test_pred)
test_pred = replace(test_pred,test_pred < 0.5,0)
test_pred = replace(test_pred,test_pred>=0.5,1)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$o_bullied))

# calculate performance measures assuming YES is positive

cm <- performance_measures$table
cm 

tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa', 'Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k, accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures = calculate_measures(tp, fp, tn, fn)
performance_measures

#Random Forest Atrribute selection on CFS

#Random Forest on CFS

train_control <- trainControl(method = "CV",summaryFunction = defaultSummary)

ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

mtryValues <- seq(2, ncol(data_prep)-1, by = 1)

set.seed(31)
rfFit <- caret::train(o_bullied~ ., data = bal_train, method = "rf",
                      ntree = 30, tuneGrid = data.frame(mtry = mtryValues),
                      trControl = train_control)
rfFit
plot(rfFit)

## variable importance
imp <- varImp(rfFit)
imp

pred <- predict(rfFit, test)
cm <- caret::confusionMatrix(pred, test$class)
cm

library(pROC)
rfRoc <- roc(response = rfFit$pred$obs,
             predictor = rfFit$pred$Y,
             levels = rev(levels(rfFit$pred$obs)))
plot(rfRoc, main ="ROC curve", print.auc=TRUE)



#SVM on CFS

set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.05), C = seq(1.0, 2.0, by = 0.1))

model <- caret::train(o_bullied~ ., data = trcfs, method = "svmRadial",
                      preProc = c("center", "scale"),
                      trControl = train_control, tuneGrid = svmGrid)
model
plot(model)

pred <- predict(model, tscfs)
pred = as.numeric(pred)
str(pred)
pred = replace(pred,pred < 0.5,0)
pred = replace(pred,pred>=0.5,1)
pred = as.factor(pred)
levels(pred)
pred
performance_measures <- caret::confusionMatrix(pred, as.factor(tscfs$o_bullied))
performance_measures

# calculate performance measures assuming YES is positive

cm <- performance_measures$table
cm 

tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa', 'Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k, accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures = calculate_measures(tp, fp, tn, fn)
performance_measures

## logistic on CFS

logitModel <- glm(o_bullied~ ., data = trcfs, family = "binomial") 
# use predict() with type = "response" to compute predicted probabilities. 
logitModel.pred <- predict(logitModel, tscfs, type = "response")

# performance measures on the test dataset
pred <- factor(ifelse(logitModel.pred >= 0.5, 1, 0))
pred
performance_measures  <- confusionMatrix(data=pred, 
                                         reference = as.factor(tscfs$o_bullied))
performance_measures

# calculate performance measures assuming YES is positive

cm <- performance_measures$table
cm 

tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa', 'Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k, accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures = calculate_measures(tp, fp, tn, fn)
performance_measures

# BORUTA

dataset.boruta <- Boruta(o_bullied~.,data=bal_train)
dataset.boruta
important_features <- getSelectedAttributes(dataset.boruta)
important_features

#Training and testing dataset based on feature selection (boruta)

trboruta = bal_train[,c(important_features, "o_bullied")]
tsboruta = bal_test[,c(important_features, "o_bullied")]

##Naive Bayes on Boruta

# build a Naïve Bayes model from training dataset
data_nb <- naiveBayes(o_bullied ~ ., data=trboruta)
# test on test dataset
pred <- predict(data_nb, newdata = tsboruta, type = "class")
# produce performance measures
performance_measures  <- confusionMatrix(data=pred, reference = as.factor(ts$o_bullied))
performance_measures

## J48 on Boruta

model <- J48(as.factor(o_bullied)~., trboruta)
predicted <- data.frame(predict(model, tsboruta))
colnames(predicted)[1] <- "predicted"
cm <- table(predicted$predicted, tsboruta$o_bullied)
cm

sens = 349/(349+112)
sens
spec = 178/(178+161)
spec

## C5.0 on Boruta

library(C50)
C5.tree <- C5.0(as.factor(trboruta$o_bullied)~., data = trboruta)

# test 
pred <- predict(C5.tree, newdata = tsboruta, type = "class")
levels(pred)
pred
performance_measures  <- confusionMatrix(data=pred, reference = as.factor(tsboruta$o_bullied))
performance_measures

## rpart on Boruta
library(rpart)
library(rpart.plot)
rpart.tree <- rpart(o_bullied~ ., data = trboruta, 
                    method = "class", parms = list(split = "information"))
# plot the tree
prp(rpart.tree, type = 1, extra = 1, under = TRUE, 
    split.font = 1, varlen = -10)

# test 
pred <- predict(rpart.tree, newdata = trboruta, type = "class")
levels(pred)
pred
performance_measures  <- confusionMatrix(data=pred, reference = as.factor(trboruta$o_bullied))
performance_measures

#Random Forest on Boruta

library(randomForest)
rf60 <- randomForest(o_bullied~., data = trboruta) 
pred <- predict(rf60,tsboruta)
levels(pred)
pred
performance_measures  <- confusionMatrix(data=pred,
                                         reference = as.factor(tsboruta$o_bullied))
performance_measures

#KNN on Boruta

train_control <- trainControl(method = "CV",summaryFunction = defaultSummary)
knnGrid <-  expand.grid(k = seq(1, 100, 2))
knnModel <- train(o_bullied~., data = trboruta, method = "knn",
                  trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneGrid = knnGrid)

knnModel
plot(knnModel)

test_pred <- predict(knnModel, newdata = tsboruta)
levels(test_pred)
test_pred
confusionMatrix(test_pred, as.factor(tsboruta$o_bullied))

# INFORMATION GAIN

bal_train <- as.data.frame(unclass(bal_train), stringsAsFactors = TRUE)
bal_train$o_bullied <- factor(bal_train$o_bullied)
dataset.infogain <- InfoGainAttributeEval(o_bullied ~. , data = bal_train)
sorted.features <- sort(dataset.infogain, decreasing = TRUE)
sorted.features[1:30]
important_features2 = c("VS0046","VS0124","VS0070","VS0067","VS0055","VS0059","VS0051","VS0053","VS0049","VS0154","VS0062","VS0031","VS0047","VS0058","VS0155","VS0050","VS0135","VS0125","VS0060","VS0061","VS0022","VS0017","VS0065","VS0064","VS0152","V3020","V3072","VS0057","VS0023","VS0052")

#Training and testing dataset based on feature selection (infogain)

trinfogain = bal_train[,c(important_features2, "o_bullied")]
tsinfogain = bal_test[,c(important_features2, "o_bullied")]

##Naive Bayes on InfoGain

# build a Naïve Bayes model from training dataset
data_nb <- naiveBayes(o_bullied ~ ., data=trinfogain)

# test on test dataset
pred <- predict(data_nb, newdata = tsinfogain, type = "class")

# produce performance measures
performance_measures  <- confusionMatrix(data=pred, reference = as.factor(tsinfogain$o_bullied))
performance_measures

## J48 on InfoGain

model <- J48(as.factor(o_bullied)~., trinfogain)
predicted <- data.frame(predict(model, tsinfogain))
colnames(predicted)[1] <- "predicted"
cm <- table(predicted$predicted, tsinfogain$o_bullied)
cm

sens = 350/(350+111)
sens
spec = 177/(162+177)
spec

## C5.0 on InfoGain

library(C50)
C5.tree <- C5.0(as.factor(o_bullied)~., data = trinfogain)

# test 
pred <- predict(C5.tree, newdata = tsinfogain, type = "class")
performance_measures  <- confusionMatrix(data=pred, reference = as.factor(tsinfogain$o_bullied))
performance_measures

## rpart on InfoGain

library(rpart)
library(rpart.plot)
rpart.tree <- rpart(o_bullied~ ., data = tsinfogain, method = "class", parms = list(split = "information"))

# plot the tree
prp(rpart.tree, type = 1, extra = 1, under = TRUE, 
    split.font = 1, varlen = -10)

# test 
pred <- predict(rpart.tree, newdata = tsinfogain, type = "class")
performance_measures  <- confusionMatrix(data=pred,
                                         reference = as.factor(tsinfogain$o_bullied))
performance_measures

#Random Forest on InfoGain

library(randomForest)
rf60 <- randomForest(o_bullied~., data = trinfogain) 
pred <- predict(rf60,tsinfogain)
performance_measures  <- confusionMatrix(data=pred,
                                         reference = as.factor(tsinfogain$o_bullied))
performance_measures

#KNN on InfoGain

knnGrid <-  expand.grid(k = seq(1, 100, 2))
knnModel <- train(o_bullied~., data = trinfogain, method = "knn",
                  trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneGrid = knnGrid)

knnModel
plot(knnModel)

test_pred <- predict(knnModel, newdata = tsinfogain)
confusionMatrix(test_pred, as.factor(tsinfogain$o_bullied))

# PCA 

bal_train$o_bullied <- as.numeric(bal_train$o_bullied)

pc <- prcomp(bal_train[, -104], center = TRUE, scale = TRUE) # exclude class attribute
summary(pc)

# first map (project) original attributes to new attributes created by PCA

tr <- predict(pc, bal_train)
tr <- data.frame(tr, bal_train[104])
ts <- predict(pc, bal_test)
ts <- data.frame(ts, bal_test[104])

##Naive Bayes on PCA

# build a Naïve Bayes model from training dataset
tr1 = tr[c(1:25,104)]
data_nb <- naiveBayes(o_bullied ~ ., data=tr1)

# test on test dataset
pred <- predict(data_nb, newdata = ts[c(1:25)], type = "class")

# produce performance measures
performance_measures  <- confusionMatrix(data=pred, reference = as.factor(ts$o_bullied))
performance_measures

#Training and testing dataset based on feature selection (pca)
##J48 on PCA

model <- J48(as.factor(o_bullied)~., tr[c(1:25, 104)])
predicted <- data.frame(predict(model, ts[c(1:25)]))
colnames(predicted)[1] <- "predicted"
cm <- table(predicted$predicted, ts$o_bullied)
cm

#predicted$predicted[predicted$predicted == 1] <- 0
#predicted$predicted[predicted$predicted == 2] = 1
#performance_measures  <- confusionMatrix(data=predicted$predicted, reference = as.factor(ts$o_bullied))
#str(predicted$predicted)
#str(ts$o_bullied)
#levels(predicted$predicted)
#levels(as.factor(ts$o_bullied))

sens = 276/(276+185)
sens
spec = 191/(191+148)
spec

## C5.0 on PCA

library(C50)
C5.tree <- C5.0(as.factor(o_bullied)~., data = tr1)

# test 

pred <- predict(C5.tree, newdata = ts[c(1:25)], type = "class")
pred = as.numeric(pred)
pred = pred-1
pred = as.factor(pred)
performance_measures  <- confusionMatrix(data=pred, reference = as.factor(ts$o_bullied))
performance_measures

## rpart on PCA

library(rpart)
library(rpart.plot)
rpart.tree <- rpart(o_bullied~ ., data =tr1, 
                    method = "class", parms = list(split = "information"))
# plot the tree
prp(rpart.tree, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)

# test 
pred <- predict(rpart.tree, newdata = ts[c(1:25)], type = "class")
pred = as.numeric(pred)
pred = pred - 1
pred = as.factor(pred)
performance_measures  <- confusionMatrix(data=pred, reference = as.factor(ts$o_bullied))
performance_measures



library(tidyverse)
library(tidyr)

glass<- read.csv("C:/Users/USNHIT/Desktop/Machine Learning Projects/Glass Detector Project/glass.csv")

head(glass)

#Data Cleaning

columns_missing_values <- glass %>%
  summarise(across(everything(),~sum(is.na(.))))%>%
  pivot_longer(cols=everything(),names_to = "Column",values_to = "Missing_values")%>%
  filter(Missing_values>0)

proportion_missing <- columns_missing_values %>%
  mutate(proportion_missing = Missing_values/nrow(glass) *100)




#EDA

Sodium <- glass%>%
  group_by(Type)%>%
  summarise(max = max(glass$Na))

Aluminium
Sodium

plot(glass$Al,glass$Type)

#Whether data is balanced or not  
glass%>%
  group_by(Type)%>%
  count()


#test train split

set.seed(101)
size_glass = round(nrow(glass)*0.20)
testing = sample(nrow(glass),size_glass)

train_glass = glass[-testing,]
test_glass = glass[testing,]

library(nnet)

mult.log = multinom(Type~.,data=train_glass)

summary(mult.log)$coefficients

head(predict(mult.log, newdata = test_glass,"probs"))

pred.mn = predict(mult.log,newdata = test_glass, type = "class")

cm = confusionMatrix(factor(pred.mn),factor(test_glass$Type))

cm$byClass

#70% accuracy 

# LDA 

lda_glass <- lda(Type~.,data = train_glass)
lda_glass


plot(lda_glass)

lda_pred = predict(lda_glass,newdata =test_glass)

table(test_glass$Type, lda_pred$class)

confusionMatrix(as.factor(test_glass$Type),lda_pred$class)

#72% accuracy 



lda_cv_glass <- lda(Type~.,CV=TRUE,data= glass,subset = -testing)

lda_cv_glass

qda_glass <- qda(Type~., data=train_glass)

qda_pred <- predict(qda_glass, newdata = test_glass)
#some groups are too small


knn_glass_train = train_glass[,-ncol(train_glass)]
knn_glass_test = test_glass[,-ncol(test_glass)]
knn_glass_train_labels = train_glass[,ncol(train_glass)]
knn_glass_test_labels = test_glass[,ncol(test_glass)]

knn3 = knn(train = knn_glass_train, test = knn_glass_test, cl = knn_glass_train_labels, k=3)

confusionMatrix(knn3,as.factor(knn_glass_test_labels))

set.seed(1234)
k.grid=1:100
error=rep(0, length(k.grid))

for (i in seq_along(k.grid)) {
  pred = knn(train = scale(knn_glass_train), 
             test  = scale(knn_glass_test), 
             cl    = knn_glass_train_labels, 
             k     = k.grid[i])
  error[i] = mean(knn_glass_test_labels !=pred)
}

min(error)

plot(k.grid,error)

# reason for getting higher accuracy in logistic regression as compared to KNN can be because of high dimensionaltiy in the data set. 
# the second reason can be because model is following a linear relationship and logistic regression performs well in that. 
# Logistic regression can get benefit from feature selection and regularization techniques to handle irrelevent features. 
# As compared to the KNN, it follows a non parametric method.
# this can also be a reason that our data set size is small and logistic regression performs well in that.






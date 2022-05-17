set.seed(20162)
library(knitr)
library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(corrplot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(data.table)
library(corrplot)
library(plotly)
library(gbm)

train_data <- read.csv("data/pml-training.csv")
test_data <- read.csv("data/pml-testing.csv")
dim(train_data)

inTrain <- createDataPartition(train_data$classe, p=0.7, list = FALSE)

trainSet <- train_data[inTrain, ]
testSet <- train_data[-inTrain, ]

nzValues <- nearZeroVar(trainSet)

trainSet <- trainSet[ , -nzValues]
testSet  <- testSet [ , -nzValues]

naValue <- sapply(trainSet, function(x) mean(is.na(x))) > 0.95

trainSet <- trainSet[ , naValue == FALSE]
testSet  <- testSet [ , naValue == FALSE]

trainSet <- trainSet[ , -(1:5)]
testSet  <- testSet [ , -(1:5)]

classeIndex <- 54
names(trainSet)[classeIndex]

corrMatrix <- cor(trainSet[ , -classeIndex])
corrplot(corrMatrix, order = "FPC", method = "circle", type = "lower",
         tl.cex = 0.6, tl.col = rgb(0, 0, 0))

# Decision Tree
set.seed(20162)

fitModelDT <- rpart(classe~., data=trainSet, method="class")

predictDT <- predict(fitModelDT, newdata=testSet, type="class")

confMatrixDT <- confusionMatrix(table(predictDT, testSet$classe))
confMatrixDT

# GBM
set.seed(20162)

ctrlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fitModelGBM <- train(classe~., data=trainSet, method="gbm", trControl=ctrlGBM, verbose=FALSE)
fitModelGBM$finalModel

predictGBM <- predict(fitModelGBM, newdata=testSet)

confMatrixGBM <- confusionMatrix(table(predictGBM, testSet$classe))
confMatrixGBM

# Random Forest
set.seed(20162)

ctrlRF <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fitModelRF <- train(classe~., data=trainSet, method="rf", trControl=ctrlRF, verbose=FALSE)
fitModelRF$finalModel

predictRF <- predict(fitModelRF, newdata=testSet)
confMatrixRF <- confusionMatrix(table(predictRF, testSet$classe))
confMatrixRF

plot(fitModelRF)

# Quiz prediction
cat("Predictions: ", paste(predict(fitModelRF, test_data)))
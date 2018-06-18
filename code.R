# Loading packages
if(!require("VIM")) install.packages("VIM")
if(!require("mice")) install.packages("mice")
if(!require("caret")) install.packages("caret")
if(!require("RANN")) install.packages("RANN")
if(!require("Boruta")) install.packages("Boruta",dependencies = T)
if (!require("corrplot")) install.packages("corrplot")


# Loading dataset
data = read.csv("auto.csv",header=T, na.strings=c("?",""," ","NA"))

# Exploring
str(data)
summary(data)
nrow(data)
sum(is.na(data))
md.pattern(data)
aggr_plot <- aggr(data, col=c('LightBlue','LightYellow'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.3,cex.numbers=.9, gap=1, ylab=c("Histogram of missing data","Pattern"))
target=data$price
data$price = NULL
preProcValues <- preProcess(data, method = c("knnImpute","center","scale"))
data_processed <- predict(preProcValues, data) 
data_processed$price=target
data_processed=na.omit(data_processed)
sum(is.na(data_processed))


#Converting every categorical variable to numerical using dummy variables
dmy <- dummyVars(" ~ .", data = data_processed,fullRank = T)
data_transformed <- data.frame(predict(dmy, newdata = data_processed))
str(data_transformed)

#Feature Selection using boruta
set.seed(108)
boruta.train <- Boruta(price~., data = data_transformed, doTrace = 2)
print(boruta.train)
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)
getSelectedAttributes(boruta.train, withTentative = FALSE)
getSelectedAttributes(boruta.train, withTentative = TRUE)
fixed.boruta=TentativeRoughFix(boruta.train)
attStats(fixed.boruta)
finalvars=getSelectedAttributes(fixed.boruta,withTentative=F)

#New dataset
data_final=data_transformed[finalvars]
data_final$price=data_processed$price
str(data_final)

#Spliting training set into two parts based on outcome: 80% and 20%
set.seed(108)
index <- createDataPartition(data_final$price, p=0.80, list=FALSE)
trainSet <- data_final[ index,]
testSet <- data_final[-index,]
str(trainSet)
str(testSet)

price=trainSet$price
trainSet$price=NULL

#Model Training
# LM
model_lm<-train(trainSet, price,method='lm')
varImp(object=model_lm)
plot(varImp(object=model_lm),main="LM - Variable Importance")
# GBM
model_gbm<-train(trainSet, price,method='gbm')
plot(model_gbm)
varImp(object=model_gbm)
plot(varImp(object=model_gbm),main="GBM - Variable Importance")
# NNET
model_nnet<-train(trainSet,price,method='nnet')
plot(model_nnet)
varImp(object=model_nnet)
plot(varImp(object=model_nnet),main="NNET - Variable Importance")


#Predictions
predictions<-predict(model_gbm,newdata =testSet,type="raw")


summary(model_lm)
# Computatuion of MAE, RSE, RAE
MAE = mean(abs(predictions - testSet$price))
MAE
RSE = (mean((predictions-testSet$price)^2))/(mean((mean(testSet$price)-testSet$price)^2))
RSE
RAE = (mean(abs(predictions-testSet$price)))/(mean(abs(mean(testSet$price)-testSet$price)))
RAE

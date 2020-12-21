
#__________________________________________________________________________________________________________________________________________________________________________________________
#  LIBRARY DECLARATION
#-----------------------------------------------------------------------------------------------------------------------------------------------
library(caret)
library(doMC)
registerDoMC(20) # specify the number of core to be used
library(abind)
library(e1071)
library(readr)
library(readxl)
library(pROC)
source("InternalScalinglinsvmrfeFeatureRanking.R")
set.seed(2018)
start.time <- Sys.time()
#----------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________________________________________________________________________

myFile  <-"~/data/Auto_OML/Results/RFSVM/SVM_Training_Results_smote_bmcombat.csv"
myData  <- read.csv(myFile)

#__________________________________________________________________________________________________________________________________________________________________________________________
#  DATA DECLARATION
#-----------------------------------------------------------------------------------------------------------------------------------------------
#  Train Sample
#-----------------------------------------------------------------------------------------------------------------------------------------------
# !!!!!!!!!!!!! Load dataset (Matrix, row=patients col=features) >>> Example: ~/PhD/Machine_learning_Radiomics/TRAINING.csv
#!!!!!!!!!!!!!! First column should be Patients ID and last column should always be class for datamatrix !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#TrainSample_Temp           <- read_csv("../.././Dataset/Training_dataset/Training_Dataset.csv")
TrainSample_Temp           <- read.csv("~/data/Auto_OML/Dataset/Training_dataset/train_smote_bmcombat1.csv")
dimnames(TrainSample_Temp) <- list(rownames(TrainSample_Temp, do.NULL = FALSE, prefix = "row"), colnames(TrainSample_Temp, do.NULL = FALSE, prefix = "col"))

# !!!!!!! vector denoted by 1 and 0 for Classes >>> Example: c(0,0,0,1,1,1)
Class<-as.factor(unlist(TrainSample_Temp[,ncol(TrainSample_Temp)]))
levels(Class) <- list(no="0", yes="1")
TrainSample<-TrainSample_Temp[,-c(1,ncol(TrainSample_Temp))] # remove first and last column represneting Patients and class respectively
#----------------------------------------------------------------------------------------------------------------------------------------------
#  Test Sample
#-----------------------------------------------------------------------------------------------------------------------------------------------
# !!!!!!!!!!!!! Load dataset (Matrix, row=patients col=features) >>> Example: ~/PhD/Machine_learning_Radiomics/TRAINING.csv
#!!!!!!!!!!!!!! First column should be Patients ID and last column should always be class for datamatrix !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#TestSample_Temp           <- read_csv("../.././Dataset/Testing_dataset/Testing_Dataset.csv")
TestSample_Temp           <- read.csv("~/data/Auto_OML/Dataset/Testing_dataset/test_data_bmcombat.csv")
dimnames(TestSample_Temp) <- list(rownames(TestSample_Temp, do.NULL = FALSE, prefix = "row"), colnames(TestSample_Temp, do.NULL = FALSE, prefix = "col"))

# !!!!!!! vector denoted by 1 and 0 for Classes >>> Example: c(0,0,0,1,1,1)
Test_Class                <- as.factor(unlist(TestSample_Temp[,ncol(TestSample_Temp)]))
levels(Test_Class)        <- list(no="0", yes="1")
TestSample                <- TestSample_Temp[,-c(1:29,ncol(TestSample_Temp))] # remove first and last column represneting Patients and class respectively

#----------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________________________________________________________________________

#__________________________________________________________________________________________________________________________________________________________________________________________
#  PARAMETERS DECLARATION
#-----------------------------------------------------------------------------------------------------------------------------------------------

# Removing (i.e. a zero-variance predictor). For many models (excluding tree-based models), this may cause the model to crash or the fit to be unstable.
if (length(nearZeroVar(TrainSample)) > 0) {
  Near0Var        <- nearZeroVar(TrainSample)
  TrainSample     <- TrainSample[, -Near0Var] 
  TestSample      <- TestSample[, -Near0Var] 
}

fitControl        <- trainControl(method = "none", classProbs=T) # no we use the previously tuened parameters 


# Previously tune parameters 
SVMTrainCost      <- myData$SVM_train_cost
featureRankedList <- myData$Ranked_feature_List
Accuracydetails   <- matrix(, nrow = ncol(TrainSample), ncol = 8)

#----------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________________________________________________________________________

#__________________________________________________________________________________________________________________________________________________________________________________________
#-----------------------------------------------------------------------------------------------------------------------------------------------

for(nfeatures in 1:length(featureRankedList)){
  
  SVMModel       <- train(x=TrainSample[, featureRankedList[1:nfeatures],drop = FALSE], y=(Class), method = "svmLinear2",Type="Classification",trControl = fitControl,tuneGrid =data.frame(cost=SVMTrainCost[nfeatures]),preProc = c("center", "scale")) # Caret
  predictions_prob <- predict(SVMModel,TestSample[, featureRankedList[1:nfeatures],drop=FALSE],type = "prob")
  predictionsAcc <- predict(SVMModel,TestSample[, featureRankedList[1:nfeatures],drop=FALSE])
  
  cat("############################## Feature Index ###############################","\n")
  cat(featureRankedList[1:nfeatures],"\n")
  cat("############################## Test results ################################","\n")
  Test_roc_obj <- roc(Test_Class,predictions_prob$no)
  Test_AUC<-(Test_roc_obj$auc)
  cat("AUC",Test_AUC,"\n")
  Test_Results<-confusionMatrix(predictionsAcc,Test_Class)
  cat("Accuracy",Test_Results$overall[1],"\n")
  cat("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!","\n")
  MCC_Test=((Test_Results$table[1]*Test_Results$table[4])-(Test_Results$table[2]*Test_Results$table[3]))/(sqrt(Test_Results$table[1]+Test_Results$table[2])*sqrt(Test_Results$table[3]+Test_Results$table[4])*sqrt(Test_Results$table[2]+Test_Results$table[4])*sqrt(Test_Results$table[1]+Test_Results$table[3]))
  Accuracydetails[nfeatures, 1] <- Test_Results$overall[1]
  Accuracydetails[nfeatures, 2] <- Test_Results$byClass[1]
  Accuracydetails[nfeatures, 3] <- Test_Results$byClass[2]
  Accuracydetails[nfeatures, 4] <- Test_Results$byClass[3]
  Accuracydetails[nfeatures, 5] <- Test_Results$byClass[4]
  Accuracydetails[nfeatures, 6] <- Test_AUC
  Accuracydetails[nfeatures, 7] <- MCC_Test
  Accuracydetails[nfeatures, 8] <- Test_Results$byClass[11]

}
#----------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________________________________________________________________________

#__________________________________________________________________________________________________________________________________________________________________________________________
#  WRITING TESTING RESULTS
#-----------------------------------------------------------------------------------------------------------------------------------------------
Result<-data.frame(featureRankedList,Accuracydetails)
write.table(Result, file = "~/data/Auto_OML/Results/RFSVM/SVM_Testing_Results_smote_bmcombat.csv",row.names=FALSE, na="",col.names=c("Ranked_feature_List","Test_Accuracy","Test_Sensitivity","Test_Specificity","Test_Positive_Predictive_Value", "Test_Negative_Predictive_Value","Test_AUC","Test_MCC","Test_Balanced_Accuracy"), sep=",")

#----------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________________________________________________________________________ 
end.time <- Sys.time()
Algo_time<-end.time - start.time
print("ALGO TIME:")
print(Algo_time)  
#******************************************************************************************************************************************************************************************
#********************************************************************************************* END ****************************************************************************************
#******************************************************************************************************************************************************************************************

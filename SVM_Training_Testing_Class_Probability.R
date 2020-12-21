#******************************************************************************************************************************************************************************************
#********************************************************************************************* TESTING SUPPORT VECTOR MACHINE ALGORITHM ***************************************************
#******************************************************************************************************************************************************************************************
#@author  Taman Upadhaya <tamanupadhaya@gmail.com>
#@version 2.0, 09/11/2018
#@since   R version (3.4.4).
#__________________________________________________________________________________________________________________________________________________________________________________________
#  LIBRARY DECLARATION
#-----------------------------------------------------------------------------------------------------------------------------------------------
library(caret)
library(doMC)
registerDoMC(10) # specify the number of core to be used
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
myFile  <-"~/data/Auto_OML/Results/Support_Vector_Machine/SVM_Training_Results_bcombat.csv"
myData  <- read.csv(myFile)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nfeatures         <- which.max(myData$CV_Accuracy) # state the final number of feature based on ranking (default combination of feature that gives max CV accuracy)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#__________________________________________________________________________________________________________________________________________________________________________________________
#  DATA DECLARATION
#-----------------------------------------------------------------------------------------------------------------------------------------------
#  Train Sample
#-----------------------------------------------------------------------------------------------------------------------------------------------
# !!!!!!!!!!!!! Load dataset (Matrix, row=patients col=features) >>> Example: ~/PhD/Machine_learning_Radiomics/TRAINING.csv
#!!!!!!!!!!!!!! First column should be Patients ID and last column should always be class for datamatrix !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
TrainSample_Temp           <- read.csv("~/data/Auto_OML/Dataset/Training_dataset/train_data_bcombat.csv")
dimnames(TrainSample_Temp) <- list(rownames(TrainSample_Temp, do.NULL = FALSE, prefix = "row"), colnames(TrainSample_Temp, do.NULL = FALSE, prefix = "col"))

# !!!!!!! vector denoted by 1 and 0 for Classes >>> Example: c(0,0,0,1,1,1)
Class<-as.factor(unlist(TrainSample_Temp[,ncol(TrainSample_Temp)]))
levels(Class) <- list(no="0", yes="1")
TrainSample   <-TrainSample_Temp[,-c(1:29,ncol(TrainSample_Temp))] # remove first and last column represneting Patients and class respectively
#----------------------------------------------------------------------------------------------------------------------------------------------
#  Test Sample
#-----------------------------------------------------------------------------------------------------------------------------------------------
# !!!!!!!!!!!!! Load dataset (Matrix, row=patients col=features) >>> Example: ~/PhD/Machine_learning_Radiomics/TRAINING.csv
#!!!!!!!!!!!!!! First column should be Patients ID and last column should always be class for datamatrix !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
TestSample_Temp           <- read.csv("~/data/Auto_OML/Dataset/Testing_dataset/test_data_bcombat.csv")
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

fitControl        <- trainControl(method = "none",classProbs=T) # no we use the previously tuened parameters 

# Previously tune parameters 
SVMTrainCost      <- myData$SVM_train_cost
featureRankedList <- myData$Ranked_feature_List
#----------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________________________________________________________________________

#__________________________________________________________________________________________________________________________________________________________________________________________
#-----------------------------------------------------------------------------------------------------------------------------------------------


  
SVMModel                 <- train(x=TrainSample[, featureRankedList[1:nfeatures],drop = FALSE], y=(Class), method = "svmLinear2",Type="Classification",trControl = fitControl,tuneGrid =data.frame(cost=SVMTrainCost[nfeatures]),preProc = c("center", "scale")) # Caret

#Write probability for train sample without smote
predictions_prob_Train   <- predict(SVMModel ,TrainSample[, featureRankedList[1:nfeatures],drop = FALSE],type = "prob")
predictions_class_Train  <- predict(SVMModel ,TrainSample[, featureRankedList[1:nfeatures],drop = FALSE])

# Write probability for test sample
predictions_Test_prob     <- predict(SVMModel ,TestSample[, featureRankedList[1:nfeatures],drop = FALSE],type = "prob")
predictions_Test_class    <- predict(SVMModel ,TestSample[, featureRankedList[1:nfeatures],drop = FALSE])
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________________________________________________________________________
#__________________________________________________________________________________________________________________________________________________________________________________________
#  WRITING RESULTS
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
write.table((cbind(predictions_prob_Train$no,predictions_prob_Train$yes,predictions_class_Train)), file = "~/data/Auto_OML/Results/Support_Vector_Machine/SVM_Train_bcombat_class_probability.csv", row.names = FALSE, na="",col.names=c("Probability_Class_1","Probability_Class_2","Class"),sep = ",")
write.table((cbind(predictions_Test_prob$no,predictions_Test_prob$yes,predictions_Test_class)), file = "~/data/Auto_OML/Results/Support_Vector_Machine/SVM_Test_bcombat_class_probability.csv", row.names = FALSE, na="",col.names=c("Probability_Class_1","Probability_Class_2","Class"),sep = ",")  

#----------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________________________________________________________________________ 
end.time <- Sys.time()
Algo_time<-end.time - start.time
print("ALGO TIME:")
print(Algo_time)
#******************************************************************************************************************************************************************************************
#********************************************************************************************* END ****************************************************************************************
#******************************************************************************************************************************************************************************************

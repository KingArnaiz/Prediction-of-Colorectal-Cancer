
#__________________________________________________________________________________________________________________________________________________________________________________________
#  INSTALLING THE PACKAGES IF NOT INSTALLED
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#packages <- c("caret","doMC","readr","pROC","readxl")
#if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
#  install.packages(setdiff(packages, rownames(installed.packages())))  
#  install.packages("caret", dependencies = c("Depends", "Suggests"))
#}
#----------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________________________________________________________________________

#__________________________________________________________________________________________________________________________________________________________________________________________
#  LIBRARY DECLARATION
#-----------------------------------------------------------------------------------------------------------------------------------------------
library(readr)
library(readxl)
library(caret)
library(doMC)
library(pROC)
library(e1071)
registerDoMC(20) #Specify number of core to be used
set.seed(2018)
source("InternalScalinglinsvmrfeFeatureRanking.R")
start.time <- Sys.time()
#----------------------------------------------------------------------------------------------------------------------------------------------
#____________________________________________________________________________________________________________________________________________________________________________________
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CV         <- 1 # choose option for cross-validation (default LOOCV)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#____________________________________________________________________________________________________________________________________________________________________________________
#  DATA DECLARATION
#-----------------------------------------------------------------------------------------------------------------------------------------------
# !!!!!!!!!!!!! Load dataset (Matrix, row=patients col=features) >>> Example: ~/PhD/Machine_learning_Radiomics/TRAINING.csv
#!!!!!!!!!!!!!! First column should be Patients ID and last column should always be class for datamatrix !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#TrainSample_Temp           <- read.csv("~/data/Auto_OML/Dataset/Training_dataset/cervixTraining_Dataset70percentClinicMRIwithoutComBatWithoutPatientsWithNear0values.csv", sep = "\t", header = TRUE)
TrainSample_Temp           <- read.csv("~/Desktop/result/new result/train_smote_orig1.csv")
dimnames(TrainSample_Temp) <- list(rownames(TrainSample_Temp, do.NULL = FALSE, prefix = "row"), colnames(TrainSample_Temp, do.NULL = FALSE, prefix = "col"))

# !!!!!!! vector denoted by 1 and 0 for Classes >>> Example: c(0,0,0,1,1,1)
Class<-as.factor(unlist(TrainSample_Temp[,ncol(TrainSample_Temp)]))
levels(Class) <- list(no="0", yes="1")
TrainSample<-TrainSample_Temp[,-c(1,ncol(TrainSample_Temp))] # remove first and last column represneting Patients and class respectively
#----------------------------------------------------------------------------------------------------------------------------------------------
#____________________________________________________________________________________________________________________________________________________________________________________

#____________________________________________________________________________________________________________________________________________________________________________________
#  PARAMETERS DECLARATION
#-----------------------------------------------------------------------------------------------------------------------------------------------
# removing non zero variance features
if (length(nearZeroVar(TrainSample)) > 0) {
  TrainSample <- TrainSample[, -nearZeroVar(TrainSample)] 
}

#VARIOUS CROSS-VALIDATION
if(CV==1){
  fitControl <-trainControl(method="LOOCV", allowParallel = T, savePredictions = TRUE, classProbs=T)
}else if(CV==2){
  fitControl <- trainControl(method="cv", number=10, allowParallel = T, savePredictions = TRUE, classProbs=T)
}else if(CV==3){
  fitControl <- trainControl(method="cv", number=5, allowParallel = T, savePredictions = TRUE, classProbs=T)
}else if(CV==4){
  fitControl <- trainControl(method="boot",allowParallel = T, savePredictions = TRUE, classProbs=T)
}else if(CV==5){
  fitControl <- trainControl(method="repeatedcv", number=5, repeats=5, allowParallel = T, savePredictions = TRUE, classProbs=T)
}else{
  fitControl <- trainControl(method="repeatedcv", number=10, repeats=5, allowParallel = T, savePredictions = TRUE, classProbs=T)}

# Linear SVM C parametrs in a grid
SVMgrid      <- expand.grid(cost=seq(0.001,7,0.1)) # grid search

#----------------------------------------------------------------------------------------------------------------------------------------------
#____________________________________________________________________________________________________________________________________________________________________________________

#____________________________________________________________________________________________________________________________________________________________________________________
#  INITIAL MODEL BUILDING 
#-----------------------------------------------------------------------------------------------------------------------------------------------

#Tunning using Caret package !!!! method svmLinear = Kernallab package and svmLinear2 = e1071 package
SVMfit   <-  train(x=TrainSample, y = Class, method = "svmLinear2",Type="Classification",preProc = c("center", "scale"),trControl = fitControl,tuneGrid = SVMgrid)
#----------------------------------------------------------------------------------------------------------------------------------------------
#____________________________________________________________________________________________________________________________________________________________________________________

#____________________________________________________________________________________________________________________________________________________________________________________
#  FEATURE RANKING BASED ON THE INITIAL MODEL 
#-----------------------------------------------------------------------------------------------------------------------------------------------
SVMCost           <- SVMfit$bestTune$cost
featureRankedList <- InternalScalinglinsvmrfeFeatureRanking(TrainSample,Class,SVMCost)
#print(featureRankedList) # list of ranked features
#----------------------------------------------------------------------------------------------------------------------------------------------
#____________________________________________________________________________________________________________________________________________________________________________________

#_______________________________________________________________________________________________________________________________________________
#  MODEL BUILDING BASED ON RANKED FEATURES 
#-----------------------------------------------------------------------------------------------------------------------------------------------
# tunning for all the variable together according to added number
##################################### tunning the parameter in parallel ###############################
iters           <- ncol(TrainSample)
SVMTrainCost    <- vector('list',length=iters)
Accuracydetails <- matrix(, nrow = ncol(TrainSample), ncol = 17)

for (icount in 1:iters ) {
  nfeatures            <- icount
  SVMModel             <- train(x=TrainSample[, featureRankedList[1:nfeatures],drop = FALSE], y = Class, method = "svmLinear2",Type="Classification",preProc = c("center", "scale"),trControl = fitControl,tuneGrid = SVMgrid)
  SVMTrainCost[icount] <- SVMModel$bestTune$cost
  
  cat("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!","\n")
  cat("############################# Feature Index ######################################","\n")
  cat(featureRankedList[1:nfeatures],"\n")
  cat("######################### Development results ####################################","\n")
  fit_prob<-predict(SVMModel,type = "prob")
  Dev_roc_obj <- roc(Class,fit_prob$no)
  Dev_AUC<-(Dev_roc_obj$auc)
  cat("AUC",Dev_AUC,"\n")
  Dev_Results <- confusionMatrix(predict(SVMModel),Class)
  cat("Accuracy",Dev_Results$overall[1],"\n") #Accuracy
  MCC_Dev=((Dev_Results$table[1]*Dev_Results$table[4])-(Dev_Results$table[2]*Dev_Results$table[3]))/(sqrt(Dev_Results$table[1]+Dev_Results$table[2])*sqrt(Dev_Results$table[3]+Dev_Results$table[4])*sqrt(Dev_Results$table[2]+Dev_Results$table[4])*sqrt(Dev_Results$table[1]+Dev_Results$table[3]))
  Accuracydetails[nfeatures, 1] <- Dev_Results$overall[1]
  Accuracydetails[nfeatures, 2] <- Dev_Results$byClass[1]
  Accuracydetails[nfeatures, 3] <- Dev_Results$byClass[2]
  Accuracydetails[nfeatures, 4] <- Dev_Results$byClass[3]
  Accuracydetails[nfeatures, 5] <- Dev_Results$byClass[4]
  Accuracydetails[nfeatures, 6] <- Dev_AUC
  Accuracydetails[nfeatures, 7] <- MCC_Dev
  Accuracydetails[nfeatures, 8] <- Dev_Results$byClass[11]
  Accuracydetails[nfeatures, 9] <- SVMTrainCost[[icount]]
  
  Pred_Index=which(SVMModel$pred$cost==SVMModel$bestTune$cost)
  CV_roc_obj <- roc(SVMModel$pred$obs[Pred_Index],SVMModel$pred$no[Pred_Index]) #roc(groundtruth,prediction)
  CV_AUC<-(CV_roc_obj$auc)
  cat("###################### Internal validation results ###############################","\n")
  cat("AUC:",CV_AUC,"\n")
  CV_Results <- confusionMatrix(SVMModel$pred$pred[Pred_Index],SVMModel$pred$obs[Pred_Index])
  cat("Accuracy",CV_Results$overall[1],"\n") #Accuracy
  cat("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!","\n")
  CV_Accuracy                       <- CV_Results$overall[1]
  CV_Sensitivity                    <- CV_Results$byClass[1]
  CV_Specificity                    <- CV_Results$byClass[2]
  CV_Pos_Pred_Value                 <- CV_Results$byClass[3]
  CV_Neg_Pred_Value                 <- CV_Results$byClass[4]
  CV_Balanced_Accuracy              <- CV_Results$byClass[11]
  MCC_CV=((CV_Results$table[1]*CV_Results$table[4])-(CV_Results$table[2]*CV_Results$table[3]))/(sqrt(CV_Results$table[1]+CV_Results$table[2])*sqrt(CV_Results$table[3]+CV_Results$table[4])*sqrt(CV_Results$table[2]+CV_Results$table[4])*sqrt(CV_Results$table[1]+CV_Results$table[3]))
  Accuracydetails[nfeatures, 10]    <- CV_Accuracy
  Accuracydetails[nfeatures, 11]    <- CV_Sensitivity
  Accuracydetails[nfeatures, 12]    <- CV_Specificity
  Accuracydetails[nfeatures, 13]    <- CV_Pos_Pred_Value
  Accuracydetails[nfeatures, 14]    <- CV_Neg_Pred_Value
  Accuracydetails[nfeatures, 15]    <- CV_AUC
  Accuracydetails[nfeatures, 16]    <- MCC_CV
  Accuracydetails[nfeatures, 17]    <- CV_Balanced_Accuracy
}
#----------------------------------------------------------------------------------------------------------------------------------------------
#____________________________________________________________________________________________________________________________________________________________________________________      

#____________________________________________________________________________________________________________________________________________________________________________________
#  WRITING TRANING RESULTS
#-----------------------------------------------------------------------------------------------------------------------------------------------
Result <- data.frame(featureRankedList,Accuracydetails)
write.table(Result, file = "~/data/Auto_OML/Results/RFSVM/SVM_Training_Results_smote_bmcombat.csv",row.names=FALSE, na="",col.names=c("Ranked_feature_List","Development_Accuracy","Development_Sensitivity","Development_Specificity","Development_Positive_Predictive_Value", "Development_Negative_Predictive_Value","Development_AUC","Development_MCC","Development_Balanced_Accuracy","SVM_train_cost","CV_Accuracy","CV_Sensitivity","CV_Specificity","CV_Positive_Predictive_Value", "CV_Negative_Predictive_Value","CV_AUC","CV_MCC","CV_Balanced_Accuracy"), sep=",")#----------------------------------------------------------------------------------------------------------------------------------------------
#____________________________________________________________________________________________________________________________________________________________________________________ 
end.time <- Sys.time()
Algo_time<-end.time - start.time
print("ALGO TIME:")
print(Algo_time)      
#******************************************************************************************************************************************************************************************
#********************************************************************************************* END ****************************************************************************************
#******************************************************************************************************************************************************************************************
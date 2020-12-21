InternalScalinglinsvmrfeFeatureRanking = function(TrainSample,ytrain,C){
  n = ncol(TrainSample)
  
  survivingFeaturesIndexes = seq(1:n)
  featureRankedList = vector(length=n)
  rankedFeatureIndex = n
  tuned_C <- numeric()
  
  while(length(survivingFeaturesIndexes)>0){
    #train the support vector machine
    svmModel = svm(x=TrainSample[, survivingFeaturesIndexes], y=factor(ytrain), cost = C, cachesize=500,  scale=T, type="C-classification", kernel="linear" )
    #compute the weight vector
    w = t(svmModel$coefs)%*%svmModel$SV
    
    #compute ranking criteria
    rankingCriteria = w * w
    
    #rank the features
    ranking = sort(rankingCriteria, index.return = TRUE)$ix #sorting in decending order
    
    #update feature ranked list
    featureRankedList[rankedFeatureIndex] = survivingFeaturesIndexes[ranking[1]]
    rankedFeatureIndex = rankedFeatureIndex - 1
    
    #eliminate the feature with smallest ranking criterion
    (survivingFeaturesIndexes = survivingFeaturesIndexes[-ranking[1]]) 
  }
  
  return (featureRankedList) # return list of Ranked features
}

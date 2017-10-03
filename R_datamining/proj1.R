#change working directory
setwd("C:\\Users\\xiyang\\Desktop\\cap5771\\R_datamining")

#install and import all the required packages
install.packages("RWeka", dependencies = TRUE)
install.packages("e1071", dependencies = TRUE)
install.packages("caret", dependencies = TRUE)
install.packages("dplyr", dependencies = TRUE)
install.packages("class", dependencies = TRUE)
install.packages("data", dependencies = TRUE)
install.packages("klaR", dependencies = TRUE)
library(data.table)
library(RWeka)
library(e1071)
library(caret)
library(klaR)
require(class)
require(dplyr)

#read csv file into the memory
input_data <- function(csv_file){
  proj1_data = fread(csv_file, select=c(3,4,5,6))
  proj1_data$continent <- as.factor(proj1_data$continent) 
  df = as.data.frame(proj1_data)
  return(df)
}

#generate the dataframe and seperate it into training and testing dataset by stratified sampling
create_dataframe <- function(df, mySeed){
  m <- split(x=df, f=df$continent)
  Overall_life_expectancy_at_birth <- c()
  Male_life_expectancy_at_birth <- c()
  Female_life_expectancy_at_birth <- c()    
  continent <- c()
  label <- c()
  dt_training <- data.frame(Overall_life_expectancy_at_birth, Male_life_expectancy_at_birth, Female_life_expectancy_at_birth, continent, label)
  dt_testing <- data.frame(Overall_life_expectancy_at_birth, Male_life_expectancy_at_birth, Female_life_expectancy_at_birth, continent, label)
  set.seed(seed=mySeed)
  for(each in m){
    #shuffle
    each <- each[sample(nrow(each)),]
    #print(each)
    #create train and test set
    co <- as.integer(nrow(each)*0.8)
    dtrain <- each[1:co, ]
    dtest <- each[(co+1):nrow(each),]
    dt_training <- bind_rows(dt_training, dtrain)
    dt_testing <- bind_rows(dt_testing, dtest)
  }
  
  dt_training$continent <- as.factor(dt_training$continent)
  dt_testing$continent <- as.factor(dt_testing$continent)
  
  return(list(dt_training, dt_testing))
}

#scale method (not needed)
#myNormalize <- function(x){
#  return ((x-min(x))/(max(x)-min(x)))
#}

#train control parameter
trctrl <- trainControl(method = "cv", number=3)

#knn
knn_model <- function(data_train){
  fit <- train(data_train[,c(1,2,3)], data_train[,4], method="knn", trControl=trctrl, tuneGrid=expand.grid(.k=1:25), 
               tuneLength = 10, preProcess=c("center", "scale"), metric="Accuracy")
  return(fit)
}

#c4.5
c45_model <- function(data_train){
  fit <- train(data_train[,c(1,2,3)], data_train[,4], method="J48",tuneLength = 10,trControl=trctrl, metric="Accuracy")
  return(fit)
}

#ripper
RIP_model <- function(data_train){
  fit <- train(data_train[,c(1,2,3)], data_train[,4], method="JRip", tuneLength = 10, trControl=trctrl,metric="Accuracy")
  return(fit)
}

#svm
svm_model <- function(data_train){
  fit <- train(data_train[,c(1,2,3)], data_train[,4], method="svmRadial",tuneLength = 10,trControl=trctrl, metric="Accuracy")
}

myPrediction <- function(fit, data_test){
  #predict
  p <- predict(fit, data_test[, c(1,2,3)])
  #get accuracy
  res <- postResample(p, data_test[, 4]) ##list(accurary, kappa)
  #get percision and recall
  tf <- as.data.frame(table(p, data_test[, 4]))
  len <- nrow(tf)
  tmc <- c("Africa", "Europe", "Asia", "North America", "Oceania", "South America")
  rc <- 0.0
  pc <- 0.0
  for(mc in tmc){
    tp <- 0
    fp <- 0
    tn <- 0
    fn <- 0
    for(i in 1:len){
      li <- tf[i,]
      if(li$p == mc && li$Var2==mc){
        tp <- tp + li$Freq
      }
      
      if(li$p == mc && li$ Var2 != mc){
        fp <- fp + li$Freq
      }
      
      if(li$p != mc && li$ Var2 == mc){
        tn <- tn + li$Freq
      }
      
      if(li$p != mc && li$ Var2 != mc){
        fn <- fn + li$Freq
      }
    }
    
    rc <- rc + (tp/(tp + fn))
    pc <- pc + (tp/(tp + fp))
  }
  
  return(c(res, rc, pc))
}

get_fscore <- function(rc, pc){
  return((2*rc*pc/(rc+pc)))
}

main <- function(){
  df <- input_data("proj1_data.csv")
  results <- list(kknn=c(), kc45=c(), krip=c(), ksvn=c())
  recalls <- list(c(), c(), c(), c())
  percisions <- list(c(), c(), c(), c())  
  #run exp 5 times
  for (i in 1:5){
    train_and_test <- create_dataframe(df, i*1000)
    train_set <- train_and_test[[1]]
    test_set <- train_and_test[[2]]
    #all_models <- c(knn_model(train_set), c45_model(train_set), RIP_model(train_set), svm_model(train_set))
    for(j in 1:4){
      model <- knn_model(train_set)
      
      if(j==2){
        model <- c45_model(train_set)
      }
      
      if(j==3){
        model <- RIP_model(train_set)
      }
      
      if(j==4){
        model <- svm_model(train_set)
      }
      
      res_set <- myPrediction(model, test_set)
      res <- res_set[1]
      rc <- res_set[2]
      pc <- res_set[3]
      results[[j]] <- c(results[[j]], res)
      recalls[[j]] <- c(recalls[[j]], rc)
      percisions[[j]] <- c(percisions[[j]], pc)
    }
  }
  
  #calc results avg and sd
  avg <- c()
  msd <- c()
  for (i in 1:4){
    temp <- c()
    temp1 <- c()
    temp2 <- c()
    for (each in results[i]){
      temp <- append(temp, each)
    }
    
    for (each in recalls[i]){
      temp1 <- append(temp1, each)
    }
    
    for (each in percisions[i]){
      temp2 <- append(temp2, each)
    }
    
    avg <- append(avg, mean(temp))
    msd <- append(msd, sd(temp))
    
    avg_recall <- append(avg_recall, mean(temp1))
    avg_percision <- append(avg_percision, mean(temp2))
  }
  
  #present the results
  for(i in 1:4){
    umethod <- "KNN"
    
    if(i==2){
      umethod <- "C4.5"
    }
    
    if(i==3){
      umethod <- "RIPPER"
    }
    
    if(i==4){
      umethod <- "SVM"
    }
    
    print(sprintf("method name: %s; averaged accuracy: %.2f; accuracy standard deviation: %.3f", umethod, avg[i], msd[i]))
    print(sprintf("averaged recall: %.3f; averaged percision: %.3f; averaged f-score: %.3f", avg_recall[i], avg_percision[i], get_fscore(avg_recall[i], avg_percision[i])))
  }
}

#execute main function
main()

#detach all the packages
detach("package:RWeka", unload = TRUE)
detach("package:dplyr", unload = TRUE)
detach("package:e1071", unload = TRUE)
detach("package:caret", unload = TRUE)
detach("package:klaR", unload = TRUE)
detach("package:class", unload = TRUE)
detach("package:data", unload = TRUE)
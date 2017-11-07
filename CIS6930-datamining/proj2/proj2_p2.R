if(!require(broom)) install.packages("broom", dependencies = TRUE, quiet=TRUE)

help("install.packages")

library(broom)

#read in data
d2 <- read.csv(file="dataset2.csv", header=TRUE, sep=",")
d2 <- scale(d2)
head(d2)

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

d2n <- scale(d2)
print("preview scaled dataset")
head(d2n)

#Elbow Method for finding the optimal number of clusters
#set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
#k.max <- 10:25
#k.max <- c(10, 20, 30, 40, 50, 60, 70, 80,  90, 100)
#data <- d2
#wss <- sapply(k.max, function(k){kmeans(data, k, nstart=100, iter.max = 1000)$tot.withinss})
#wss
#plot(k.max, wss,type="b", pch = 19, frame = FALSE, xlab="Number of clusters K",ylab="Total within-clusters sum of squares")
bss <- c()
tss <- c()
for (i in 1:5){
  k <- kmeans(d2n, 16, iter.max = 1000,  algorithm="MacQueen")
  print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
  print("Evaluation parameters for k=16: ")
  print(glance(k))
  bss <- c(k$betweenss, bss)
  tss <- c(k$totss, tss)
}
print("**********************************************************************************************")
print("**********************************************************************************************")
print(sprintf("average BSS/TSS ratio of k=16: %.3f", mean(bss)/mean(tss)))
print("**********************************************************************************************")
print("**********************************************************************************************")
bss <- c()
tss <- c()
for (i in 1:5){
  k <- kmeans(d2n, 15, iter.max = 1000,  algorithm="MacQueen")
  print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
  print("Evaluation parameters for k=15: ")
  print(glance(k))
  bss <- c(k$betweenss, bss)
  tss <- c(k$totss, tss)
}
print("**********************************************************************************************")
print("**********************************************************************************************")
print(sprintf("average BSS/TSS ratio of k=15: %.3f", mean(bss)/mean(tss)))
print("**********************************************************************************************")
print("**********************************************************************************************")
#k <- kmeans(d2n, 13, iter.max = 1000,  algorithm="MacQueen")
#print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#print("Evaluation parametersfor k=13: ")
#glance(k)
#print(sprintf("BSS/TSS ratio of k=13: %.3f", k$betweenss/k$totss))

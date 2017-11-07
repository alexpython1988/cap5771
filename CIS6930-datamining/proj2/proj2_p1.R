#setwd("C:\\Users\\xiyang\\Desktop\\cap5771\\CIS6930-datamining\\proj2")

#if(!require(scatterplot3d)) install.packages("scatterplot3d", dependencies = TRUE)
if(!require(cluster)) install.packages("cluster", dependencies = TRUE,  quiet=TRUE)
if(!require(dbscan)) install.packages("dbscan", dependencies = TRUE,  quiet=TRUE)

library(cluster)
library(dbscan)


d1 <- read.csv(file="dataset1.csv", header=TRUE, sep=",")
print("preview dataset")
head(d1)

myNormalize <- function(x){
  if(typeof(x)=="double"){
    return ((x-min(x))/(max(x)-min(x)))
  }else{
    return(x)
  }
}
d1n <- as.data.frame(lapply(d1, myNormalize))
print("preview scaled dataset")
head(d1n)

#plot data
#shapes = c(11,12,13,14,15,16,17,18) 
#shapes <- shapes[as.numeric(d1$cluster)]
#colors <- c("red", "yellow", "blue", "black", "green", "orange", "cyan", "mediumpurple2")
#colors <- colors[as.numeric(d1$cluster)]
#p <- scatterplot3d(x= d1$x, y=d1$y, z=d1$z, pch = shapes, color=colors)

print("**********************************************************************************************")
print("hirearchy clustering...")
clusters <- hclust(dist(d1n[, 1:3], method = "euclidean"), method = 'centroid')

#plot(clusters)
clusterCut <- cutree(clusters, 8)
print("result table: ")
table(clusterCut, d1n$cluster)
acc <- sum(clusterCut==d1n$cluster)/nrow(d1n)
print(sprintf("accuracy: %.3f", acc))
print("**********************************************************************************************")
cat()
print("**********************************************************************************************")
print("K means clustering...")
l <- c()
for(i in 1:5){
  k <- kmeans(d1n[,1:3], 8, nstart=25, iter.max = 100,  algorithm="MacQueen")
  print(sprintf("The %sth round result table and accuracy: ", i))
  print(table(k$cluster, d1n$cluster))
  res <- sum(k$cluster==d1n$cluster)/nrow(d1n)
  print(res)
  l <- append(res, l)
}
print(sprintf("The average K means clustering accuracy: %.3f",mean(l)))
print("**********************************************************************************************")
cat()
print("**********************************************************************************************")
#knn decided 
#kNNdistplot(d1n[,1:3], k=8)
#abline(h=0.1078, col="red")
print("density based clustering:")
db <- dbscan(d1n[,1:3], 0.1075, 8)
print("result table: ")
table(db$cluster, d1n$cluster)
acc <- sum(db$cluster==d1n$cluster)/nrow(d1n)
print(sprintf("accuracy: %.3f", acc))
print("**********************************************************************************************")
cat()
print("**********************************************************************************************")
print("Graph based clustering: ")
cl <- sNNclust(d1n[,1:3], k = 8, eps = 0.05, minPts = 8)
print("result table: ")
table(cl$cluster, d1n$cluster)
acc <- sum(cl$cluster==d1n$cluster)/nrow(d1n)
print(sprintf("accuracy: %.3f", acc))
print("**********************************************************************************************")

#detach("package:scatterplot3d", unload = TRUE)
detach("package:cluster", unload = TRUE)
detach("package:dbscan", unload = TRUE)


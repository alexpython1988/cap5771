#http://www.sthda.com/english/articles/30-advanced-clustering/105-dbscan-density-based-clustering-essentials/

setwd("C:\\Users\\xiyang\\Desktop\\cap5771\\CIS6930-datamining\\proj2")

#read in data
d1 <- read.csv(file="dataset1.csv", header=TRUE, sep=",")
head(d1)

myNormalize <- function(x){
  if(typeof(x)=="double"){
    return ((x-min(x))/(max(x)-min(x)))
  }else{
    return(x)
  }
}

d1 <- as.data.frame(lapply(d1, myNormalize))

d1

plot(clusters)


set.seed(123456789)
d1 <- d1[sample(nrow(d1)),]
d1
d1[,1:3]

library(scatterplot3d)
library(cluster)
#library(caret)
library(dbscan)

shapes = c(11,12,13,14,15,16,17,18) 
shapes <- shapes[as.numeric(d1$cluster)]
colors <- c("red", "yellow", "blue", "black", "green", "orange", "cyan", "mediumpurple2")
colors <- colors[as.numeric(d1$cluster)]
p <- scatterplot3d(x= d1$x, y=d1$y, z=d1$z, pch = shapes, color=colors)

## hirearchy
clusters <- hclust(dist(d1[, 1:3], method = "euclidean"), method = 'centroid')
#plot(clusters)
clusterCut <- cutree(clusters, 8)
table(clusterCut, d1$cluster)
clusterCut

t0 = table(clusterCut, d1$label)
t0


c2 <- agnes(d1[, 1:3], method = "ward")
c2$ac
pltree(c2, cex = 0.6, hang = -1, main = "Dendrogram of agnes") 


clusterCut1 <- cutree(c2, 8)
t = table(clusterCut1, d1$cluster)
sum(diag(t0))/sum(t0)

#k means
l <- c()
for(i in 1:5){
  k <- kmeans(d1[,1:3], 8, nstart=25, iter.max = 100,  algorithm="Lloyd")
  print(table(k$cluster, d1$cluster))
  res <- sum(k$cluster==d1$cluster)/nrow(d1)
  print(res)
  l <- append(res, l)
}
mean(l)

#density
kNNdistplot(d1n[,1:3], k=8)
abline(h=0.10, col="red")
db <- dbscan(d1n[,1:3], 0.106, 8)
db
table(db$cluster, d1n$cluster)
sum(db$cluster==d1n$cluster)/nrow(d1n)

hullplot(d1[,1:3], db$label)
t1 = 
t1

sum(diag(t1))/sum(t1)

kNNdistplot(d1[,1:3], k=8)
abline(h=1.48, col="red")
db <- dbscan(d1[,1:3], 1.48, 8)
db
sum(db$cluster==d1$cluster)/nrow(d1)
table(db$cluster, d1$cluster)

#graphic 
#kernel k means
library(kernlab)
kk <- kkmeans(as.matrix(d1[,1:3]),  centers=5)
kk

d1
cl <- sNNclust(d1[,1:3], k = 8, eps = 0.05, minPts = 8)
cl
table(cl$cluster, d1$cluster)
sum(cl$cluster==d1$cluster)

d2 <- read.csv(file="dataset2.csv", header=TRUE, sep=",")
#d2$x <- as.double(d2$x)
#d2 <- as.data.frame(lapply(d2, myNormalize))
d2 <- scale(d2)
head(d2)

kNNdistplot(d2)
abline(h=0.003, col="red")
db <- dbscan(d2, 0.0025)
db$cluster


#Elbow Method for finding the optimal number of clusters
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- c(10, 20, 30, 40, 50, 60, 70, 80,  90, 100)
data <- d2
wss <- sapply(k.max, function(k){kmeans(data, k, nstart=100, iter.max = 1000)$tot.withinss})
wss
plot(k.max, wss,type="b", pch = 19, frame = FALSE, xlab="Number of clusters K",ylab="Total within-clusters sum of squares")

k <- kmeans(d2, 16, iter.max = 1000,  algorithm="Lloyd")
k







##
pkgs <- c("factoextra",  "NbClust")
install.packages(pkgs)
library(factoextra)
library(NbClust)
fviz_nbclust(d2, kmeans, method = "wss") 
geom_vline(xintercept = 4, linetype = 2) 
labs(subtitle = "Elbow method")


set.seed(123)
fviz_nbclust(d2, kmeans, nstart = 25,  method = "gap_stat", nboot = 50)
labs(subtitle = "Gap statistic method")


nb <- NbClust(d2, distance = "euclidean", min.nc = 2,max.nc = 10, method = "kmeans")



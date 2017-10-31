#http://www.sthda.com/english/articles/30-advanced-clustering/105-dbscan-density-based-clustering-essentials/

setwd("C:\\Users\\xiyang\\Desktop\\cap5771\\CIS6930-datamining\\proj2")

#read in data
d1 <- read.csv(file="dataset1.csv", header=TRUE, sep=",")
d1
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
plot(clusters)
clusterCut <- cutree(clusters, 8)
t0 = table(clusterCut, d1$cluster)
table(clusterCut, d1$cluster)

c2 <- agnes(d1[, 1:3], method = "ward")
c2$ac
pltree(c2, cex = 0.6, hang = -1, main = "Dendrogram of agnes") 


clusterCut1 <- cutree(c2, 8)
t = table(clusterCut1, d1$cluster)
sum(diag(t0))/sum(t0)

#k means
for(i in 1:5){
  k <- kmeans(d1[,1:3], 8, nstart=25, iter.max = 100,  algorithm="Lloyd")
  t1<- table(k$cluster, d1$cluster)
  print(sum(diag(t1))/sum(t1))
}

#density
kNNdistplot(d1[,1:3], k=8)
abline(h=1.5, col="red")
db <- dbscan(d1[,1:3], 1.5, 8)
db$cluster
hullplot(d1[,1:3], db$cluster)
table(db$cluster, d1$cluster)
sum(db$cluster==d1$cluster)
#sum(diag(t1))/sum(t1)


#graphic 
#kernel k means
library(kernlab)
kk <- kkmeans(as.matrix(d1[,1:3]),  centers=8)
kk



#cl <- sNNclust(d1[,1:3], k = 100, eps = 0.5, minPts = 8)
#table(cl$cluster, d1$cluster)
#sum(cl$cluster==d1$cluster)


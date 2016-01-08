library(dplyr)
library(ggplot2)
library(tidyr)
library(grid)
library(gridExtra)

k <- 1:20

# Random Forest classifier values

rf_precision <- c(1: 0.222, 0.333, 1.0, 1.0, 1.0, 0.667, 0.5, 1.0, 1.0, 
                  0.0, 0.0, 0.0, 0.0, 0.333, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0)

rf_recall <- c(0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.2, 0.4, 0.2, 0.0, 
               0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.4, 0.0)

rf_df <- data.frame(k, rf_precision, rf_recall)
long_rf_df <- gather(rf_df, "Method", "n", 2:3)

p1 <- ggplot(aes(x=k, y=n, colour=Method), data=long_rf_df) +
  geom_line(aes(group=Method), size=1) +
  geom_point() + 
  theme(legend.position='bottom', legend.title=element_blank()) +
  ggtitle("Random Forest") +
  ylab("Score")

# Support Vector Machine classifier values

svm_precision <- c(1.0, rep(0.0, 19))
svm_recall <- c(0.2, rep(0.0,19))

svm_df <- data.frame(k, svm_precision, svm_recall)
long_svm_df <- gather(svm_df, "Method", "n", 2:3)

p2 <- ggplot(aes(x=k, y=n, colour=Method), data=long_svm_df) +
  geom_line(aes(group=Method), size=1) +
  geom_point() + 
  theme(legend.position='bottom', legend.title=element_blank()) +
  ggtitle("Support Vector Machine") +
  ylab("Score")


# K Nearest Neighbors classifier values

knn_precision <- c(0.25, 0.5, 0.333, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 
                   0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.333, 0.333)


knn_recall <- c(rep(0.2, 5), 0.6, rep(0.2,7), rep(0.4, 3), rep(0.2, 4))

knn_df <- data.frame(k, knn_precision, knn_recall)
long_knn_df <- gather(knn_df, "Method", "n", 2:3)

p3 <- ggplot(aes(x=k, y=n, colour=Method), data=long_knn_df) +
  geom_line(aes(group=Method), size=1) +
  geom_point() + 
  theme(legend.position='bottom', legend.title=element_blank()) +
  ggtitle("K Nearest Neighbors") +
  ylab("Score")

grid.arrange(p1, p2, p3, ncol=3, nrow=1, 
             top=textGrob("Precision & Recall vs Number of Features", 
                          gp=gpar(fontsize=20, font=3)))




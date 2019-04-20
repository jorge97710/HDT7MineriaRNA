# install.packages("nnet")
# install.packages("RWeka")
# install.packages("neuralnet")

library(caret)
library(nnet)
library(RWeka)
library(neural)
library(dummy)
library(neuralnet)

perros5 <- read.csv("train.csv")
porcentaje<-0.7
datos<-perros5
set.seed(123)
prueba <- perros5[,3:15]
prueba[14] <- perros5[24]
datos<- prueba
corte <- sample(nrow(datos),nrow(datos)*porcentaje)
train<-datos[corte,]
test<-datos[-corte,]

#-------------------------------------------------
# Red Neuronal con nnet
#-------------------------------------------------

modelo.nn2 <- nnet(AdoptionSpeed~.,data = datos,subset = corte, size=2, rang=0.1,
                   decay=5e-4, maxit=400) 
prediccion2 <- as.data.frame(predict(modelo.nn2, newdata = test[,1:13]))
columnaMasAlta<-apply(prediccion2, 1, function(x) colnames(prediccion2)[which.max(x)])
test$prediccion2<-columnaMasAlta #Se le añade al grupo de prueba el valor de la predicción

cfm<-confusionMatrix(test$prediccion2,test$AdoptionSpeed)
cfm

#-------------------------------------------------
# Red Neuronal con RWeka
#-------------------------------------------------
NB <- make_Weka_classifier("weka/classifiers/functions/MultilayerPerceptron")
NB 
WOW(NB)
nnodos='4'

modelo.bp<-NB(Species~., data=datos,subset = corte, control=Weka_control(H=nnodos, N=1000, G=TRUE), options=NULL)
test$prediccionWeka<-predict(modelo.bp, newdata = test[,1:4])
cfmWeka<-confusionMatrix(test$prediccionWeka,test$Species)
cfmWeka

#-------------------------------------------------
# Red Neuronal con caret
#-------------------------------------------------

modeloCaret <- train(AdoptionSpeed~., data=train, method="nnet", trace=F)
test$prediccionCaret<-predict(modeloCaret, newdata = test[,1:13])
cfmCaret<-confusionMatrix(as.factor(test$prediccionCaret),as.factor(test$AdoptionSpeed))
cfmCaret


#-------------------------------------------------
# Red Neuronal con NeuralNet
#-------------------------------------------------
train$y<-as.numeric(train$AdoptionSpeed)
test$y<-as.numeric(test$AdoptionSpeed)

modelo.nn <- neuralnet(y~Sepal.Length+Petal.Length, train[,c(1,3,6)], hidden = 2, rep = 3)
plot(modelo.nn, newdata=test) #Sale un gr?fico por cada repetici?n, en este caso saldr?n 3 gr?ficos
test$predNeuralNet<-round(predict(modelo.nn,newdata = test[,1:4]),0)
cfmNeuralNet<-confusionMatrix(as.factor(test$predNeuralNet),as.factor(test$y))
cfmNeuralNet

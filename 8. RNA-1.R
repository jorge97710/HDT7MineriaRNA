# install.packages("nnet")
# install.packages("RWeka")
# install.packages("neuralnet")

library(caret)
library(nnet)
library(RWeka)
library(neural)
library(dummy)
library(neuralnet)


porcentaje<-0.7
datos<-iris
set.seed(123)


corte <- sample(nrow(datos),nrow(datos)*porcentaje)
train<-datos[corte,]
test<-datos[-corte,]

#-------------------------------------------------
# Red Neuronal con nnet
#-------------------------------------------------

modelo.nn2 <- nnet(Species~.,data = datos,subset = corte, size=2, rang=0.1,
                   decay=5e-4, maxit=200) 
prediccion2 <- as.data.frame(predict(modelo.nn2, newdata = test[,1:4]))
columnaMasAlta<-apply(prediccion2, 1, function(x) colnames(prediccion2)[which.max(x)])
test$prediccion2<-columnaMasAlta #Se le añade al grupo de prueba el valor de la predicción

cfm<-confusionMatrix(as.factor(test$prediccion2),test$Species)
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

modeloCaret <- train(Species~., data=train, method="nnet", trace=F)
test$prediccionCaret<-predict(modeloCaret, newdata = test[,1:4])
cfmCaret<-confusionMatrix(test$prediccionCaret,test$Species)
cfmCaret


#-------------------------------------------------
# Red Neuronal con NeuralNet
#-------------------------------------------------
train$y<-as.numeric(train$Species)
test$y<-as.numeric(test$Species)

modelo.nn <- neuralnet(y~Sepal.Length+Petal.Length, train[,c(1,3,6)], hidden = 2, rep = 3)
plot(modelo.nn, newdata=test) #Sale un gr?fico por cada repetici?n, en este caso saldr?n 3 gr?ficos
test$predNeuralNet<-round(predict(modelo.nn,newdata = test[,1:4]),0)
cfmNeuralNet<-confusionMatrix(as.factor(test$predNeuralNet),as.factor(test$y))
cfmNeuralNet

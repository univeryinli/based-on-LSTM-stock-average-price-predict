library(forecast)#auto.arima() arma阶数确定方法

x<-read.csv(file.choose())
x1<-x[(x[,2]>20130000)&(x[,2]<20170000),]
x2<-x[(x[,2]>20170000),]
y1<-x1[,12]
y2<-x2[,12]
y<-c(y1,y2)
arma_y1<-auto.arima(y1)
arma_y1


##################################  AR
j<-1
yhat<-c()
p<-3
for(j in 1:length(y2))
{
  yhat[j]<-arma_y1$coef[1:p]%*%y[(length(y1)+j-p-9):(length(y1)+j-10)]
}

#结果输出
mse<-mean(((y2-yhat)^2))
mae<-mean((abs(y2-yhat)))
rmse<-sqrt(mse)
error<-c(mse,mae,rmse)

yhat_1<-c(rep(0,length(x[,2])-length(yhat)),yhat)
error_1<-c(error,rep(0,length(x[,2])-length(error)))
write.csv(cbind(x,yhat_1,error_1),file = "data20.csv")
#########################################

##################################   MA
j<-1
yhat<-c()
q<-1
e<-arma_y1$residuals
for(j in 1:length(y2))
{
  yhat[j]<-arma_y1$coef[1:q] %*% e[(length(e)-q-9):(length(e)-10)]
  e<-c(e,yhat[j]-y2[j])
}
yhat<-abs(yhat)

#结果输出
mse<-mean(((y2-yhat)^2))
mae<-mean((abs(y2-yhat)))
rmse<-sqrt(mse)
error<-c(mse,mae,rmse)

yhat_1<-c(rep(0,length(x[,2])-length(yhat)),yhat)
error_1<-c(error,rep(0,length(x[,2])-length(error)))
write.csv(cbind(x,yhat_1,error_1),file = "data23.csv")
#########################################

##################################  ARMA
j<-1
yhat<-c()
n<-6
p<-1#AR的项数
e<-arma_y1$residuals
for(j in 1:length(y2))
{
  yhat[j]<-arma_y1$coef[1:p]%*%y[(length(y1)+j-p-9):(length(y1)+j-10)]
  +arma_y1$coef[(p+1):n]%*%e[(length(e)-n+p-9):(length(e)-10)]
  e<-c(e,yhat[j]-y2[j])
}

#结果输出
mse<-mean(((y2-yhat)^2))
mae<-mean((abs(y2-yhat)))
rmse<-sqrt(mse)
error<-c(mse,mae,rmse)

yhat_1<-c(rep(0,length(x[,2])-length(yhat)),yhat)
error_1<-c(error,rep(0,length(x[,2])-length(error)))
write.csv(cbind(x,yhat_1,error_1),file = "data1.csv")
#########################################





#画图对比预测和真实值
plot(y2,type="l",col="blue")
plot(yhat,type="l",col="red")

##读取同一目录下的所有文件
path1 <- "F:/统计相关资料/研究生资料/深度学习/to 深度学习小组/timeseries_1to50" ##文件目录
fileNames1 <- dir(path1) 
filePath1 <- sapply(fileNames1, function(x){ 
  paste(path1,x,sep='/')})   

path2 <- "F:/统计相关资料/研究生资料/深度学习/to 深度学习小组/data_output_csv" 
fileNames2 <- dir(path2)  
filePath2 <- sapply(fileNames2, function(x){ 
  paste(path2,x,sep='/')}) 

length(filePath1) 
error1<-c()
error2<-c()
for(k in 1:length(filePath1) ){
  
  data1=read.csv(filePath1[k], header=T)
  y<-data1[,13]
  yhats0<-data1[,14]
  yhats<-yhats0[yhats0>0]
  y1<-y[yhats0>0]
  
  data2<-read.csv(filePath2[k], header=T)
  ydl<-data2[,2]/(10^7)
  y2<-y[291495:(291494+length(ydl))]
  
  mse1<-mean(((y1-yhats)^2))
  mae1<-mean((abs(y1-yhats)))
  rmse1<-sqrt(mse1)
  qmse1<-mean((sqrt(y1)-sqrt(yhats))^2)
  qmae1<-mean((abs(sqrt(y1)-sqrt(yhats))))
  error01<-c(mse1,mae1,rmse1,qmse1,qmae1)
  error1<-cbind(error1,error01)
  
  mse2<-mean(((y2-ydl)^2))
  mae2<-mean((abs(y2-ydl)))
  rmse2<-sqrt(mse2)
  qmse2<-mean((sqrt(y2)-sqrt(ydl))^2)
  qmae2<-mean((abs(sqrt(y2)-sqrt(ydl))))
  error02<-c(mse2,mae2,rmse2,qmse2,qmae2)
  error2<-cbind(error2,error02)
}
write.csv(error1,file='timesout_error.csv')
write.csv(error2,file='dl_error.csv')




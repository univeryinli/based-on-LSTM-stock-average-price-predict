import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import interpolate
import numpy as np
import csv as csv

alpha=128
belta=5
x_data=[]
y_data=[]
input_dim=2
x_cell=np.zeros((alpha,input_dim))
train_size=10000
test_size=1000
batch_size=100
#WSZ: smoothing window size needs, which must be odd number
wsz5=11
wsz10=21
wsz101=11
predict_days=11
smooth150=[[7.59,5.58,5.29,5.26,5.25,4.69,4.22,4.04,3.91,3.91],[2.70,3.11,3.27,3.28,3.22,1.38,1.51,1.61,1.45,1.46]]
smooth50=[[6.83,4.96,4.33,4.10,3.92,3.77,3.66,3.58,3.50,3.45],[3.31,1.73,1.56,1.61,1.65,1.60,1.48,1.64,1.60,1.58]]
smooth10_30=[[6.37,4.42,3.85,3.70,3.57,3.51,3.43,3.40,3.35,3.31],[3.19,1.71,1.85,1.52,1.63,1.46,1.48,1.54,1.57,1.50]]
smooth5_20=[[6.24,4.43,3.80,3.66,3.53,3.48,3.44,3.38,3.43,3.29],[3.27,1.58,1.54,1.48,1.61,1.64,1.43,1.63,1.50,1.56]]


def smooth(a,WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

drop0_7=sio.loadmat('./history.losses_drop0.7.mat')['loss']
drop0_8=sio.loadmat('./history.losses_drop0.8.mat')['loss']
conv1=sio.loadmat('./history.losses_conv1.mat')['loss']
conv2=sio.loadmat('./history.losses_conv2.mat')['loss']
layer3=sio.loadmat('./history.losses_layer3.mat')['loss']
layer4=sio.loadmat('./history.losses_layer4.mat')['loss']
layer5=sio.loadmat('./history.losses_layer5.mat')['loss']
dense16=sio.loadmat('./history.losses_dense16.mat')['loss']
dense64=sio.loadmat('./history.losses_dense64.mat')['loss']
dense64_32_16=sio.loadmat('./history.losses_dense64-32-16.mat')['loss']

value_drop0_7=sio.loadmat('./valos.losses_drop0.7.mat')['loss']
value_drop0_8=sio.loadmat('./valos.losses_drop0.8.mat')['loss']
value_conv1=sio.loadmat('./valos.losses_conv1.mat')['loss']
value_conv2=sio.loadmat('./valos.losses_conv2.mat')['loss']
value_layer3=sio.loadmat('./valos.losses_layer3.mat')['loss']
value_layer4=sio.loadmat('./valos.losses_layer4.mat')['loss']
value_layer5=sio.loadmat('./valos.losses_layer5.mat')['loss']
value_dense16=sio.loadmat('./valos.losses_dense16.mat')['loss']
value_dense64=sio.loadmat('./valos.losses_dense64.mat')['loss']
value_dense64_32_16=sio.loadmat('./valos.losses_dense64-32-16.mat')['loss']

#print(value_loss)
drop0_7=drop0_7[0]
drop0_8=drop0_8[0]
conv1=conv1[0]
conv2=conv2[0]
layer3=layer3[0]
layer4=layer4[0]
layer5=layer5[0]
dense16=dense16[0]
dense64=dense64[0]
dense64_32_16=dense64_32_16[0]

value_drop0_7=value_drop0_7[0]
value_drop0_8=value_drop0_8[0]
value_conv1=value_conv1[0]
value_conv2=value_conv2[0]
value_layer3=value_layer3[0]
value_layer4=value_layer4[0]
value_layer5=value_layer5[0]
value_dense16=value_dense16[0]
value_dense64=value_dense64[0]
value_dense64_32_16=value_dense64_32_16[0]


x1=[i for i in range(len(drop0_7))]
x1=np.array(x1)
#print(x1)
#x2=[i for i in range(len(data2))]
x1_new=(np.linspace(0,len(drop0_7)-1,500))

print(x1.shape,x1_new.shape)

# "nearest","zero"为阶梯插值  # slinear 线性插值  # "quadratic","cubic" 为2阶、3阶B样条曲线插值  
f = interpolate.interp1d(x1, drop0_7, kind='quadratic')
drop0_7=f(x1_new)
f = interpolate.interp1d(x1, value_drop0_7, kind='quadratic')
value_drop0_7=f(x1_new)
f = interpolate.interp1d(x1, drop0_8, kind='quadratic')
drop0_8=f(x1_new)
f = interpolate.interp1d(x1, value_drop0_8, kind='quadratic')
value_drop0_8=f(x1_new)
f = interpolate.interp1d(x1, conv1, kind='quadratic')
conv1=f(x1_new)
f = interpolate.interp1d(x1, value_conv1, kind='quadratic')
value_conv1=f(x1_new)
f = interpolate.interp1d(x1, conv2, kind='quadratic')
conv2=f(x1_new)
f = interpolate.interp1d(x1, value_conv2, kind='quadratic')
value_conv2=f(x1_new)
f = interpolate.interp1d(x1, layer3, kind='quadratic')
layer3=f(x1_new)
f = interpolate.interp1d(x1, value_layer3, kind='quadratic')
value_layer3=f(x1_new)
f = interpolate.interp1d(x1, layer4, kind='quadratic')
layer4=f(x1_new)
f = interpolate.interp1d(x1, value_layer4, kind='quadratic')
value_layer4=f(x1_new)
f = interpolate.interp1d(x1, layer5, kind='quadratic')
layer5=f(x1_new)
f = interpolate.interp1d(x1, value_layer5, kind='quadratic')
value_layer5=f(x1_new)
f = interpolate.interp1d(x1, dense16, kind='quadratic')
dense16=f(x1_new)
f = interpolate.interp1d(x1, value_dense16, kind='quadratic')
value_dense16=f(x1_new)
f = interpolate.interp1d(x1, dense64, kind='quadratic')
dense64=f(x1_new)
f = interpolate.interp1d(x1, value_dense64, kind='quadratic')
value_dense64=f(x1_new)
f = interpolate.interp1d(x1, dense64_32_16, kind='quadratic')
dense64_32_16=f(x1_new)
f = interpolate.interp1d(x1, value_dense64_32_16, kind='quadratic')
value_dense64_32_16=f(x1_new)
f = interpolate.interp1d(x1, smooth50[0], kind='quadratic')
smooth50_t=f(x1_new)
f = interpolate.interp1d(x1, smooth50[1], kind='quadratic')
smooth50_test=f(x1_new)
f = interpolate.interp1d(x1, smooth150[0], kind='quadratic')
smooth150_t=f(x1_new)
f = interpolate.interp1d(x1, smooth150[1], kind='quadratic')
smooth150_test=f(x1_new)
f = interpolate.interp1d(x1, smooth5_20[0], kind='quadratic')
smooth5_20_t=f(x1_new)
f = interpolate.interp1d(x1, smooth5_20[1], kind='quadratic')
smooth5_20_test=f(x1_new)
f = interpolate.interp1d(x1, smooth10_30[0], kind='quadratic')
smooth10_30_t=f(x1_new)
f = interpolate.interp1d(x1, smooth10_30[1], kind='quadratic')
smooth10_30_test=f(x1_new)
# ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)  


plt.figure(1)

plt.plot(x1_new,drop0_7,label='drop0_7')
plt.plot(x1_new,drop0_8,label='drop0_8')
plt.plot(x1_new,conv1,label='conv1')
plt.plot(x1_new,conv2,label='conv2')
plt.plot(x1_new,layer3,label='layer3')
plt.plot(x1_new,layer4,label='layer4')
plt.plot(x1_new,layer5,label='layer5')
plt.plot(x1_new,dense16,label='dense16')
plt.plot(x1_new,dense64,label='dense64')
plt.plot(x1_new,dense64_32_16,label='dense64_32_16')
plt.plot(x1_new,smooth5_20_t,label='smooth5_20')
plt.plot(x1_new,smooth10_30_t,label='smooth10_30')
plt.legend()

#plt.plot(x1_new,drop0_7,label='drop0
# _7',x1_new,drop0_8,x1_new,conv1,x1_new,conv2,x1_new,layer3,
# x1_new,layer4,x1_new,layer5,x1_new,dense16,x1_new,dense64,x1_new,dense64_32_16)
plt.figure(2)
plt.plot(x1_new,value_drop0_7,label='value_drop0_7')
plt.plot(x1_new,value_drop0_8,label='value_drop0_8')
plt.plot(x1_new,value_conv1,label='value_conv1')
plt.plot(x1_new,value_conv2,label='value_conv2')
plt.plot(x1_new,value_layer3,label='value_layer3')
plt.plot(x1_new,value_layer4,label='value_layer4')
plt.plot(x1_new,value_layer5,label='value_layer5')
plt.plot(x1_new,value_dense16,label='dense16')
plt.plot(x1_new,value_dense64,label='dense64')
plt.plot(x1_new,value_dense64_32_16,label='value_dense64_32_16')
plt.plot(x1_new,smooth5_20_test,label='smooth5_20_test')
plt.plot(x1_new,smooth10_30_test,label='smooth10_30_test')
#plt.plot(x1_new,value_drop0_7,x:1_new,value_drop0_8,
# x1_new,value_conv1,x1_new,value_conv2,x1_new,value_layer3,
# x1_new,value_layer4,x1_new,value_layer5,x1_new,value_dense16,
# x1_new,value_dense64,x1_new,value_dense64_32_16)
#plt.figure(2)
#plt.plot(x2,data3)
plt.legend()
plt.show()


from keras.models import Sequential
import scipy.io as sio
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.models import *
from keras.layers import Dense, Activation,LSTM, Conv1D
from keras.layers import Conv2D, MaxPooling2D, Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, LocallyConnected2D,MaxPool1D
from keras.models import load_model
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint,Callback
import matplotlib.pyplot as plt
import csv
import numpy as np

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
wsz=5
predict_days=11

'''
def smooth(a,wsz):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
    # WSZ: smoothing window size needs, which must be odd number,
    '''out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))
    '''
    temp=a
    for i in range(len(a)):
        if i<wsz:
            mean=float(sum(temp[0:i+1]))/len(temp[0:i+1])
        else:
            mean=float(sum(temp[i+1-wsz:i+1]))/len(temp[i+1-wsz:i+1])
    temp[i]=mean
    return temp
'''

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

with open("./5_XSHG.600837.h5.csv",encoding='utf-8') as f:
#csv_file=csv.reader(open('./5_XSHG.600837.h5.csv'),'r')
    csv_file=csv.reader(f)
    csv_list=list(csv_file)

    close=[float(line[5]) for line in csv_list[1:]]
    sample_size=len(close)-alpha
    
    for i in range(alpha,sample_size,belta):
        temp=close[i-alpha:i+predict_days]
#        x_cell[:,0]=np.array(temp)
        rt_temp=[1]+[(temp[close_index]/temp[close_index-1]) for close_index in range(1,len(temp))]
        rt=(np.log(np.array(rt_temp[0:alpha])))**2
        rt_pre=rt_temp[alpha:alpha+predict_days]
        rt_mean=(sum(rt_pre)/(len(rt_pre)))*1e6
        smooth_rt=smooth(rt,wsz)
        x_cell[:,0]=rt
        x_cell[:,1]=smooth_rt
        x_data.append(x_cell)
        y_data.append(rt_mean)
        y_data=smooth(y_data,wsz)         
x_data=np.array(x_data)
y_data=np.array(y_data)

x_train,y_train=x_data[0:train_size,:,:] , y_data[0:train_size]
x_test,y_test=x_data[train_size:(train_size+test_size),:,:] , y_data[train_size:(train_size+test_size)]

sio.savemat('y_test.mat', {'y_test': y_test})
print('train_size:','x-',x_train.shape,'y-',y_train.shape)
print('test_size','x-',x_test.shape,'y-',y_test.shape)
print('start lstm')

#LSTM
class LossHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
      self.losses = []

  def on_epoch_end(self, epoch, logs={}):
      self.losses.append(logs.get('loss'))

class ValueLoss(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.vallos = []

    def on_epoch_end(self, epoch, logs={}):
        self.vallos.append(logs.get('val_loss'))

class myLSTM(object):
    def __init__(self, time_length=alpha , channels=input_dim):
        self.time_length = alpha
        self.channels = channels

    def get_net(self):
        inputs = Input((self.time_length, self.channels))
        Conv1 = Conv1D(16, 1, input_shape=(alpha, 2), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        print('Conv1.shape', Conv1.shape)
        LSTM1 = LSTM(32,return_sequences=True)(Conv1)
        LSTM1_1 = LSTM(32, return_sequences=True)(LSTM1)
        LSTM1_2 = LSTM(32, return_sequences=True)(LSTM1_1)
        LSTM1_3 = LSTM(32, return_sequences=True)(LSTM1_2)
        LSTM1 = LSTM(64, return_sequences=True)(LSTM1_3)

        Conv2 = Conv1D(16, 16, input_shape=(alpha, 2), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print('Conv2.shape', Conv2.shape)
        LSTM2 = LSTM(32, return_sequences=True)(Conv2)
        LSTM2_1 = LSTM(32, return_sequences=True)(LSTM2)
        LSTM2_2 = LSTM(32, return_sequences=True)(LSTM2_1)
        LSTM2_3 = LSTM(32, return_sequences=True)(LSTM2_2)
        LSTM2 = LSTM(64, return_sequences=True)(LSTM2_3)

        Conv3 = Conv1D(16, 32, input_shape=(alpha, 2), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print('Conv3.shape', Conv3.shape)
        LSTM3 = LSTM(32, return_sequences=True)(Conv3)
        LSTM3_1 = LSTM(32, return_sequences=True)(LSTM3)
        LSTM3_2 = LSTM(32, return_sequences=True)(LSTM3_1)
        LSTM3_3 = LSTM(32, return_sequences=True)(LSTM3_2)
        LSTM3 = LSTM(64, return_sequences=True)(LSTM3_3)

        merge1=merge([LSTM1,LSTM2,LSTM3],mode='concat',concat_axis=-1)
        print('merge1.shape',merge1.shape)
        LSTM4=LSTM(alpha,return_sequences=True)(merge1)
        LSTM5=LSTM(64,return_sequences=False)(LSTM4)
        dense4=Dense(32,activation='relu')(LSTM5)
        output = Dense(1,activation='linear')(dense4)
        model = Model(input=inputs, output=[output])
        model.compile(optimizer=Adam(lr=1e-4), loss='mae')

        return model

    def train(self):
        model = self.get_net()
        print("got net")
        history = LossHistory()
        VALOS = ValueLoss()
        model_checkpoint = ModelCheckpoint('LSTMnet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(x_train, y_train, shuffle=True, epochs=10, batch_size=batch_size, callbacks=[history, VALOS, model_checkpoint], validation_data=(x_test, y_test))
        print('predict test data')
        Vt_result = model.predict(x_test,batch_size=batch_size)
        print(Vt_result)
        sio.savemat('test_dpm_conv_test1.mat', {'result': Vt_result})
        '''
        plt.figure(1)
        ax1 = plt.subplot(111)
        plt.figure(1)
        plt.plot(history.losses)
        plt.sca(ax1)
        plt.plot(VALOS.vallos)
        plt.show()
        '''
        sio.savemat('history.losses_conv_test1.mat',{'loss':history.losses})
        sio.savemat('valos.losses_conv_test1.mat',{'loss':VALOS.vallos})
if __name__ == '__main__':
    myunet = myLSTM()
    myunet.train()

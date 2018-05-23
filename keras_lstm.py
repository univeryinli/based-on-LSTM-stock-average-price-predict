import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from scipy.misc import imsave
from keras.models import *
from keras.layers import Dense, Activation,LSTM, Conv1D
import scipy.io as sio
from keras.layers import Conv2D, MaxPooling2D, Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, LocallyConnected2D,MaxPool1D
from keras.models import load_model
from keras.optimizers import *
from  keras.callbacks import ModelCheckpoint,Callback
import matplotlib.pyplot as plt
import h5py


load_fn1 = r'C:\Users\Charlie\Desktop\input.mat'
data1 = sio.loadmat(load_fn1)
L1 = data1['patch']

load_fn2 = r'C:\Users\Charlie\Desktop\output.mat'
data2 = sio.loadmat(load_fn2)
L2 = data2['value']
X_train=np.zeros((15000,128,2))
X_test=np.zeros((3000,128,2))

X_train[:,:,:] ,Y_train= L1[0:15000,:,:],L2[0:15000]
X_test[:,:,:] ,Y_test= L1[15001:18001,:,:],L2[15001:18001]
print(X_train.shape)
print(Y_train.shape)

# Y_train = keras.utils.to_categorical(Y_train, num_classes)
# Y_test = keras.utils.to_categorical(Y_test, num_classes)

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

#  class myLSTM

class myLSTM(object):
    def __init__(self, time_length=128, channels=2):
        self.time_length= time_length
        self.channels = channels

    def get_net(self):
        inputs = Input((self.time_length, self.channels))
        Conv1 = Conv1D(16, 1, input_shape=(128, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            inputs)
        print('Conv1.shape', Conv1.shape)
        LSTM1 = LSTM(32,return_sequences=True)(Conv1)
        LSTM1_1 = LSTM(32, return_sequences=True)(LSTM1)
        LSTM1_2 = LSTM(32, return_sequences=True)(LSTM1_1)
        LSTM1_3 = LSTM(32, return_sequences=True)(LSTM1_2)
        LSTM1 = LSTM(64, return_sequences=True)(LSTM1_3)



        Conv2 = Conv1D(16, 16, input_shape=(128, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            inputs)
        print('Conv2.shape', Conv2.shape)
        LSTM2 = LSTM(32, return_sequences=True)(Conv2)
        LSTM2_1 = LSTM(32, return_sequences=True)(LSTM2)
        LSTM2_2 = LSTM(32, return_sequences=True)(LSTM2_1)
        LSTM2_3 = LSTM(32, return_sequences=True)(LSTM2_2)
        LSTM2 = LSTM(64, return_sequences=True)(LSTM2_3)


        Conv3 = Conv1D(16, 32, input_shape=(128, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            inputs)
        print('Conv3.shape', Conv3.shape)
        LSTM3 = LSTM(32, return_sequences=True)(Conv3)
        LSTM3_1 = LSTM(32, return_sequences=True)(LSTM3)
        LSTM3_2 = LSTM(32, return_sequences=True)(LSTM3_1)
        LSTM3_3 = LSTM(32, return_sequences=True)(LSTM3_2)
        LSTM3 = LSTM(64, return_sequences=True)(LSTM3_3)


        merge1=merge([LSTM1,LSTM2,LSTM3],mode='concat',concat_axis=-1)
        print('merge1.shape',merge1.shape)
        LSTM4=LSTM(128,return_sequences=True)(merge1)
        LSTM5=LSTM(64,return_sequences=False)(LSTM4)
        #LSTM1 = LSTM(16,input_shape=(512,2),return_sequences=True)(inputs)
        #LSTM2 = LSTM(32, input_shape=(512, 2), return_sequences=False)(LSTM1)
        dense4=Dense(32,activation='relu')(LSTM5)
        output = Dense(1,activation='linear')(dense4)
        model = Model(input=inputs, output=[output])

        model.compile(optimizer=Adam(lr=1e-4), loss='mae')
        #loss='binary_crossentropy'
        return model


    def train(self):
        model = self.get_net()
        print("got net")
        history = LossHistory()
        VALOS = ValueLoss()

        model_checkpoint = ModelCheckpoint('LSTMnet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(X_train, Y_train, shuffle=True, epochs=10, batch_size=100, callbacks=[history, VALOS, model_checkpoint], validation_data=(X_test, Y_test))

        print('predict test data')
        Vt_result = model.predict(X_test, batch_size=100)
        print(Vt_result)
        sio.savemat('test_dpm.mat', {'result': Vt_result})

        plt.figure(1)
        ax1 = plt.subplot(111)
        plt.figure(1)
        plt.plot(history.losses)
        plt.sca(ax1)
        plt.plot(VALOS.vallos)
        plt.show()

if __name__ == '__main__':
    myunet = myLSTM()
    myunet.train()






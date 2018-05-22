import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from scipy.misc import imsave
from keras.models import *
from keras.layers import Dense, Activation,LSTM, Conv1D
import scipy.io as sio
from keras.layers import Conv2D, MaxPooling2D, Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, LocallyConnected2D
from keras.models import load_model
from keras.optimizers import *
from  keras.callbacks import ModelCheckpoint,Callback
import matplotlib.pyplot as plt
import h5py


load_fn1 = r'C:\Users\david\Desktop\input.mat'
data1 = sio.loadmat(load_fn1)
L1 = data1['patch']

load_fn2 = r'C:\Users\david\Desktop\output.mat'
data2 = sio.loadmat(load_fn2)
L2 = data2['value']
X_train=np.zeros((500,256,2))
X_test=np.zeros((100,256,2))
X_train[:,:,:], Y_train = L1[0:500,:,:], L2[0:500]
X_test[:,:,:], Y_test = L1[501:601,:,:], L2[501:601]

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
    def __init__(self, time_length=256, channels=2):
        self.time_length= time_length
        self.channels = channels

    def get_unet(self):
        inputs = Input((self.time_length, self.channels))
        Conv1 = Conv1D(16, 1, input_shape=(256, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            inputs)
        print('Conv1.shape', Conv1.shape)
        LSTM1 = LSTM(32,return_sequences=True)(Conv1)

        Conv2 = Conv1D(16, 5, input_shape=(256, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            inputs)
        print('Conv2.shape', Conv2.shape)
        LSTM2 = LSTM(32, return_sequences=True)(Conv2)

        Conv3 = Conv1D(16, 9, input_shape=(256, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            inputs)
        print('Conv3.shape', Conv3.shape)
        LSTM3 = LSTM(32, return_sequences=True)(Conv3)

        merge1=merge([LSTM1,LSTM2,LSTM3],mode='concat',concat_axis=-1)
        print('merge1.shape',merge1.shape)
        Conv4=Conv1D(64,3,input_shape=(256,96),activation='relu',padding='same', kernel_initializer='he_normal')(merge1)
        flatten=Flatten()(Conv4)
        #LSTM1 = LSTM(16,input_shape=(512,2),return_sequences=True)(inputs)
        #LSTM2 = LSTM(32, input_shape=(512, 2), return_sequences=False)(LSTM1)
        dense1=Dense(32,activation='relu')(flatten)
        output = Dense(1,activation='linear')(dense1)
        model = Model(input=inputs, output=output)

        model.compile(optimizer=Adam(lr=1e-4), loss='mae')
        #loss='binary_crossentropy'
        return model

    def train(self):
        model = self.get_unet()
        print("got unet")
        history = LossHistory()
        VALOS = ValueLoss()

        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(X_train, Y_train, shuffle=True, epochs=10, batch_size=100, callbacks=[history, VALOS ,model_checkpoint], validation_data=(X_test, Y_test))

        print('predict test data')
        result = model.predict(X_test, batch_size=1)
        np.save('imgs_mask_test.npy', result)

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






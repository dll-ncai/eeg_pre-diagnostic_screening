import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN,LSTM, Dense, Activation, Bidirectional
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Convolution2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
print('starting')

file_path1 = 'tr_feat.mat'
A01T = h5py.File(file_path1,'r')
tr_data = np.copy(A01T['x'])          #............
#tr_data = np.swapaxes(tr_data, 1,2)            #comment it for final run
tr_data = np.asarray(tr_data, dtype=np.float32)   #............
print(tr_data.dtype)
print(tr_data.shape)
bs,t,f = tr_data.shape
print('training data has been loaded')
file_path2 = 'tr_f_label.mat'
A02T = h5py.File(file_path2,'r')
tr_labels= np.copy(A02T['y'])
print(tr_labels)
print(tr_labels.shape)
print(tr_labels.dtype)
tr_labels = tr_labels[0,0:tr_data.shape[0]:1]
print(tr_labels.shape)
print(tr_labels.dtype)
tr_labels = np.asarray(tr_labels, dtype=np.float32)     #............
print(tr_labels.shape)
print(tr_labels.dtype)
enc_tr_labels = to_categorical(tr_labels, num_classes=2)              
tr_labels= enc_tr_labels
print(tr_labels.shape)
print(tr_labels.dtype)
print('training labels have been loaded')

# ----------------------Model-----------------------
from keras.layers import Input,Dense,concatenate,Flatten,GRU,Conv1D
from keras.models import Model
inputsin= Input(shape=(t,f))

x=LSTM(50,activation='tanh',unroll=True)(inputsin)
predictions = Dense(2,activation='softmax')(x)
model = Model(inputs=inputsin, outputs=predictions)
#opt=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
print(model.metrics_names)
print(model.summary())
#
# # learning schedule callback
# lrate = LearningRateScheduler(step_decay)

# early stopping
es = EarlyStopping(monitor='val_loss', min_delta=0.01, mode='min', verbose=1, patience=15)                          #patience
mc = ModelCheckpoint('model_LSTM_1acc.hdf5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)        #filepath (save model as)
mces = ModelCheckpoint('model_LSTM_1loss.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)      #filepath (save model as)
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('LSTM.csv', append=True, separator=';')

# fit model
history=model.fit(tr_data,tr_labels,validation_split=0.2,epochs=500,batch_size=64,verbose=1,callbacks=[es, mc,mces, csv_logger],shuffle=True) #epochs #split #shuffle


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()
print('The End')
print('The End')


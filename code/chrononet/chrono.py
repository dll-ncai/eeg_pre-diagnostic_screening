/********************************************
* Author: Rahat Ul Ain 
* Based on chrononet implementation by Kunal Patel
* location: https://github.com/kunalpatel1793/Neural-Nets-Final-Project
********************************************/
from pdb import set_trace
import mne
import pandas as pd
import numpy as np
import math
import os
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,LSTM, Dense, Activation, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
def readDatafromPath(path):

	matrix= np.empty((15000, 22), dtype='f')
	for file in os.listdir(path):
		if '.edf' in file:
			f=os.path.join(path, file)
			print(f)
			edf_file = mne.io.read_raw_edf(f, montage = None, eog = ['FP1', 'FP2', 'F3', 'F4',
			'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
			'T3', 'T4', 'T5', 'T6', 'PZ', 'FZ', 'CZ', 'A1', 'A2'
			], verbose = 'error', preload = True)
			edf_file_down_sampled = edf_file.resample(250, npad = "auto")# set sampling frequency to 250 Hz
			ed = edf_file_down_sampled.to_data_frame(picks = None, index = None, scaling_time = 1000.0, scalings = None,
			copy = True, start = None, stop = None)# converting into dataframe
			Fp1_Fp7 = (ed.loc[: , 'FP1']) - (ed.loc[: , 'F7'])
			FP2_F8 = (ed.loc[: , 'FP2']) - (ed.loc[: , 'F8'])
			F7_T3 = (ed.loc[: , 'F7']) - (ed.loc[: , 'T3'])
			F8_T4 = (ed.loc[: , 'F8']) - (ed.loc[: , 'T4'])
			T3_T5 = (ed.loc[: , 'T3']) - (ed.loc[: , 'T5'])
			T4_T6 = (ed.loc[: , 'T4']) - (ed.loc[: , 'T6'])
			T5_O1 = (ed.loc[: , 'T5']) - (ed.loc[: , 'O1'])
			T6_O2 = (ed.loc[: , 'T6']) - (ed.loc[: , 'O2'])
			A1_T3 = (ed.loc[: , 'A1']) - (ed.loc[: , 'T3'])
			T4_A2 = (ed.loc[: , 'T4']) - (ed.loc[: , 'A2'])
			T3_C3 = (ed.loc[: , 'T3']) - (ed.loc[: , 'C3'])
			C4_T4 = (ed.loc[: , 'C4']) - (ed.loc[: , 'T4'])
			C3_CZ = (ed.loc[: , 'C3']) - (ed.loc[: , 'CZ'])
			CZ_C4 = (ed.loc[: , 'CZ']) - (ed.loc[: , 'C4'])
			FP1_F3 = (ed.loc[: , 'FP1']) - (ed.loc[: , 'F3'])
			FP2_F4 = (ed.loc[: , 'FP2']) - (ed.loc[: , 'F4'])
			F3_C3 = (ed.loc[: , 'F3']) - (ed.loc[: , 'C3'])
			F4_C4 = (ed.loc[: , 'F4']) - (ed.loc[: , 'C4'])
			C3_P3 = (ed.loc[: , 'C3']) - (ed.loc[: , 'P3'])
			C4_P4 = (ed.loc[: , 'C4']) - (ed.loc[: , 'P4'])
			P3_O1 = (ed.loc[: , 'P3']) - (ed.loc[: , 'O1'])
			P4_O2 = (ed.loc[: , 'P4']) - (ed.loc[: , 'O2'])
			data = {
			'Fp1_Fp7': Fp1_Fp7,
			'FP2_F8': FP2_F8,
			'F7_T3': F7_T3,
			'F8_T4': F8_T4,
			'T3_T5': T3_T5,
			'T4_T6': T4_T6,
			'T5_O1': T5_O1,
			'T6_O2': T6_O2,
			'A1_T3': A1_T3,
			'T4_A2': T4_A2,
			'T3_C3': T3_C3,
			'C4_T4': C4_T4,
			'C3_CZ': C3_CZ,
			'CZ_C4': CZ_C4,
			'FP1_F3': FP1_F3,
			'FP2_F4': FP2_F4,
			'F3_C3': F3_C3,
			'F4_C4': F4_C4,
			'C3_P3': C3_P3,
			'C4_P4': C4_P4,
			'P3_O1': P3_O1,
			'P4_O2': P4_O2
			}
			new_data_frame = pd.DataFrame(data, columns = ['Fp1_Fp7', 'FP2_F8', 'F7_T3', 'F8_T4', 'T3_T5', 'T4_T6', 'T5_O1', 'T6_O2', 'A1_T3', 'T4_A2', 'T3_C3', 'C4_T4', 'C3_CZ',
			'CZ_C4', 'FP1_F3', 'FP2_F4', 'F3_C3', 'F4_C4', 'C3_P3', 'C4_P4', 'P3_O1', 'P4_O2'
			])
			fs = edf_file_down_sampled.info['sfreq']
			[row, col] = new_data_frame.shape
			n = math.ceil(row / (15000 - (fs * 5)))
			i = 0;
			j = 15000;
			print(row)

			for y in range(n - 1):

				print(i, j)
				if y>1:
					break
				elif y == 0:
					example_1 = new_data_frame[0: 15000]
					matrix=np.dstack((matrix,example_1.to_numpy()))
				elif j < row:
					example = new_data_frame[i: j]
					matrix = np.dstack((matrix, example.to_numpy()))
				else :
					example = new_data_frame[-15000: ]
					matrix = np.dstack((matrix, example.to_numpy()))
					break
				i = int(j - (fs * 5))
				j = int(j + 15000 - (fs * 5))
			print(matrix.shape)
		matrix=matrix[:,:,1:]
	return matrix.astype('float32')

import config
print('starting')
normal_train = readDatafromPath(path = config.normaldir)
normal_train_dim = normal_train.shape[-1]
# print("normal original dim")
# print(normal_train_dim)
normal_train_zeros = np.zeros(normal_train_dim)
# print("zeros array dim")
# print(normal_train_zeros)

abnormal_train = readDatafromPath(path = config.abnormaldir)
abnormal_train_dim = abnormal_train.shape[-1]
#print(abnormal_train_dim)
abnormal_train_ones = np.ones(abnormal_train_dim)
#print(abnormal_train_dim)

train_data = np.dstack((normal_train, abnormal_train))
train_label = np.append(normal_train_zeros, abnormal_train_ones)

train_data = np.swapaxes(train_data,0,2)

bs,t,f = train_data.shape


print(train_data.shape)
print(train_label.shape)
print(train_data.dtype)
print(train_label.dtype)
enc_labels = to_categorical(train_label, num_classes=2)              
train_label= enc_labels
print(train_data.shape)
print(train_label.shape)
print(train_data.dtype)
print(train_label.dtype)
print('training labels have been loaded')

# ----------------------CHRONONET Testing-----------------------
from tensorflow.keras.layers import Input,Dense,concatenate,Flatten,GRU,Conv1D
from tensorflow.keras.models import Model
inputsin= Input(shape=(t,f))
# ------------------First Inception
tower1 = Conv1D(32, 2, strides=2,activation='relu',padding="causal")(inputsin)
tower1 = BatchNormalization()(tower1)
tower2 = Conv1D(32, 4, strides=2,activation='relu',padding="causal")(inputsin)
tower2 = BatchNormalization()(tower2)
tower3 = Conv1D(32, 8, strides=2,activation='relu',padding="causal")(inputsin)
tower3 = BatchNormalization()(tower3)
x = concatenate([tower1,tower2,tower3],axis=2)
x = Dropout(0.45)(x)

# ----------------------Second Inception
tower1 = Conv1D(32, 2, strides=2,activation='relu',padding="causal")(x)
tower1 = BatchNormalization()(tower1)
tower2 = Conv1D(32, 4, strides=2,activation='relu',padding="causal")(x)
tower2 = BatchNormalization()(tower2)
tower3 = Conv1D(32, 8, strides=2,activation='relu',padding="causal")(x)
tower3 = BatchNormalization()(tower3)
x = concatenate([tower1,tower2,tower3],axis=2)
x = Dropout(0.45)(x)

# ----------------------------------Third Inception
tower1 = Conv1D(32, 2, strides=2,activation='relu',padding="causal")(x)
tower1 = BatchNormalization()(tower1)
tower2 = Conv1D(32, 4, strides=2,activation='relu',padding="causal")(x)
tower2 = BatchNormalization()(tower2)
tower3 = Conv1D(32, 8, strides=2,activation='relu',padding="causal")(x)
tower3 = BatchNormalization()(tower3)
x = concatenate([tower1,tower2,tower3],axis=2)
x = Dropout(0.55)(x)

res1 = GRU(32,activation='tanh',return_sequences=True)(x)
res2 = GRU(32,activation='tanh',return_sequences=True)(res1)
res1_2 = concatenate([res1,res2],axis=2)
res3 = GRU(32,activation='tanh',return_sequences=True)(res1_2)
x = concatenate([res1,res2,res3])
x = GRU(32,activation='tanh')(x)

predictions = Dense(2,activation='softmax')(x)
model = Model(inputs=inputsin, outputs=predictions)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
print(model.metrics_names)
print(model.summary())

# early stopping
es = EarlyStopping(monitor='val_loss', min_delta=0.01, mode='min', verbose=1, patience=25)                          #patience
mc = ModelCheckpoint('modelbest_acc.hdf5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)        #filepath (save model as)
mces = ModelCheckpoint('modelbest_loss.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)      #filepath (save model as)

# fit model
hist=model.fit(train_data,train_label,validation_split=0.2,epochs=500,batch_size=32,verbose=1,callbacks=[es, mc,mces],shuffle=True) #epochs #split #
print('The End')

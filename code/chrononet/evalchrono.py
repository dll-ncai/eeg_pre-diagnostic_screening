/*******************
* Author: Rahat Ul Ain
********************/
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
print('starting')
normal_eval = readDatafromPath(path = "normal/eval")
normal_eval_dim = normal_eval.shape[-1]
# print("normal original dim")
# print(normal_eval_dim)
normal_eval_zeros = np.zeros(normal_eval_dim)
# print("zeros array dim")
# print(normal_eval_zeros)

abnormal_eval = readDatafromPath(path = "abnormal/eval")
abnormal_eval_dim = abnormal_eval.shape[-1]
#print(abnormal_eval_dim)
abnormal_eval_ones = np.ones(abnormal_eval_dim)
#print(abnormal_eval_dim)

eval_data = np.dstack((normal_eval, abnormal_eval))
eval_label = np.append(normal_eval_zeros, abnormal_eval_ones)

eval_data = np.swapaxes(eval_data,0,2)

bs,t,f = eval_data.shape

enc_labels = to_categorical(eval_label, num_classes=2)
eval_label= enc_labels


print('testing')
# load the saved accuracy model
savedmodel = load_model('model3flipped_loss.hdf5')                      # model name
print('model loaded')
test_score = savedmodel.evaluate(test_data, test_labels, batch_size=32)
print ("Evaluation loss and Evaluation accuracy for best accuracy model is: ", test_score)
print ('The End')

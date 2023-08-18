# CNN & CNN-LSTM code to compute biomass time series modeling
# Author: Efrain Noa-Yarasca. Email: enoay7@yahoo.com
# Date: 04-30-2023

import random, csv
import matplotlib.pyplot as plt
import numpy as np

from pandas import read_csv
from math import sqrt
from numpy import array, mean, std, median
from pandas import DataFrame
from pandas import concat

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping
from keras.layers import TimeDistributed          

from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D 


def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def smape(actual, forecast):
    return 1/len(actual) * np.sum(2 * np.abs(forecast-actual) / (np.abs(actual) + np.abs(forecast))*100)

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

def series_to_supervised2(data, n_steps_in, n_steps_out):
  X, y = list(), list()
  for i in range(len(data)):
    end_ix = i + n_steps_in
    out_end_ix = end_ix + n_steps_out
    if out_end_ix > len(data):
      break
    seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

def reshape_data_cnn(train_x2, train_y2):
    train_y2 = train_y2.reshape((train_y2.shape[0], train_y2.shape[1])) 
    return train_x2, train_y2


def model_cnn (n_in, n_out, activ_m, n_nodes):
    model = Sequential()
    n_features = 1
    #model.add(Conv1D(filters=16, kernel_size=3, strides=1, activation='relu', input_shape=(n_in, n_features), padding = 'same'))  # p02
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', input_shape=(n_in, n_features), padding = 'valid'))  # p03
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    
    #model.add(Dense(50, activation='relu'))    # For P02
    model.add(Dense(100, activation='relu'))   # For P03
    model.add(Dense(n_out))
    model.compile(loss='mse', optimizer= Adamax(learning_rate = 0.0005), metrics=['mse']) 
    #model.compile(loss='mse', optimizer = SGD(learning_rate=0.005), metrics=['mse']) # For P03
    #model.compile(loss='mse', optimizer = SGD(learning_rate=0.02), metrics=['mse']) # For P03
    return model

def model_cnn_lstm(n_sub_in, n_out, activ_m, n_nodes):
    n_features = 1
    model = Sequential()
    model.add(TimeDistributed(Conv1D(16, 3, strides=1, activation='relu', padding = 'same'), input_shape=(None, n_sub_in, n_features)))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(n_out))
    model.compile(loss='mse', optimizer= Adam(learning_rate = 0.001))
    #model.compile(loss='mse', optimizer= Adam(learning_rate = 0.001), metrics=['mse'])
    #model.compile(loss='mse', optimizer = SGD(learning_rate=0.001), metrics=['mse']) # For P03
    return model


series = read_csv('D:/work/research_t/pp_02/supplementary/time_series_data.csv', header=0, index_col=0)  # From local folder
#series = read_csv('../time_series_data.csv', header=0, index_col=0)  # 
site = 'TS1_S369' # 'TS2_S372',	'TS3_S391',	'TS4_S426',	'TS5_S21642'
#site = 'S_369' #,'S_372','S_391','S_426','S_21642'

data = series[site].values
dd = data.reshape(496,1)

### Normalization ----------------------
from sklearn.preprocessing import MinMaxScaler
min_sc, max_sc = 0, 1
sc = MinMaxScaler(feature_range = (min_sc, max_sc))
data_sc = sc.fit_transform(dd) # Scaling
data = data_sc #
# --------------------------------------
data = data.ravel()
print (type(data), data.shape)
print () #'''


import time
start_time = time.perf_counter () 

list_n_input = [24]                                   # < ---------------------------- INPUTS
all_rmse2, all_rmse2_inv = [],[]
all_mape2, all_mape2_inv = [],[]
all_smape2, all_smape2_inv = [],[]
pred2_inv_r_n3 = []
cc1 = 1
for ni in list_n_input:
  print(colored(153, 51, 255, (site + "  N_INPUTs: ........... "+ str(ni)))) #Magenta

  n_test = 24            
  n_repeats = 1
  scores_rmse2, scores_rmse2_inv = [],[]
  scores_mape2, scores_mape2_inv = [],[]
  scores_smape2, scores_smape2_inv = [],[]
  pred2_inv_r_n2 = []
  for i in range(n_repeats):
    print(colored(0, 255, 255, (site + '  n_repeat ..... '+ str(i+1)+"/"+ str(n_repeats)+
                              '    n_in...'+ str(cc1)+"/"+ str(len(list_n_input))+' ('+str(ni)+')')))
    n_input = ni # 
    n_out = 1
    n_features = 1
    train, test = train_test_split(data, n_test)  
    test_m = data[-(n_input+n_test):]

    # ------ Converting data into supervised
    train_x2, train_y2 = series_to_supervised2(train, n_input, n_out)       # ---------> Call Fn
    test_x2, test_y2 = series_to_supervised2(test_m, n_input, n_out)        # ---------> Call Fn
    print ("Shapes (train_x2, train_y2,test_x2, test_y2): >>>>> ", train_x2.shape, train_y2.shape, test_x2.shape, test_y2.shape) 
    train_x2, train_y2 = reshape_data_cnn(train_x2, train_y2)
    print ("Shapes (train_x2, train_y2,test_x2, test_y2): >>>>> ", train_x2.shape, train_y2.shape, test_x2.shape, test_y2.shape) 
    print ("-------------------------------------------------------------------------------")  # 
    #train_x2 = train_x2.reshape(train_x2.shape[0], train_x2.shape[1], n_features)       # Page 91
    #test_x2 = test_x2.reshape(test_x2.shape[0], test_x2.shape[1], n_features)
    
    

    # ------ Setting the model - define config
    activat_set = 'relu'
    n_nodes0 = 50 # 

    ### ------------------------------------   For CNN model ------------------  %%%%%%%%%%%%%%%%%%%%%%%% (1)
    print(colored(255, 0, 0, ('CNN ....... ')))
    #model = model_cnn(n_input, n_out, activat_set,  n_nodes0) 
    
    
    ### ------------------------------------   For CNN-LSTM model -------------  *************
    
    print(colored(255, 0, 0, ('CNN-LSTM ....... ')))
    n_seq = 2                                                                       # ------------> NEW  SETTINGS
    n_sub_in = 12
    model = model_cnn_lstm(n_sub_in, n_out, activat_set,  n_nodes0) 
    #print("train_x2: \n", train_x2)
    train_x3 = train_x2.reshape(train_x2.shape[0], n_seq, n_sub_in, n_features)
    train_y3 = train_y2.ravel()
    #print("train_x3: \n", train_x3)
    test_x3 = test_x2.reshape(test_x2.shape[0], n_seq, n_sub_in, n_features)
    test_y3 = test_y2.ravel()
    print("------------- shapes for CNN-LSTM: ", train_x3.shape, train_y3.shape, test_x3.shape, test_y3.shape) # '''
    #print(dasdsada)

    # ------ Fit the model
    #n_batch = 64           # For P02
    n_batch = 16                # For P03
    #n_epochs = 100          # fOR p02
    n_epochs = 500              # fOR P03
    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    #early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    #  -----   For CNN --------------------------- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (2)
    #hist_model = model.fit(train_x2, train_y2, epochs=n_epochs, batch_size = n_batch, verbose=0, validation_data=(test_x2, test_y2), callbacks=[early_stop]) 
    
    #  -----   For CNN-LSTM --- ***************************************
    hist_model = model.fit(train_x3, train_y3, epochs=n_epochs, batch_size = n_batch, verbose=0, validation_data=(test_x3, test_y3), callbacks=[early_stop])  
    
    ### Plot Learning-curve (summarize history for accuracy)
    
    plt.plot(hist_model.history['loss'])         # plt.plot(hist_model.history['mse']) #
    plt.plot(hist_model.history['val_loss'])     # plt.plot(hist_model.history['val_mse'])
    plt.title('Learning curve - model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.yscale('log')
    plt.show() #'''
    
    
    
    ### Prediction
    prediction2 = list()
    history2 = [x for x in train] # 
    for j in range(int(np.ceil(len(test)/n_out))): 
      #x_input2 = array(history2[-n_input:]).reshape(1, n_input,1)            # For CNN      %%%%%%%%%%%%%%%%%%%%%%%% (3)
      x_input2 = array(history2[-n_input:]).reshape(1, n_seq, n_sub_in,1)     # For CNN-LSTM  ***************************

      yhat2 = model.predict(x_input2, verbose=0)      # prediction
      prediction2.append(yhat2[0])

      ee2 = yhat2[0].reshape(n_out) # Reshape the just predicted values
      aa1 = np.r_[history2, ee2] # Add the just predicted values to the last 24 values from dataset
      history2 = aa1[-n_input:] # Take the last 24 values to input into the next prediction
        
    print("----------------------------------------------------------------------------------")
    pred2_flat = np.concatenate(prediction2).ravel().tolist() 

    # estimate prediction error
    rmse2 = sqrt(mean_squared_error(test, pred2_flat)) 
    mape2 = mean_absolute_percentage_error(test, pred2_flat)
    smape2 = smape(test, pred2_flat) 

    scores_rmse2.append(rmse2)  
    scores_mape2.append(mape2) 
    scores_smape2.append(smape2) 

    pred2_inv_flat = sc.inverse_transform(prediction2).reshape(n_test,1) 
    test_invt = sc.inverse_transform(test.reshape(n_test,1)) # (6,2)

    rmse2_inv = sqrt(mean_squared_error(test_invt, pred2_inv_flat)) # Error using inversed-transformed data 
    mape2_inv = mean_absolute_percentage_error(test_invt, pred2_inv_flat)
    smape2_inv = smape(test_invt, pred2_inv_flat.reshape(n_test,1))

    scores_mape2_inv.append(mape2_inv)
    scores_rmse2_inv.append(rmse2_inv)
    scores_smape2_inv.append(smape2_inv)

    print(' > RMSE2 %.3f'% rmse2, ' > RMSE2_inv %.3f'% rmse2_inv)
    print(' > MAPE2 %.3f'% mape2, ' > MAPE2_inv %.3f'% mape2_inv)
    print(' > SMAPE2 %.3f'% smape2, ' > SMAPE2_inv %.3f'% smape2_inv)

    pred2_inv_r_n1 = [item for sublist in pred2_inv_flat.tolist() for item in sublist]  # ***
    pred2_inv_r_n2.append(pred2_inv_r_n1) 
    print("------------------------------------------------------------------------")
    print()
  
  # summarize scores_rmse1 (summarize model performance)    
  print('%s: %.3f SD (+/- %.3f)' % ('RMSE2', mean(scores_rmse2), std(scores_rmse2)))
  print('%s: %.3f SD (+/- %.3f)' % ('RMSE2_i', mean(scores_rmse2_inv), std(scores_rmse2_inv)))
  print('%s: %.3f SD (+/- %.3f)' % ('MAPE2', mean(scores_mape2), std(scores_mape2)))
  print('%s: %.3f SD (+/- %.3f)' % ('MAPE2_i', mean(scores_mape2_inv), std(scores_mape2_inv)))
  print('%s: %.3f SD (+/- %.3f)' % ('SMAPE2', mean(scores_smape2), std(scores_smape2)))
  print('%s: %.3f SD (+/- %.3f)' % ('SMAPE2_i', mean(scores_smape2_inv), std(scores_smape2_inv)))
  
  all_rmse2.append(scores_rmse2)
  all_rmse2_inv.append(scores_rmse2_inv)
  all_mape2.append(scores_mape2)
  all_mape2_inv.append(scores_mape2_inv)
  all_smape2.append(scores_smape2)
  all_smape2_inv.append(scores_smape2_inv)
  pred2_inv_r_n3.append(pred2_inv_r_n2)     # ***  recurrent   

  cc1 += 1
  print ()
  
# --------------- Print running time ----------- #
end_time = time.perf_counter ()
print(end_time - start_time, "seconds")
# ------------- End Print running time ----------- #





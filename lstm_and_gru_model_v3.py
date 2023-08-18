# LSTM & GRU code to compute biomass time series modeling
# Author: Efrain Noa-Yarasca. Email: enoay7@yahoo.com
# Date: 04-30-2023

import numpy as np
import math
import matplotlib.pyplot as plt

from math import sqrt
from numpy import mean, std, array
from pandas import DataFrame
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LSTM, GRU
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from sklearn.metrics import mean_absolute_percentage_error
from keras.layers import Bidirectional

import time
start_time = time.perf_counter () 
# ------------------------------ Set running time ----------------------------- #

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

def smape(actual, forecast):
    return 1/len(actual) * np.sum(2 * np.abs(forecast-actual) / (np.abs(actual) + np.abs(forecast))*100)

# transform list into supervised learning format
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


def model_lstm_bidirect (n_in, n_steps_out, n_features, activ_m, n_nodes):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_in, n_features), dropout=0.15)) # Bidirectional LSTM. Learn both forward and backwards
    model.add(Dense(n_steps_out))
    model.compile(loss='mse', optimizer= Adamax(learning_rate = 0.01)) 
    return model

def model_gru (n_in, n_steps_out, n_features, activ_m, n_nodes):
    model_1 = Sequential()
    model_1.add(GRU(50, activation='relu', input_shape=(n_in, n_features), dropout=0.15))
    model_1.add(Dense(n_steps_out))
    model_1.compile(optimizer='adam', loss='mse')
    return model_1
    
def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

# ----------------------------------------------------
# Read data
series = read_csv('D:/work/research_t/pp_02/supplementary/time_series_data.csv', header=0, index_col=0)  # From local folder
series = read_csv('../time_series_data.csv', header=0, index_col=0)  # 
site = 'TS1_S369' # 'TS2_S372',	'TS3_S391',	'TS4_S426',	'TS5_S21642'
#site = 'S_369' #,'S_372','S_391','S_426','S_21642' 


data = series[site].values
data_ax = series[site].values
dd = data.reshape(496,1)
print("dd.shape:  ", dd.shape)

### Normalization ----------------------
from sklearn.preprocessing import MinMaxScaler
min_sc, max_sc = 0, 1
sc = MinMaxScaler(feature_range = (min_sc, max_sc))
data_sc = sc.fit_transform(dd) # Scaling
data = data_sc #'''
# --------------------------------------
data = data.ravel()
print (type(data), data.shape)
print ()

list_n_input = [24]

all_rmse2, all_rmse2_inv = [],[]
all_mape2, all_mape2_inv = [],[]
all_smape2, all_smape2_inv = [],[]
pred2_inv_r_n3 = []

cc1 = 1
for ni in list_n_input:
    print(colored(255, 255, 0, ("N_INPUTs:  ___________ "+ str(ni))))
    n_test = 24 #
    n_repeats = 25
    
    scores_rmse2, scores_rmse2_inv = [],[]
    scores_mape2, scores_mape2_inv = [],[]
    scores_smape2, scores_smape2_inv = [],[]
    pred2_inv_r_n2 = []

    for i in range(n_repeats):
        print(colored(0, 255, 255, ('n_repeat ....... '+ str(i+1)+"/"+ str(n_repeats)+
                                    '  '+ str(cc1)+"/"+ str(len(list_n_input))+' '+str(ni))))
        predictions2 = list()
        n_input = ni # 2
        n_out = 1
        n_features = 1
        
        train, test = train_test_split(data, n_test)  # split dataset 
        test_m1 = data[-(n_input + n_test):] # 24 values from train are add to test data (TS-1)
    
        ### Covert Train-data into input/output (Convert to Supervised) # ------> Fn
        train_x, train_y = series_to_supervised2(train, n_input, n_out)
        print ("train_x.shape, train_y.shape: ", train_x.shape, train_y.shape, "  ", type(train_x))
        
        ### Covert Test-data into input/output -  (Convert to Supervised) # ------> Fn
        test_x, test_y = series_to_supervised2(test_m1, n_input, n_out)
        print ("test_x.shape, test_y.shape: ", test_x.shape, test_y.shape, "  ", type(train_x))
        print ("------------------------------------------------------------")
        n_nodes = 100
        n_epochs = 200
        n_batch = 64
        activ_m = 'relu'

        ### LSTM Model
        #model_lstm = model_lstm_bidirect(n_input, n_out, n_features, activ_m, n_nodes)   ## LSTM Model
        
        ### GRU Model
        model_lstm = model_gru(n_input, n_out, n_features, activ_m, n_nodes)              ## GRU Model
        
        early_stop = EarlyStopping(monitor='loss', patience=30, verbose=1, mode='auto')
        hist_model_lstm = model_lstm.fit(train_x, train_y, epochs = n_epochs, batch_size = n_batch, 
                               verbose=0, validation_data=(test_x, test_y), callbacks=[early_stop])
        
        ### Prediction
        history2 = [x for x in train]
        a1 = 0
        for j in range(math.trunc(n_test/n_out)):
            x_input2x = array(history2[-n_input:]).reshape(1, n_input)
            yhat2 = model_lstm.predict(x_input2x, verbose=0)
            predictions2.append((yhat2[0]))
            ee2 = yhat2[0].reshape(n_out) # Reshape the just predicted values
            aa1 = np.r_[history2, ee2] # Add the just predicted values to the last 24 values from dataset
            history2 = aa1[-n_input:] # Take the last 24 values to input into the next prediction

        pred2_flat = np.concatenate(predictions2).ravel().tolist() # Prediction with recurrency
        
        # estimate prediction error
        rmse2 = sqrt(mean_squared_error(test, pred2_flat))
        mape2 = mean_absolute_percentage_error(test, pred2_flat)
        smape2 = smape(test, pred2_flat)
        scores_rmse2.append(rmse2)
        scores_mape2.append(mape2)
        scores_smape2.append(smape2)

        pred2_inv_flat = sc.inverse_transform(predictions2) # Inverse 
        test_invt = sc.inverse_transform(test.reshape(n_test,1))
        
        rmse2_inv = sqrt(mean_squared_error(test_invt, pred2_inv_flat)) # Error using inversed-transformed data
        mape2_inv = mean_absolute_percentage_error(test_invt, pred2_inv_flat)
        smape2_inv = smape(test_invt, pred2_inv_flat.reshape(n_test,1))
        scores_mape2_inv.append(mape2_inv)
        scores_rmse2_inv.append(rmse2_inv)
        scores_smape2_inv.append(smape2_inv)
        
        
        pred2_inv_r_n1 = [item for sublist in pred2_inv_flat.tolist() for item in sublist]  # ***
        pred2_inv_r_n2.append(pred2_inv_r_n1)  
        
        print("-------------------------------------------------------")
        print()
        
    # summarize scores (summarize model performance)    
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
    
    # BOX plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,2), dpi=250)
    title_fig = str(site)+"; n="+str(n_repeats)
    fig.suptitle(title_fig, fontsize=10)
    ax1.boxplot(scores_rmse2)
    ax1.title.set_text('Iterative prediction transformed data')
    ax1.title.set_size(7)
    ax1.grid()
    ax2.boxplot(scores_rmse2_inv)
    ax2.title.set_text('Iterative prediction inverse-transformed data')
    ax2.title.set_size(7)
    ax2.grid()
    pyplot.show() #'''
    
    cc1 += 1
    print ()

### BOX PLOTs
names = list_n_input# ['1', '2']#, '3','4', '6', '12','18', '24', '36','48', '60', '72','84', '96']

# Plot RMSE
fig, (ax1) = plt.subplots(1, 1, figsize=(6,4), dpi=250)
ax1.boxplot(all_rmse2)
means = [np.mean(data) for data in all_rmse2] #*
positions = np.arange(len(list_n_input)) + 1 #*
ax1.plot(positions, means, 'bo', markersize=4) #*
ax1.title.set_text('RMSE   Site: '+ site + ' - Recurrent')
ax1.set_xticklabels(names)
ax1.set_xlabel('Number of time-step inputs')
ax1.set_ylabel('MAPE')
ax1.grid()
pyplot.show() #'''

# Plot MAPE
fig, (ax1) = plt.subplots(1, 1, figsize=(6,4), dpi=250)
ax1.boxplot(all_mape2)
means = [np.mean(data) for data in all_mape2] #*
positions = np.arange(len(list_n_input)) + 1 #*
ax1.plot(positions, means, 'bo', markersize=4) #*
ax1.title.set_text('MAPE   Site: '+ site + ' - Recurrent')
ax1.set_xticklabels(names)
ax1.set_xlabel('Number of time-step inputs')
ax1.set_ylabel('MAPE')
ax1.grid()
pyplot.show() #'''


### Plot Prediction Time-Series 
flat_pred = [item for sublist in test_invt.tolist() for item in sublist]
for sim in pred2_inv_r_n3[0]:
    pl = plt.plot(sim, color="#bfbfbf")
plt.plot(flat_pred, color="red")
plt.title(site+"  Recurrent", x=0.5, y=0.9)
plt.show()

# --------------- Print running time ----------- #
end_time = time.perf_counter ()
print(end_time - start_time, "seconds")
# ------------- End Print running time ----------- #












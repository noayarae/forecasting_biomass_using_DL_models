# SARIMA code to compute biomass time series modeling
# Author: Efrain Noa-Yarasca. Email: enoay7@yahoo.com
# Date: 04-30-2023

import numpy as np
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_percentage_error
# -------------------------------------------------------------------------------- #

def smape(actual, forecast):
  return 1/len(actual) * np.sum(2 * np.abs(forecast-actual) / (np.abs(actual) + np.abs(forecast))*100)

# -------------------------------------------------------------------------------- #
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

import time
start_time = time.perf_counter () 
# ------------------------------ Set running time ----------------------------- #

series = read_csv('D:/work/research_t/forecast_dec/003_340_ts_data_for_decomp_v1.csv', header=0, index_col=0)
#series = read_csv('../time_series_data.csv', header=0, index_col=0)  
site = 'S_21642' # 'S_369' # 'S_372' # 'S_391' #   'S_426' # 'S_21642' 

data = series[site].values
dd = data.reshape(496,1)

X = data
n_test = 6 # forecast horizon
size = 496 - n_test 
train, test = X[0:size], X[size:len(X)]

history2 = [x for x in train]
pred2 = list()

r2 = []
model_t = []
aic_m2, bic_m2 = [],[]
cc1 = 1
# Set values for (p,d,q) and (P,D,Q)
for p in [2]: #  [1,2,3]:
  for d in [1]: # [0,1]:
    for q in [3]: #  [1,2,3]:
      for P in [1]: # [1,2]:
        for D in [1]: # [0,1]:
          for Q in [2]: # [1,2]:
            try:
                print()
                print(colored(0,255,255, ("cc1: "+str(cc1))))
                print(colored(255,0,0, ("(p,1,q)(P,1,Q): "+str(p)+str(d)+str(q)+"-"+ str(P)+str(D)+str(Q))))
                history2 = [x for x in train]
                pred2 = list()
                
                for t in range(len(test)):
                  model2 = SARIMAX(history2, order=(p, d, q), seasonal_order=(P,D,Q,24))
                  model2_fit = model2.fit()
                  yhat2 = model2_fit.forecast()[0]
                  pred2.append(yhat2)
                  obs = test[t]
                  history2.append(yhat2)
                  print(colored(255, 0, 0, ("> "+str(cc1)+" - "+str(t+1)+"  y_hat2: "+ str(yhat2)+"   Expected: "+ str(obs))))
    
                # evaluate forecasts
                model_t.append(str(p)+str(d)+str(q)+'-'+str(P)+str(D)+str(Q))
                rmse2 = sqrt(mean_squared_error(test, pred2))
                mape2 = mean_absolute_percentage_error(test, pred2)
                smape2 = smape(test, pred2)
                r2.append(rmse2)
    
                aic_m2.append(model2_fit.aic)
                bic_m2.append(model2_fit.bic)
    
                print(colored(255,0,0, ("(p,d,q)(P,D,Q): "+str(p)+str(d)+str(q)+"-"+ str(P)+str(D)+str(Q))))
                print('  RMSE2: %.3f' % rmse2)
                print('  AIC2: %.3f' % model2_fit.aic)
                print('  BIC2: %.3f' % model2_fit.bic)
                print('  MAPE2: %.5f' % mape2)
                print('  sMAPE2: %.5f' % smape2)
            
            except:
                print(colored(0,255,255, ("Something else went wrong")))
                model_t.append(str(p)+str(d)+str(q)+'-'+str(P)+str(D)+str(Q))
                rmse2 = 0
                mape2 = 0
                smape2 = 0
                r2.append(rmse2)
    
                aic_m2.append(0)
                bic_m2.append(0)
    
                print(colored(255,0,0, ("(p,d,q)(P,D,Q): "+str(p)+str(d)+str(q)+"-"+ str(P)+str(D)+str(Q))))
                print('RMSE2: %.3f' % rmse2)
                print('AIC2: %.3f' % 0)
                print('BIC2: %.3f' % 0)
                print('MAPE2: %.5f' % mape2)
                print('sMAPE2: %.5f' % smape2)
                pass
    
            cc1 += 1

print("-------------------------------------------------------------------------")
print(model_t)
print(r2)
print(aic_m2)
print(bic_m2)

# --------------- Print running time ----------- #
end_time = time.perf_counter ()
print(end_time - start_time, "seconds")
# ------------- End Print running time ----------- #

### plot forecasts against actual outcomes
pyplot.plot(test, color='black', label='obs')
pyplot.plot(pred2, color='blue', label='pred2')
pyplot.legend()
pyplot.show() #'''
print()
print("Predicted values")
print(pred2)

### Print model paameters
#print(model2_fit.arparams)
#print(model2_fit.summary())




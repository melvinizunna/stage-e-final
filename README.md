# stage-e-final

#Izunna Eliogu
#ID: 146ea351c801f000

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cx=pd.read_csv('/content/Time_series_analysis_and_forecast_DATASET.csv',parse_dates=['FullDate'],index_col=['FullDate'])
cx




cx.isnull().sum()

cx.info()

cx.corr()

len(cx)

#resample data to get daily info
cx_sampled=cx.resample('D').sum()
cx_sampled

len(cx_sampled)

#plot the graph of ElecPrice vs time
x=cx.index
y=cx['ElecPrice']
plt.figure(figsize=(12,8))
plt.plot(x,y,marker='*')
plt.grid()
plt.xlabel('Datetime')
plt.ylabel('ElecPrice')

cx_sampled.info()

#exploring the data
import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize']=12,8
decomposed_series=sm.tsa.seasonal_decompose(cx['ElecPrice'],model='additive')
decomposed_series.plot()
plt.show()

#QUESTION 12

#performing adf test for sysload
from statsmodels.tsa.stattools import adfuller
adfull=adfuller(cx_sampled['SysLoad'])
print(f'p values:{round(adfull[1],6)}')

#performing adf test for gas price
adfull1=adfuller(cx_sampled['GasPrice'])
print(f'p value:{round(adfull1[1],6)}')

#QUESTION 13

#critical_values
adfull2=adfuller(cx_sampled['GasPrice'])
print('critical value')
for k,v in adfull2[4].items():
  print(f'{k}:{v}')

#critical values for electric price
adfull3=adfuller(cx_sampled['ElecPrice'])
print('critical value')
for k,v in adfull3[4].items():
  print(f'{k}:{v}')


#QUESTION 14 AND 15
#BUILDING THE MODEL
#reset the index to its original form
cx_reset=cx_sampled.reset_index()
cx_reset

#selecting the columns
modelling_data=cx_reset[['FullDate','ElecPrice']]
modelling_data.head()

#renaming the columns to ds and y
modelling_data=modelling_data.rename(columns={'FullDate':'ds','ElecPrice':'y'})
modelling_data.head()

#splitting the dataset int train and test data
train_data=modelling_data[:2757]
test_data=modelling_data[2757:]
#to check the data
print(f'train_data shape:{train_data.shape}')
print(f'test_data shape:{test_data.shape}')


#fitting the training data into the fbprophet model
from fbprophet import Prophet
model=Prophet()
model.fit(train_data)
predicted=model.predict(test_data)
predicted.head()

predicted[['ds','yhat','yhat_lower','yhat_upper','trend_lower','trend_upper','trend']].head()

predicted.shape

len(cx)

#forecasting perfromance measures
#from sklearn.metrics import mean_squared_error
#def mape(test_data,predicted):
  #mape=np.mean(np.abs(np.array(test_data['y'])-np.array(predicted['yhat'])/np.array(test_data['y'])))*100
  ##test_data, predicted = np.array(test_data), np.array(predicted)
  #return np.mean(np.abs((test_data - predicted) / actual)) * 100
  #rmse=np.sqrt(mean_squared_error(np.array(test_data['y']),np.array(predicted['yhat'])))
  #return f'mape value for the model is :{round(mape,2)} and rmse value for the model is :{round(rmse,2)}'

def mape(test_data, predicted): 
    test_data, predicted = np.array(test_data['y']), np.array(predicted['yhat'])
    return np.mean(np.abs((test_data['y'] - predicted['yhat']) / test_data['y'])) * 100

mape(test_data,predicted)



#QUESTION 16`
from fbprophet.plot import plot_yearly
plot_yearly(model)


#QUESTION 17 AND 18

cx_reset.head()


multi_model=cx_reset.rename(columns={'FullDate':'ds','Tmax':'add2','SysLoad':'add1','GasPrice':'add3','ElecPrice':'y'})
multi_model.head()


multi_model.shape

#splitting the data int train and test
train_multi=multi_model[:2757]
test_multi=multi_model[2757:]



#checking the values to see they were split appropiately
train_multi.shape
test_multi.shape

#creating the multivariaate model
modelMulti=Prophet()
modelMulti.add_regressor('add1')
modelMulti.add_regressor('add2')
modelMulti.add_regressor('add3')


modelMulti.fit(train_multi)


predictedMulti=modelMulti.predict(test_multi)


predictedMulti

def mape_2(test_multi,predictedMulti):
  mape2=np.mean(np.abs(np.array(test_multi['y'])-np.array(predictedMulti['yhat']))/np.array(test_multi['y']))*100
  rmse2 = np.sqrt(mean_squared_error(np.array(test_multi['y']),np.array(predictedMulti['yhat'])))
  return f'mape value for the MultiVariate model is: {round(mape2,2)} and rmse value for the MultiVariate model is: {round(rmse2,2)}'


mape_2(test_multi,predictedMulti)

#question 19
#visualizing trend and monthly components
from fbprophet.plot import plot_weekly
plot_weekly(modelMulti)



#QUESTION 20


multimodel_2 = cx_reset.rename(columns = {'FullDate':'ds','ElecPrice':'y','SysLoad':'add1','GasPrice':'add2'})
multimodel_2.drop('Tmax', axis=1,inplace=True)

multimodel_2

#splitting into train and test
train_data3 = multimodel_2[:2757]
test_data3 = multimodel_2[2757:]


#checking to see if they were properly split
train_data3.shape
test_data3.shape


modelMulti2= Prophet()
modelMulti2.add_regressor('add1')
modelMulti2.add_regressor('add2')

modelMulti2.fit(train_data3)
 


predictedMulti3 = modelMulti2.predict(test_data3)
predictedMulti3



def mape_3(test_data3,predictedMulti3):
  mape3 = np.mean(np.abs(np.array(test_data3['y']) - np.array(predictedMulti3['yhat']))/ np.array(test_data3['y'])) *100
  return f'mape value for the model is: {round(mape3,2)}'

mape_3(test_data3,predictedMulti3)

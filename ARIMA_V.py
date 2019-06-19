import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import xlrd
from sklearn.metrics import mean_squared_error

train_percent = 0.7 #0.978
file = r'FINAL_EURUSD_Candlestick_1_D_BID_01.01.2009-01.01.2019.xls'

def dataset(file):
    excel_file = xlrd.open_workbook(file)
    first_page = excel_file.sheet_by_index(0)     # open first sheet
    actual_dataset = first_page.col_values(colx=slice(4, 5),start_rowx=1)  # these are all close values in arrays with a parent array
    data_a = [y for x in actual_dataset for y in x]  # makes everything into just one list!
    return data_a

def My_ARIMA(Real, P, D, Q):
    m = ARIMA(Real, order=(P, D, Q)) # IF USED WITHOUT D,Q THEN WILL LEAD TO AR
    m_fit = m.fit(disp=0)
    fit = m_fit.forecast()
    p = fit[0]
    return p

data = dataset(file)

#Number of datapoints
datapoints = len(data)

#70% = training, 30% = testing
bound = int(datapoints * train_percent)
test = data[bound:]
train = data[0:bound]

Predictions = list()
Real = train
#Real1 = [x for x in train]

for x in range(len(test)):

    prediction1 = My_ARIMA(Real, 1,0,0)
    real_val = test[x]
    print('Real: %f, Predicted: %f' % (real_val, prediction1))

    Predictions.append(prediction1)
    Real.append(real_val)

Error = mean_squared_error(test, Predictions)
print('Mean Squared Error %.7f' % Error)

Accuracy = trend(test, Predictions)
print('Accuracy %.5f'% Accuracy)

#Accuracy2 = f1_score(test, Predictions, average='macro')
#print('fscore: {}'.format(Accuracy2))


plt.plot(test, color='red')
plt.plot(Predictions, color='blue')
plt.legend(["Real", "Predicted"], loc='lower right')
plt.xlabel("Day")
plt.ylabel("Price")
plt.show()






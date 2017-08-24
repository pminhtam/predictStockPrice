import quandl
import numpy as np
import math
from sklearn import preprocessing, cross_validation,svm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# df = quandl.get("EOD/KO", authtoken="37T_cbaygisqktDcF7zZ",start_date = "2016-1-1")
data = pd.read_csv("D:/google_driver/Code/python/machine_learning_Web_Toturial/lay_du_lieu_internet/NASDAQ_yahoo.csv")
df = data[['High','Low','Volume','Adj Close']]
df = df.astype('float32')
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)
df = np.array(df)

scalerLabel = MinMaxScaler(feature_range=(0,1))
dataLabel = data[['Adj Close']]
dataLabel = scalerLabel.fit_transform(dataLabel)

train_size = int(len(df)*0.8)
test_size = len(df) - train_size
train,test = df[0:train_size,:],df[train_size:len(data),:]


def create_dataset(dataset,look_back = 1,predictNext = 0):
    dataX, dataY = [],[]
    for i in range(len(dataset)-look_back-predictNext):
        a = dataset[i:(i+look_back)]
        a = np.reshape(a, (a.shape[0]*a.shape[1]))
        dataX.append(a)
        dataY.append(dataset[i+look_back+predictNext,3])
    dataY = np.reshape(dataY, (len(dataY),1))
    return np.array(dataX),np.array(dataY)


look_back = 1
predictNext = 0
X_train,y_train = create_dataset(train,look_back,predictNext)
X_test,y_test = create_dataset(test,look_back,predictNext)


# LR = LinearRegression()
# LR.fit(X_train,y_train)
model = svm.SVR(kernel='poly')
model.fit(X_train,y_train)
accuracy = model.score(X_test,y_test)
print(accuracy)


# forecast_set = LR.predict(X_test)
# k= 0
# for i in range(len(y_test)-2):
#     if((y_test[i]-y_test[i+1])*(forecast_set[i]-forecast_set[i+1]))>0:
#         k+=1
#         print(y_test[i + 1], "     ", forecast_set[i + 1], "         Dung")
#         continue
#     print(y_test[i+1],"     ",forecast_set[i+1],"         Sai")
# print(k/(len(y_test)-1))
# print(len(y_test))

testPredict = model.predict(X_test)
trainPredict = model.predict(X_train)
trainPredict = scalerLabel.inverse_transform(trainPredict)
testPredict = scalerLabel.inverse_transform(testPredict)


#dataPlot = data[['Adj Close']]
dataPlot = scalerLabel.inverse_transform(dataLabel)

trainPlot = np.empty_like(dataPlot)
trainPlot[:,:] = np.nan
trainPlot[look_back+predictNext:train_size,:] = trainPredict



testPlot = np.empty_like(dataPlot)
testPlot[:,:] = np.nan
testPlot[train_size+look_back+predictNext:len(dataPlot),:] = testPredict



plt.plot(dataPlot)
plt.plot(trainPlot)
plt.plot(testPlot)
plt.legend(loc = 4)
plt.title('Linear')
plt.show()


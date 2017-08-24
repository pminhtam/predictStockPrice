import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error,accuracy_score
import math
np.random.seed(7)


column = ['High','Low','Volume','Adj Close']
label = 'Adj Close'


data = pd.read_csv("D:/google_driver/Code/python/machine_learning_Web_Toturial/lay_du_lieu_internet/NASDAQ_yahoo.csv")
df = data[column]
df = df.astype('float32')
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)
df = np.array(df)

scalerLabel = MinMaxScaler(feature_range=(0,1))
dataLabel = data[[label]]
dataLabel = scalerLabel.fit_transform(dataLabel)

train_size = int(len(df)*0.8)
test_size = len(df) - train_size
train,test = df[0:train_size,:],df[train_size:len(data),:]

def create_dataset(dataset,look_back = 1,predictNext = 0):
    dataX, dataY = [],[]
    for i in range(len(dataset)-look_back - predictNext):
        a = dataset[i:(i+look_back)]
        a = np.reshape(a, (a.shape[0]*a.shape[1]))
        dataX.append(a)
        dataY.append(dataset[i+look_back + predictNext,3])
    dataY = np.reshape(dataY, (len(dataY),1))
    return np.array(dataX),np.array(dataY)
look_back = 1
predictNext = 0
X_train,y_train = create_dataset(train,look_back,predictNext)
X_test,y_test = create_dataset(test,look_back,predictNext)

X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
# print(testX)
# print(testY)

print(X_train.shape)


#Tao mo hinh mang neural
model = Sequential()
model.add(LSTM(4,input_shape=(1,4*look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,y_train,epochs=2,batch_size=1,verbose=2)

trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)



#Chuyen du lieu tro lai trang thai ban dau
# tuc la dua du lieu thanh so hanh khach chu khong phai o dang tu (0,1) nhu truoc nua
trainPredict = scalerLabel.inverse_transform(trainPredict)
testPredict = scalerLabel.inverse_transform(testPredict)



##
# Tinh toan do chinh xac cua du doan

"""
trainY = scaler.inverse_transform(y_train)
testY = scaler.inverse_transform(y_test)
trainScore = math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
testScore = math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print('Train Score : %f'%(trainScore))
print('Test Score : %f'%(testScore))
"""

##
# Vi du lieu du doan la ngay hom sau nen phai dich chuyen lai 1 ngay de ve do thi

dataPlot = scalerLabel.inverse_transform(dataLabel)

trainPlot = np.empty_like(dataPlot)
trainPlot[:,:] = np.nan
trainPlot[look_back+predictNext:train_size,:] = trainPredict



testPlot = np.empty_like(dataPlot)
testPlot[:,:] = np.nan
testPlot[train_size+look_back+predictNext:len(dataPlot),:] = testPredict





plt.plot(dataPlot,color = '#418cbf')
plt.plot(trainPlot,color = 'y')
plt.plot(testPlot,color = '#ff7f0e')
plt.legend(loc = 4)
plt.title('LSTM')
plt.show()




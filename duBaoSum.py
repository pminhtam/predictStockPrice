import quandl
import pandas as pd
import math
from sklearn import cross_validation, svm
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# df = quandl.get("EOD/KO", authtoken="37T_cbaygisqktDcF7zZ",start_date = "2015-1-1")

def modelTwoNeural(X_train,y_train,look_back):
    model = Sequential()
    model.add(Dense(100,input_dim=4*look_back,activation = "tanh"))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=1, batch_size=10)
    return model

def modelLSTM(X_train,y_train,look_back):
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, 4 * look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=2, batch_size=1, verbose=2)
    return model
def modelLinearRegression(X_train,y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
def modelSVM(X_train,y_train):
    model = svm.SVR(kernel="poly")
    model.fit(X_train, y_train)
    return model

def prepareData(pathFile,column,label,look_back,predictNext):
    data = pd.read_csv(pathFile)
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
        for i in range(len(dataset)-look_back-predictNext):
            a = dataset[i:(i+look_back)]
            a = np.reshape(a, (a.shape[0]*a.shape[1]))
            dataX.append(a)
            dataY.append(dataset[i+look_back+predictNext,3])
        dataY = np.reshape(dataY, (len(dataY),1))
        return np.array(dataX),np.array(dataY)


    # look_back = 1
    # predictNext = 0
    X_train,y_train = create_dataset(train,look_back,predictNext)
    X_test,y_test = create_dataset(test,look_back,predictNext)

    X_lately = test[-look_back - predictNext:]
    return dataLabel,X_train,y_train,X_test,y_test,scalerLabel,train_size


def plotData(dataPlot,trainPlot,testPlot):
    plt.plot(dataPlot)
    plt.plot(trainPlot)
    plt.plot(testPlot)
    plt.legend(loc = 4)
    plt.title('Neural')
    plt.show()


def Predict(choosemodel,pathFile,column,label,look_back,predictNext,plot):
    dataLabel,X_train, y_train, X_test, y_test , scalerLabel,train_size = prepareData(pathFile,column,label,look_back,predictNext)

    if(choosemodel=="linear"):
        model = modelLinearRegression(X_train,y_train)
    elif(choosemodel=="twoneural"):
        model = modelTwoNeural(X_train,y_train,look_back)
    elif(choosemodel=="LSTM"):
        model = modelLSTM(X_train,y_train,look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    elif(choosemodel == "svm"):
        model = modelSVM(X_train,y_train)

    testPredict = model.predict(X_test)
    trainPredict = model.predict(X_train)

    trainPredict = np.reshape(trainPredict, (len(trainPredict),1))
    testPredict = np.reshape(testPredict, (len(testPredict),1))

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
    if(plot=="f"):
        return 1
    plotData(dataPlot,trainPlot,testPlot)


column = ['High','Low','Volume','Adj Close']
label = 'Adj Close'
pathFile = "D:/google_driver/Code/python/machine_learning_Web_Toturial/lay_du_lieu_internet/NASDAQ_yahoo.csv"
look_back = 1
predictNext = 0
choosemodel = sys.argv[1]
plot = sys.argv[2]
# choosemodel = "LSTM"
if(choosemodel == "help"):
    print("linear")
    print("twoneural")
    print("LSTM")
    print("svm")
else:
    Predict(choosemodel,pathFile,column,label,look_back,predictNext,plot)

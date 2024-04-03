# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
![image](https://github.com/Visalan-H/rnn-stock-price-prediction/assets/152077751/8929266c-31d2-4fc6-a34d-a57b4671ee56)

## Design Steps

### Step 1:
Import all necessary libraries and datasets 

### Step 2:
Do the necessary preprocessing and create the model using a single RNN Layer with 60 neurons and a dense layer for the output.
### Step 3:
Give the test set and predict using the model and plot it using matplotlib

## Program
#### Name:Visalan H
#### Register Number:212223240183

```python
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from keras import *
from keras.layers import *
import matplotlib.pyplot as plt

train=pd.read_csv('trainset.csv')
train.head()
train.columns
X=train.iloc[:,1:2].values
type(X)
X.shape

scaler=MinMaxScaler(feature_range=(0,1))
X1=scaler.fit_transform(X)
X1.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(X1[i-60:i,0])
  y_train_array.append(X1[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train1.shape
length = 60
n_features = 1

model=Sequential()
model.add(SimpleRNN(60,input_shape=(length,n_features)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mae')
model.fit(X_train1,y_train,epochs=100)
model.summary()

dataset_test=pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
dataset_test
test_set.shape
dataset_total = pd.concat((train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=scaler.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price_scaled)

print("Name:Visalan H Register Number:212223240183")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

## Output

### True Stock Price, Predicted Stock Price vs time
![Screenshot 2024-04-03 142707](https://github.com/Visalan-H/rnn-stock-price-prediction/assets/152077751/c5441244-878a-47ad-8349-4fda33baaf82)

### Mean Square Error

![image](https://github.com/Visalan-H/rnn-stock-price-prediction/assets/152077751/790ebb5b-55ee-4fb7-a027-418fce49129b)

## Result
Thus, A model to predict future stock prices is created successfully.

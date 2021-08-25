import numpy as np
from  numpy import sqrt
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = pd.read_csv(path, header=None)
x,y=df.values[:, :-1], df.values[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33)
n_features=x_train.shape[1]
model= Sequential()
model.add(tf.keras.Input(n_features,))
model.add(Dense(10, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
model.compile( loss='mse')
history=model.fit(x_train.astype('float32'),y_train,batch_size=32, epochs=500, verbose=2)
loss=model.evaluate(x_test.astype('float32'), y_test)
print('MSE : %.3f RMSE : %.3f'%(error, sqrt(error)))
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)

pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='loss')
pyplot.legend()
pyplot.show()
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = pd.read_csv(path, header=None)

x,y = df.values[:, :-1], df.values[:, -1]
y=LabelEncoder().fit_transform(y)
x_train, x_test, y_train,y_test= train_test_split(x,y, test_size=0.33)
n_features=x_train.shape[1]

model = Sequential()
model.add(tf.keras.Input(n_features,))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(x_train.astype('float32'), y_train,batch_size=32, epochs=150, verbose=2)
model.evaluate(x_test.astype('float32'),y_test)
row = [6.7,3.0,5.2,2.3]
yhat = model.predict([row])
print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='loss')
pyplot.plot(history.history['accuracy'], label='accuracy')
pyplot.legend()
pyplot.show()
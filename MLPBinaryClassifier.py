from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
df=pd.read_csv(path, header=None)
df.head()
x,y=df.values[:, :-1],df.values[:, -1]
x.astype('float32')

y=LabelEncoder().fit_transform(y)
x_train,x_tests,y_train,y_test=train_test_split(x,y,test_size=0.33)
n_features=x.shape[1]

model=Sequential()
model.add(tf.keras.Input(shape=(n_features,)))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history=model.fit(x_train.astype('float32'),y_train ,epochs=150, verbose=2)

loss,accuracy = model.evaluate(x_tests.astype('float32'),y_test)
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)

pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='loss')
pyplot.plot(history.history['accuracy'], label='accuracy')
pyplot.legend()
pyplot.show()
from tensorflow.keras.datasets.mnist import load_data
from matplotlib import pyplot
(trainX,trainY),(testX,testY)=load_data()
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
for i in range(25):
    pyplot.subplot(5,5,i+1)
    pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))

pyplot.show

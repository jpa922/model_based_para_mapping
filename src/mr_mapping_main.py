import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

use_batch_size = 20

X_train = np.arange(0,1,0.01)
y_train = [0 for ii in range(len(X_train))]
for ii in range(0,len(X_train)):
    if np.logical_and(X_train[ii]>0.2, X_train[ii]<0.6):
        y_train[ii] = (0,1)
    else:
        y_train[ii] = (1,0)

X_test = np.arange(0.005,1.005,0.01)
y_test = [0 for ii in range(len(X_test))]
for ii in range(0,len(X_test)):
    if np.logical_and(X_test[ii]>0.2, X_test[ii]<0.6):
        y_test[ii] = (0,1)
    else:
        y_test[ii] = (1,0)

X_train=X_train.reshape(X_train.shape[0],1)        #had tough time without this,
# y_train=y_train.reshape(y_train.shape[0],1)        #had tough time without this,

X_test=X_test.reshape(X_test.shape[0],1)        #had tough time without this,
# y_test=y_test.reshape(y_test.shape[0],1)        #had tough time without this,

model = Sequential()
model.add(Dense(1, 500, init='normal', activation='tanh'))

model.add(Dense(500, 500, init='normal', activation='tanh'))
# model.add(Dropout(0.5))
model.add(Dense(500, 500, init='uniform', activation='tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(50, 50, init='uniform', activation='tanh'))
# model.add(Dropout(0.5))

model.add(Dense(500, 2, init='normal', activation='softmax'))
# model.add(Activation('linear'))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, y_train, nb_epoch=5000, batch_size=use_batch_size)
# score = model.evaluate(X_test, y_test, batch_size=10)

y_predict = model.predict(X_test,batch_size=use_batch_size,verbose=1)

y_test_plot = y_test
y_predict_plot = y_predict

for ii in range(len(y_test_plot)):
    y_test_plot[ii]=y_test[ii][1]
    y_predict_plot[ii]=y_predict[ii][1]
plt.plot(X_test,y_test_plot, 'r')
plt.plot(X_test,y_predict, 'b.')
plt.show()

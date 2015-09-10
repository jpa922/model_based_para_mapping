import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import os
#import theano
os.chdir('C:/Users/pangjx/Desktop/nn_sg')
# load .mat data
# OK so X_data should be [n_samples, dimension1, dimension2, ...]
# y_data should be [n_samples, dimension1, dimension2, ...]

output = sio.loadmat('nn_sg_data/output.mat')
output = output['output']
#output = [[x,y] for x,y in zip(output[0,:],output[1,:])]
output = np.array(output)
output = output[1,:]
output=output.reshape(output.shape[0],1)        #had tough time without this

sgp_all = sio.loadmat('nn_sg_data/sgp_all.mat')
sgp_all = sgp_all['sgp_all']
sgp_all = np.transpose(sgp_all)
sgp_all -= np.mean(sgp_all)
sgp_all /= np.std(sgp_all)

sgp_reshape = sgp_all
sgp_reshape = sgp_reshape.reshape([6666,32,320])
sgp_reshape = np.swapaxes(sgp_reshape,1,2)
#%%
from keras.utils import np_utils
size_train = 1000

use_channel_start = 1
use_channel_end = 32

#X_train = sgp_all[0:size_train,(use_channel_start-1)*320+1:use_channel_end*320]
X_train = sgp_reshape[0:size_train,:,(use_channel_start-1)*320+1:use_channel_end*320]
y_train = np_utils.to_categorical(output[0:size_train,0],2)

#X_test = sgp_all[size_train:size_train+200,(use_channel_start-1)*320+1:use_channel_end*320]
X_test = sgp_reshape[size_train:size_train+200,:,(use_channel_start-1)*320+1:use_channel_end*320]

y_test = np_utils.to_categorical(output[size_train:size_train+200,0],2)

#%
#imgplot = plt.imshow(X_test)
#imgplot.set_cmap('gray')

#%%
use_batch_size = 10
model = Sequential()
#model.add(Dense(X_train.shape[1], 10000, init='lecun_uniform', activation='relu'))
model.add(Convolution2D(32, 32, 10, 1, border_mode='full')) 
model.add(Dropout(0.5))
#model.add(MaxPooling2D(poolsize = (2,2)))
model.add(Flatten())

model.add(Dense(32*32*10*1, 1000, init='lecun_uniform', activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1000, 2, init='lecun_uniform', activation='softmax'))
model.add(Dropout(0.5))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X_train, y_train, nb_epoch=100, batch_size=use_batch_size)
# score = model.evaluate(X_test, y_test, batch_size=10)

y_predict = model.predict(X_test,batch_size=use_batch_size,verbose=1)

plt.plot(y_test[0:,1], 'ro-')
plt.plot(y_predict[0:,1], 'b-')

plt.show()

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD

# load .mat data
# OK so X_data should be [n_samples, dimension1, dimension2, ...]
# y_data should be [n_samples, dimension1, dimension2, ...]

output = sio.loadmat('../data/output.mat')
output = output['output']
#output = [[x,y] for x,y in zip(output[0,:],output[1,:])]
output = np.array(output)
output = output[1,:]
output=output.reshape(output.shape[0],1)        #had tough time without this

sgp_all = sio.loadmat('../data/sgp_all.mat')
sgp_all = sgp_all['sgp_all']
sgp_all = np.transpose(sgp_all)
sgp_all -= np.mean(sgp_all)
sgp_all /= np.std(sgp_all)

#%%
from keras.utils import np_utils
size_train = 800

X_train = sgp_all[0:size_train,10*320+1:16*320]
y_train = np_utils.to_categorical(output[0:size_train,0],2)

X_test = sgp_all[size_train:1000,10*320+1:16*320]
y_test = np_utils.to_categorical(output[size_train:1000,0],2)

#%%
use_batch_size = 100
model = Sequential()
model.add(Dense(X_train.shape[1], 10000, init='lecun_uniform', activation='tanh'))

model.add(Dense(10000, 100, init='lecun_uniform', activation='tanh'))
model.add(Dropout(0.5))

#model.add(Dense(1000, 100, init='lecun_uniform', activation='tanh'))
#model.add(Dropout(0.5))

model.add(Dense(100, 2, init='lecun_uniform', activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X_train, y_train, nb_epoch=50, batch_size=use_batch_size)
# score = model.evaluate(X_test, y_test, batch_size=10)

y_predict = model.predict(X_test,batch_size=use_batch_size,verbose=1)

plt.plot(y_test[0:,1], 'ro-')
plt.plot(y_predict[0:,1], 'b.-')

plt.show()

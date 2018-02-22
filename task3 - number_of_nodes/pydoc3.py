from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt


batch_size = 128
num_classes = 10
epochs = 1


#you can edit this value to check multiple iterations
count =1
while (count<10):

	#you can edit this value to make it iterate for any multiple and check for eah step
	number_of_nodes = count*5

# the data, shuffled and split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()


	x_train = x_train.reshape(60000, 784)
	x_test = x_test.reshape(10000, 784)


	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	x_train /= 255
	x_test /= 255

	#print(x_train.shape[0], 'train samples') #not required to print everytime
	#print(x_test.shape[0], 'test samples')  #not required to print everytime

# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)



	model = Sequential()       
	model.add(Dense(number_of_nodes, activation='tanh', input_shape=(784,)))
#model.add(Dropout(0.2))		#not required at the moment
#model.add(Dense(512, activation='tanh'))  #not required 
#model.add(Dropout(0.2))		#not required at the moment
	model.add(Dense(num_classes, activation='softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

	history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0]*100)
	print('Test accuracy:', score[1]*100)
	count = count + 1
print ("done")

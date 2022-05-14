import os,codecs
import numpy as np
import matplotlib.pyplot as plt

# source for extracting mnist data - 
# https://github.com/Ghosh4AI/Data-Processors/blob/master/MNIST/MNIST_Loader.ipynb

def get_int(b):   # CONVERTS 4 BYTES TO A INT
    return int(codecs.encode(b, 'hex'), 16)

def read_data():
    datapath = 'mnist/'
    files = os.listdir(datapath)
    data_dict = {}
    for file in files:
        if file.endswith('ubyte'):
            print('Reading ',file)
            with open (datapath+file,'rb') as f:
                data = f.read()
                type = get_int(data[:4])
                length = get_int(data[4:8])
                if (type == 2051):
                    category = 'images'
                    num_rows = get_int(data[8:12])
                    num_cols = get_int(data[12:16])
                    parsed = np.frombuffer(data,dtype = np.uint8, offset = 16) 
                    parsed = parsed.reshape(length,num_rows,num_cols)          
                elif(type == 2049):
                    category = 'labels'
                    parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
                    parsed = parsed.reshape(length)                      
                if (length==10000):
                    set = 'test'
                elif (length==60000):
                    set = 'train'
                data_dict[set+'_'+category] = parsed
    return data_dict

data_dict = read_data()

X = data_dict["train_images"].reshape(60000,784)
Y = data_dict["train_labels"].reshape(60000,1)

from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from keras import initializers
from tensorflow.keras import optimizers
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Input
from keras.models import Model
from tensorflow import keras

X = (X/255) - 0.5
Y = (data_dict["test_images"].reshape(-1,784)/255) - 0.5

try:
    autoencoder = keras.models.load_model('q3auto')
except:
    layer0 = (Input(shape = (784,)))
    layer1 = Dense(512, input_dim = 784, trainable = True, activation = "relu", use_bias = True,
                    kernel_initializer="random_normal", bias_initializer="random_normal")(layer0)
    layer2 = Dense(128, trainable = True, activation = "relu", use_bias = True,
                    kernel_initializer="random_normal", bias_initializer="random_normal")(layer1)
    layer3 = Dense(64, trainable = True, activation = "relu", use_bias = True,
                    kernel_initializer="random_normal", bias_initializer="random_normal")(layer2)
    layer4 = Dense(128, trainable = True, activation = "relu", use_bias = True,
                    kernel_initializer="random_normal", bias_initializer="random_normal")(layer3)
    layer5 = Dense(512, trainable = True, activation = "relu", use_bias = True,
                    kernel_initializer="random_normal", bias_initializer="random_normal")(layer4)
    layer6 = Dense(784, trainable = True, activation = "relu", use_bias = True,
                    kernel_initializer="random_normal", bias_initializer="random_normal")(layer5)


    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    autoencoder = Model(layer0, layer6)

    autoencoder.compile(
    optimizer=adam,
    loss='mse',
    )

    autoencoder.summary()

    print("\n\n\n")

    history = autoencoder.fit(
    X,
    X,
    epochs=10,
    batch_size=600,
    shuffle = True
    )

    plt.plot(history.history['loss'], label='train')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training loss")
    plt.legend()
    plt.show()

#######################################
# Converting decoder to classifier

try:
    classifier = keras.models.load_model('q3class')
except:
    layer1.trainable = False
    layer2.trainable = False
    layer3.trainable = False

    layerA = Dense(32, trainable = True, activation = "relu", use_bias = True,
                    kernel_initializer="random_normal", bias_initializer="random_normal")(layer3)
    layerB = Dense(10, trainable = True, activation = "softmax", use_bias = True,
                    kernel_initializer="random_normal", bias_initializer="random_normal")(layerA)

    classifier = Model(layer0, layerB)

    classifier.compile(optimizer=adam, 
                loss='categorical_crossentropy', 
                metrics=['accuracy'],)

    classifier.summary()

    print("\n\n\n")

    # print(X.shape)

    history = classifier.fit(
    X,
    to_categorical(data_dict["train_labels"]),
    epochs=10,
    batch_size=600,
    shuffle = True,
    )

    print("\n\n\n")

    plt.plot(history.history['loss'], label='train')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training loss")
    plt.legend()
    plt.show()

print("\n\n\n")

scores = classifier.evaluate(
  Y,
  to_categorical(data_dict["test_labels"])
)

print('Test accuracy for this model is ', scores[1]*100, '%', sep = '')

predictions = classifier.predict(Y)
predictionsOfFFNN = np.argmax(predictions, axis=1)
classCount = [0,0,0,0,0,0,0,0,0,0]
predictionClassCount = [0,0,0,0,0,0,0,0,0,0]

for i in range(len(data_dict["test_labels"])):
    classCount[data_dict["test_labels"][i]] += 1
    if(data_dict["test_labels"][i] == predictionsOfFFNN[i]):
        predictionClassCount[data_dict["test_labels"][i]] += 1

for i in range(10):
    print("Accuracy of class ", i, " is ", predictionClassCount[i]*100/classCount[i], "%", sep = "")

autoencoder.save("q3auto")
classifier.save("q3class")


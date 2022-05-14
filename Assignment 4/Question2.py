import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras

df_train = pd.read_csv("fminst/fashion-mnist_train.csv")
df_test = pd.read_csv("fminst/fashion-mnist_test.csv")
label = {0:"T-shirt/top",
1 :"Trouser",
2 :"Pullover",
3 :"Dress",
4 :"Coat",
5 :"Sandal",
6 :"Shirt",
7 :"Sneaker",
8 :"Bag",
9 :"Ankle boot"}

trainData = np.array(df_train.iloc[:,1:])
trainLabels = np.array(df_train.iloc[:,0])
testData = np.array(df_test.iloc[:,1:])
testLabels = np.array(df_test.iloc[:,0])

trainData = (trainData/255) - 0.5
testData = (testData/255) - 0.5

try:
    model = keras.models.load_model('q2')
except:
    model = Sequential() #FFNN
    model.add(Dense(64, input_dim = 784, trainable = True, activation = "relu", use_bias = True,
                    kernel_initializer="random_normal", bias_initializer="random_normal")) #layer 1

    model.add(Dense(64, activation = "relu", kernel_initializer="random_normal", 
                    bias_initializer="random_normal")) #layer 2

    model.add(Dense(10, activation = "softmax", kernel_initializer="random_normal", 
                    bias_initializer="random_normal")) #layer 3

    sgd = optimizers.SGD(lr = 0.01, momentum = 0.9) #sgd optimizer

    model.compile(optimizer=sgd, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'],)

    model.summary()

    print("\n\n\n")

    history = model.fit(
      trainData,
      to_categorical(trainLabels),
      epochs=50,
      batch_size=600,
      shuffle = True,
    )

    plt.plot(history.history['loss'], label='train')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training loss")
    plt.legend()
    plt.show()

    print("\n\nBatch size =", 600)
    print("Epochs =", 50)
    print("learning rate = ", 0.01)
    print("momentum =", 0.9)
    print("\n\n\n")

scores = model.evaluate(
  testData,
  to_categorical(testLabels)
)

print('Test accuracy for this model is ', scores[1]*100, '%', sep = '')


predictions = model.predict(testData)
predictionsOfFFNN = np.argmax(predictions, axis=1)
classCount = [0,0,0,0,0,0,0,0,0,0]
predictionClassCount = [0,0,0,0,0,0,0,0,0,0]

for i in range(len(testLabels)):
    classCount[testLabels[i]] += 1
    if(testLabels[i] == predictionsOfFFNN[i]):
        predictionClassCount[testLabels[i]] += 1

for i in range(10):
    print("Accuracy of class ", i, " is ", predictionClassCount[i]*100/classCount[i], "%", sep = "")

model.save("q2")

####################################################

# # Using advanced concepts taught in class
# import pandas as pd
# import numpy as np

# df_train = pd.read_csv("fminst/fashion-mnist_train.csv")
# df_test = pd.read_csv("fminst/fashion-mnist_test.csv")
# label = {0:"T-shirt/top",
# 1 :"Trouser",
# 2 :"Pullover",
# 3 :"Dress",
# 4 :"Coat",
# 5 :"Sandal",
# 6 :"Shirt",
# 7 :"Sneaker",
# 8 :"Bag",
# 9 :"Ankle boot"}

# trainData = np.array(df_train.iloc[:,1:])
# trainLabels = np.array(df_train.iloc[:,0])
# valData = trainData[50000:]
# valLabels = trainLabels[50000:]
# trainData = trainData[:50000]
# trainLabels = trainLabels[:50000] #incase of validation
# testData = np.array(df_test.iloc[:,1:])
# testLabels = np.array(df_test.iloc[:,0])



# from keras.layers import Dropout
# from keras.callbacks import EarlyStopping
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from tensorflow.keras.utils import to_categorical
# from keras import initializers
# from tensorflow.keras import optimizers
# from matplotlib import pyplot
# from keras.callbacks import ModelCheckpoint
# from keras.models import load_model
# from tensorflow.keras.layers import BatchNormalization

# trainData = (trainData/255) - 0.5
# testData = (testData/255) - 0.5
# valData = (valData/255) - 0.5 #incase of validation

# model = Sequential() #FFNN
# model.add(Dense(200, input_dim = 784, trainable = True, activation = "relu", use_bias = True,
#                 kernel_initializer="random_normal", bias_initializer="random_normal")) #layer 1
# # model.add(Dropout(0.5))
# model.add(BatchNormalization())

# model.add(Dense(50, activation = "relu", kernel_initializer="random_normal", 
#                 bias_initializer="random_normal")) #layer 2
# # model.add(BatchNormalization())

# model.add(Dense(10, activation = "softmax", kernel_initializer="random_normal", 
#                 bias_initializer="random_normal")) #layer 3

# sgd = optimizers.SGD(lr = 0.01, momentum = 0.9) #sgd optimizer

# model.compile(optimizer=sgd, 
#               loss='categorical_crossentropy', 
#               metrics=['accuracy'],)

# model.summary()



# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 5)
# mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', 
#                      verbose=1, save_best_only=True)

# history = model.fit(
#   trainData,
#   to_categorical(trainLabels),
#   validation_data = (valData, to_categorical(valLabels)),
#   epochs = 50,
#   batch_size=600,
#   shuffle = True,
#   callbacks = [es, mc],
# )

# best_model = load_model("best_model.h5")
# scores = best_model.evaluate(
#   testData,
#   to_categorical(testLabels)
# )

# print('Test accuracy:', scores[1])
import os,codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import random

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

bag_size = 3
predictions = []
actual_prediction = []

training_sets = [[],[],[]]
testing_sets = [[],[],[]]

try:
    predictions = __import__("pickle").load(open("q4", "rb"), encoding = "bytes")
except:
    for i in range(bag_size):
        for j in range(60000):
            random_number = random.randint(0,59999)
            training_sets[i].append(X[random_number])
            testing_sets[i].append(data_dict["train_labels"][random_number])

    for i in range(bag_size):
        print("training bag number", i+1)
        clf = DecisionTreeClassifier(criterion="gini")
        clf.fit(training_sets[i], testing_sets[i])
        predictions.append(clf.predict(data_dict["test_images"].reshape(10000, 784)))

for x in range(10000): #majority voting
    if (predictions[0][x] == predictions[1][x]):
        actual_prediction.append(predictions[0][x])
    elif (predictions[1][x] == predictions[2][x]):
        actual_prediction.append(predictions[1][x])
    elif (predictions[2][x] == predictions[0][x]):
        actual_prediction.append(predictions[2][x])
    else:
        actual_prediction.append(-1)

countClasswise = [0,0,0,0,0,0,0,0,0,0]
count = 0
testCountClasswise = [0,0,0,0,0,0,0,0,0,0]

print()

for i in range(10000):
    testCountClasswise[data_dict["test_labels"][i]] += 1

# print("testcountclasswise", testCountClasswise)

for i in range(10000):
    if(actual_prediction[i] == data_dict["test_labels"][i]):
        count += 1
        countClasswise[actual_prediction[i]] += 1

print("\nAccuracy", count/100)

for i in range(10):
    print("Class",i,"accuracy is", countClasswise[i]*100/testCountClasswise[i])
    
__import__("pickle").dump(predictions, open("q4", "wb"))
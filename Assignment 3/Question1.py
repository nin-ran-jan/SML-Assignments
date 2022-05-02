from matplotlib import pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# b'batch_label'
# b'labels'
# b'data'
# b'filenames'

dataset1 = unpickle("cifar-10-batches-py\data_batch_1") #reading the first dataset
dataset2 = unpickle("cifar-10-batches-py\data_batch_2")
dataset3 = unpickle("cifar-10-batches-py\data_batch_3")
dataset4 = unpickle("cifar-10-batches-py\data_batch_4")
dataset5 = unpickle("cifar-10-batches-py\data_batch_5")
testDataset = unpickle("cifar-10-batches-py\\test_batch") #reading test dataset
labelNames = unpickle("cifar-10-batches-py\\batches.meta") #reading label name data

trainDataset = np.append(dataset1[b'data'], dataset2[b'data'], axis = 0) #training data
trainDataset = np.append(trainDataset, dataset3[b'data'], axis = 0)
trainDataset = np.append(trainDataset, dataset4[b'data'], axis = 0)
trainDataset = np.append(trainDataset, dataset5[b'data'], axis = 0)

trainDataLabels = np.append(dataset1[b'labels'], dataset2[b'labels'], axis = 0) #training labels
trainDataLabels = np.append(trainDataLabels, dataset3[b'labels'], axis = 0)
trainDataLabels = np.append(trainDataLabels, dataset4[b'labels'], axis = 0)
trainDataLabels = np.append(trainDataLabels, dataset5[b'labels'], axis = 0)

count = 0
classCount = 0

while classCount < 10:
    for i in range(10000):
        if(dataset1[b'labels'][i] == classCount):
            plt.subplot(1,5,(count+1))
            test = dataset1[b'data'][i].reshape(32,32,3, order = 'F')
            test[:,:,0] = test[:,:,0].T
            test[:,:,1] = test[:,:,1].T
            test[:,:,2] = test[:,:,2].T
            plt.imshow(test)
            count += 1
        if (count == 3):
            plt.title(labelNames[b'label_names'][classCount].decode("utf-8"))
        if(count >= 5):
            plt.show()
            classCount += 1
            count = 0
            break

clf = LinearDiscriminantAnalysis()
clf.fit(trainDataset,trainDataLabels)
prediction = clf.predict(testDataset[b'data']) #uses LDA from sklearn
#credit -> https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis 

correctCount = 0
classwiseMappings = [0,0,0,0,0,0,0,0,0,0]
for j in range(10000):
    if (testDataset[b'labels'][j] == prediction[j]):
        classwiseMappings[testDataset[b'labels'][j]] += 1/10
        correctCount += 1
print("Total accurarcy is ", correctCount/100,"%", sep = "")
print("Classwise accuracy is ",classwiseMappings ," (data given in percentages as per class)", sep = "")
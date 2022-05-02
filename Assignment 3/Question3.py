import pandas as pd
import numpy as np
import numpy.linalg as linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

Si = []
Mui = []
trainSortedClasswise = []

for i in range (10):
    Si.append(np.zeros((784,784)))
    Mui.append(np.zeros((784,1)))
    trainSortedClasswise.append([])

for i in range(60000):
    if(trainLabels[i] == 0):
        trainSortedClasswise[0].append(trainData[i].reshape(784,1))
    if(trainLabels[i] == 1):
        trainSortedClasswise[1].append(trainData[i].reshape(784,1))
    if(trainLabels[i] == 2):
        trainSortedClasswise[2].append(trainData[i].reshape(784,1))
    if(trainLabels[i] == 3):
        trainSortedClasswise[3].append(trainData[i].reshape(784,1))
    if(trainLabels[i] == 4):
        trainSortedClasswise[4].append(trainData[i].reshape(784,1))
    if(trainLabels[i] == 5):
        trainSortedClasswise[5].append(trainData[i].reshape(784,1))
    if(trainLabels[i] == 6):
        trainSortedClasswise[6].append(trainData[i].reshape(784,1))
    if(trainLabels[i] == 7):
        trainSortedClasswise[7].append(trainData[i].reshape(784,1))
    if(trainLabels[i] == 8):
        trainSortedClasswise[8].append(trainData[i].reshape(784,1))
    if(trainLabels[i] == 9):
        trainSortedClasswise[9].append(trainData[i].reshape(784,1))

trainSortedClasswise = np.array(trainSortedClasswise)

for i in range(10):
    Mui[i] = np.average(trainSortedClasswise[i], axis = 0)

for i in range(10):
    for j in range(6000):
        Si[i] += (np.matmul(trainSortedClasswise[i,j,:] - Mui[i], (trainSortedClasswise[i,j,:] - Mui[i]).T))

Sw = np.zeros((784,784))
Sb = np.zeros((784,784))
Mu = np.zeros((784,1))

for i in range(10):
    Sw += Si[i]
    Mu += np.multiply(Mui[i], 1/10)

for i in range(10):
    Sb += np.multiply(np.matmul(Mui[i] - Mu, (Mui[i] - Mu).T), 6000)

# print(Sw, Sb, sep = "\n\n\n")

SwInv = linalg.inv(Sw)

w,v = np.linalg.eig(np.matmul(SwInv,Sb))
idx = w.argsort()[::-1] 
w = w[idx]
u = v[:,idx]
W = u[:,:9].real #c-1 classes eigenvalues

Y = np.matmul(W.T, trainData.T)

testData = np.array(df_test.iloc[:,1:])
testLabels = np.array(df_test.iloc[:,0])

clf = LinearDiscriminantAnalysis()
clf.fit(Y.T,trainLabels)
prediction = clf.predict(np.matmul(W.T, testData.T).T) #uses LDA from sklearn
#credit -> https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis 

correctCount = 0
classwiseMappings = [0,0,0,0,0,0,0,0,0,0]
classWiseCount = [0] * 10
for j in range(10000):
    if (testLabels[j] == prediction[j]):
        classwiseMappings[prediction[j]] += 1
        correctCount += 1
    classWiseCount[testLabels[j]] += 1
print("Total accurarcy is ", correctCount/100,"%", sep = "")

for i in range(10):
    print(f"Class {i} accuracy: {classwiseMappings[i]*100/classWiseCount[i]}%")
print("Classwise accuracy is ",classwiseMappings ," (data given in frequencies as per class)", sep = "")
# print(W,Y,sep = "\n\n\n\n")
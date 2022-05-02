import os,codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import numpy.linalg as linalg

#source for extracting mnist data - 
# https://github.com/Ghosh4AI/Data-Processors/blob/master/MNIST/MNIST_Loader.ipynb


def get_int(b):   # CONVERTS 4 BYTES TO A INT
    return int(codecs.encode(b, 'hex'), 16)

def PCACalc(trainData, testData, components):
    pca = PCA(n_components=components) 
    pca.fit(trainData)
    #uses PCA from sklearn
    return pca.transform(trainData), pca.transform(testData)

def main():
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
    #dict_keys(['test_images', 'test_labels', 'train_images', 'train_labels'])
    (X, Y) = PCACalc(data_dict["train_images"].reshape(60000,784), data_dict["test_images"].reshape(10000,784), 15)

    Si = []
    Mui = []
    trainSortedClasswise = []

    for i in range (10):
        Si.append(np.zeros((15,15)))
        Mui.append(np.zeros((15,1)))
        trainSortedClasswise.append([])
    
    trainLabels = data_dict["train_labels"]
    trainLabels = np.array(trainLabels).reshape(60000,1)

    for i in range(60000):
        if(trainLabels[i] == 0):
            trainSortedClasswise[0].append(X[i].reshape(15,1))
        if(trainLabels[i] == 1):
            trainSortedClasswise[1].append(X[i].reshape(15,1))
        if(trainLabels[i] == 2):
            trainSortedClasswise[2].append(X[i].reshape(15,1))
        if(trainLabels[i] == 3):
            trainSortedClasswise[3].append(X[i].reshape(15,1))
        if(trainLabels[i] == 4):
            trainSortedClasswise[4].append(X[i].reshape(15,1))
        if(trainLabels[i] == 5):
            trainSortedClasswise[5].append(X[i].reshape(15,1))
        if(trainLabels[i] == 6):
            trainSortedClasswise[6].append(X[i].reshape(15,1))
        if(trainLabels[i] == 7):
            trainSortedClasswise[7].append(X[i].reshape(15,1))
        if(trainLabels[i] == 8):
            trainSortedClasswise[8].append(X[i].reshape(15,1))
        if(trainLabels[i] == 9):
            trainSortedClasswise[9].append(X[i].reshape(15,1))

    trainSortedClasswise = np.array(trainSortedClasswise)
    # print(trainSortedClasswise[7])

    for i in range(10):
        Mui[i] = np.average(trainSortedClasswise[i], axis = 0)

    # print(Mui[0].shape)

    for i in range(10):
        for j in range(len(trainSortedClasswise[i])):
            Si[i] += (np.matmul(trainSortedClasswise[i][j] - Mui[i], (trainSortedClasswise[i][j] - Mui[i]).T))

    # print(Si[0].shape)
    Sw = np.zeros((15,15))
    Sb = np.zeros((15,15))
    Mu = np.zeros((15,1))

    for i in range(10):
        Sw += Si[i]
        Mu += np.multiply(Mui[i], len(trainSortedClasswise[i])/60000)

    for i in range(10):
        Sb += np.multiply(np.matmul(Mui[i] - Mu, (Mui[i] - Mu).T), len(trainSortedClasswise[i]))

    # print(Sw, Sb, sep = "\n\n\n")

    SwInv = linalg.pinv(Sw)

    w,v = np.linalg.eig(np.matmul(SwInv,Sb))
    idx = w.argsort()[::-1] 
    w = w[idx]
    u = v[:,idx]
    W = u[:,:9].real

    y = np.matmul(W.T, X.T)

    # print(y.shape)

    clf = LinearDiscriminantAnalysis()
    clf.fit(y.T,trainLabels)
    prediction = clf.predict(np.matmul(W.T, Y.T).T) #uses LDA from sklearn
    #credit -> https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis 

    correctCount = 0
    classwiseMappings = [0,0,0,0,0,0,0,0,0,0]
    classWiseCount = [0] * 10
    for j in range(10000):
        if (data_dict["test_labels"][j] == prediction[j]):
            classwiseMappings[prediction[j]] += 1
            correctCount += 1
        classWiseCount[data_dict["test_labels"][j]] += 1
    print("Total accurarcy is ", correctCount/100,"%", sep = "")
    for i in range(10):
        print(f"Class {i} accuracy: {classwiseMappings[i]*100/classWiseCount[i]}%")
    print("Classwise accuracy is ",classwiseMappings ," (data given in frequencies as per class)", sep = "")
if __name__ == "__main__":
    main()
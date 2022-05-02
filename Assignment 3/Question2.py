import os,codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

#source for extracting mnist data - 
# https://github.com/Ghosh4AI/Data-Processors/blob/master/MNIST/MNIST_Loader.ipynb


def get_int(b):   # CONVERTS 4 BYTES TO A INT
    return int(codecs.encode(b, 'hex'), 16)

def PCALDACalc(trainData, testData, testLabels, components):
    pca = PCA(n_components=components) 
    pca.fit(trainData)
    Y = pca.transform(testData) #uses PCA from sklearn

    clf = LinearDiscriminantAnalysis()
    clf.fit(Y, testLabels)
    prediction = clf.predict(Y) #uses LDA from sklearn

    print()

    correctCount = 0
    classwiseMappings = [0,0,0,0,0,0,0,0,0,0]
    for j in range(10000):
        if (testLabels[j] == prediction[j]):
            classwiseMappings[testLabels[j]] += 1
            correctCount += 1
    print("Total accurarcy is ", correctCount/100,"%", sep = "")
    print("Classwise accuracy is ",classwiseMappings ," (data given in frequency as per class)", sep = "")
    #credit -> https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis 

    return correctCount/100

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
    acc15 = PCALDACalc(data_dict["train_images"].reshape(60000,784), data_dict["test_images"].reshape(10000,784), data_dict["test_labels"], 15)
    acc8 = PCALDACalc(data_dict["train_images"].reshape(60000,784), data_dict["test_images"].reshape(10000,784), data_dict["test_labels"], 8)
    acc3 = PCALDACalc(data_dict["train_images"].reshape(60000,784), data_dict["test_images"].reshape(10000,784), data_dict["test_labels"], 3)

    plt.plot([15,8,3],[acc15, acc8, acc3])
    plt.xlabel("Number of components")
    plt.ylabel("Total accuracy in percentage")
    plt.grid(True)
    plt.show()

    #Observation - as the number of components increase, the accuracy of the model increases as well.
    #Hence, the experiment with n_components = 15 has the maximum accuracy and the experiment with n_components = 3 has minimum accuracy.


if __name__ == "__main__":
    main()




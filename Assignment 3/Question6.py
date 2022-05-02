import numpy as np

trainData = np.random.rand(1000,1)
trainDataLabel = []
# print(trainData)

c1 = []
c2 = []

for i in range(1000):
    if trainData[i] >= 0.5:
        c1.append(trainData[i][0])
        trainDataLabel.append(1)
    else:
        c2.append(trainData[i][0])
        trainDataLabel.append(2)

# print(c1, c2, sep = "\n\n\n\n")

print(trainData)

b00 = 0.5
b01 = 0.5
b10 = 0.5
b11 = 0.5

def sigmoid(x):
    return 1/(1+np.exp**x)

def sign(x):
    if x >= 0.5: #temp sign function
        return 1
    else:
        return -1

alpha = 0.01

def b11diff(x, i):
    return -trainDataLabel[i]*sigmoid(b01*x + b00)

def b10diff(x, i):
    return -trainDataLabel[i]

def b01diff(x, i):
    return -trainDataLabel[i]*b11*x*sigmoid(b01*x+b00)*(1-sigmoid(b01*x + b00))

def b00diff(x, i):
    return -trainDataLabel[i]*b11*sigmoid(b10*x + b00)(1-sigmoid(b01*x + b00))

def updateb00(i):
    b00 -= b00diff(trainData[i], i)*alpha

def updateb01(i):
    b01 -= b01diff(trainData[i], i)*alpha

def updateb10(i):
    b10 -= b10diff(trainData[i], i)*alpha

def updateb11(i):
    b11 -= b11diff(trainData[i], i)*alpha

trueY = 1 #error

for i in range(1000):
    if(trainDataLabel != trueY):
        updateb00(i)
        updateb01(i)
        updateb10(i)
        updateb11(i)
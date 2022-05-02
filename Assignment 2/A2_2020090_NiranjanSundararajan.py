from cmath import log
from scipy.stats import multivariate_normal as mn
from scipy.stats import bernoulli as bn
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sd


def q3PartDOnwards():
    d = 10 
    N = 10000
    mu_given = np.random.rand(1,d)
    mu_given = list(mu_given[0])
    cov = sd.make_spd_matrix(d) #positive definite
    x = mn.rvs(mu_given, cov, N).T
    print("Data matrix X =\n",x, sep = "")

    mu = np.average(x, axis = 1)
    print("Mean =\n",mu.reshape(d,1), sep = "")

    xc = x - mu.reshape(d,1)
    print("Xc =\n",xc, sep = "")

    Sxc = np.zeros((d,d))
    for i in range(N):
        Sxc += (np.matmul(np.array([xc[:,i]]).reshape(d,1), xc[:,i].reshape(1,d)))
    Sxc /= N
    print("Sxc =\n",Sxc, sep = "")

    w,v = np.linalg.eig(Sxc)
    idx = w.argsort()[::-1]   
    w = w[idx]
    u = v[:,idx]
    print("PCA Matrix = U =\n",u, sep = "")

    y = np.matmul(u.T, xc)
    print("Y =\n",y, sep = "")

    MSE = 0
    Xrev = (np.matmul(u, y) + mu.reshape(d,1))
    print(Xrev)
    for i in range(d):
        for j in range(N):
            MSE += ((Xrev[i][j] - x[i][j])**2)
    MSE /= (d*N)
    print("MSE =",MSE)

    MSEps = []

    for p in range(d):
        yp = np.matmul(u[:,:p+1].T, xc)
        MSEp = 0
        Xrevp = (np.matmul(u[:,:p+1], yp) + mu.reshape(d,1))
        for i in range(d):
            for j in range(N):
                MSEp += ((Xrevp[i][j] - x[i][j])**2)
        MSEps.append(MSEp/(d*N))

    x = np.arange(1, d+1, 1)
    plt.plot(x, MSEps, label = "MSEp v/s p")
    plt.grid(True)
    plt.legend()
    plt.xlabel("p")
    plt.ylabel("MSEp")
    plt.show()


def q3PartC():
    x = np.array([[2.,0.],[0.,2.]])
    print("X =\n",x, sep = "")

    mu = np.average(x, axis = 1)
    print("Mean =\n",mu.reshape(2,1), sep = "")

    xc = x - mu.reshape(2,1)
    print("Xc =\n",xc, sep = "")

    Sxc = np.array([[0.,0.],[0.,0.]])

    for i in range(len(xc)):
        Sxc += (np.matmul(np.array([xc[:,i]]).reshape(2,1), xc[:,i].reshape(1,2)))
    Sxc *= 0.5
    print("Sxc =\n",Sxc, sep = "")

    w,v = np.linalg.eig(Sxc)
    idx = w.argsort()[::-1]   
    w = w[idx]
    u = v[:,idx]
    print("PCA Matrix = U =\n",u, sep = "")

    y = np.matmul(u.T, xc)
    print("Y =\n",y, sep = "")

    MSE = 0
    Xrev = (np.matmul(u, y) + mu.reshape(2,1))
    for i in range(2):
        for j in range(2):
            MSE += ((Xrev[i][j] - x[i][j])**2)/4
    print("MSE =",MSE)

def q1():
    mu1 = np.array([0.5, 0.8])
    mu2 = np.array([0.9,0.2])

    x1a = bn.rvs(mu1[0], size = 100)
    x1b = bn.rvs(mu1[1], size = 100)
    x1 = np.vstack((x1a, x1b))
    # print(x1)

    x2a = bn.rvs(mu2[0], size = 100)
    x2b = bn.rvs(mu2[1], size = 100)
    x2 = np.vstack((x2a, x2b))
    # print(x2)

    x1Train = x1[:,:50]
    x2Train = x2[:,:50]
    x1Test = x1[:,50:]
    x2Test = x2[:,50:]
    # print(x1Test)

    #Using MLE
    def MLE(xTrain, color1, color2, num):
        theta1 = (1/50)*sum(xTrain[0])
        theta2 = (1/50)*sum(xTrain[1])
        print("Final MLE for training set of class " + num + " is:", theta1, theta2)

        theta1temp = []
        theta2temp = []

        for i in range(1,51,1):
            theta1temp.append((1/i)*sum(xTrain[0,:i]))
            theta2temp.append((1/i)*sum(xTrain[1,:i]))

        plt.plot(range(1,51,1), theta1temp, label = "theta"+num+"1 v/s n", color = color1)
        plt.plot(range(1,51,1), theta2temp, label = "theta"+num+"2 v/s n", color = color2)
        plt.legend()
        plt.grid(True)
        plt.xlabel("n")
        plt.ylabel("theta")
        plt.show()

    MLE(x1Train, "red", "blue", "1")
    MLE(x2Train, "green", "pink", "2")

    def plotTrain(xTrain, color1, color2, num):
        plt.subplots(1,2)
        plt.subplot(1,2,1)
        plt.scatter(range(1,51,1), xTrain[0], color = color1, label = "Samples of class " + num + " dimension 1")
        plt.grid(True)
        plt.legend()
        plt.ylabel("Sample")
        plt.subplot(1,2,2)
        plt.scatter(range(1,51,1), xTrain[1], color = color2, label = "Samples of class " + num + " dimension 2")
        plt.legend()
        plt.grid(True)
        plt.xlabel("n")
        plt.show()

    plotTrain(x1Train, "red", "blue", "1")
    plotTrain(x2Train, "green", "pink", "2")

    
    #part E, dichotomizer

    class1 = 0
    class2 = 0

    for i in range(50):
        if (x1Test[0,i]*np.log(16) - x1Test[1,i]*np.log(9) + np.log(5/4) >= 0):
            class1+=1
        if (x2Test[0,i]*np.log(16) - x2Test[1,i]*np.log(9) + np.log(5/4) >= 0):
            class2+=1

    print(class1, "objects correctly classified as class 1 and",class2,"objects correctly classified as class 2.")

def main():
    q1()
    print("\n\n\n\n\n")
    q3PartC()
    print("\n\n\n\n\n")
    q3PartDOnwards()
    print("\n\n\n\n\n")

if __name__ == "__main__":
    main()

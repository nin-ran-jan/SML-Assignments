import numpy as np
import matplotlib.pyplot as plt

def plot(xvals, yvals, functionName, colour, no):
    plt.subplot(2, 2, no)
    plt.plot(xvals, yvals, color = colour, label = functionName)
    return

def plotGaussian(xvals, mean, variance, functionName, colour, no):
    yvals = np.e**(-0.5*(xvals-mean)**2/variance)/(variance*2*np.pi)**0.5
    plot(xvals, yvals, functionName, colour, no)
    plot(xvals, yvals, functionName, colour, 4)
    return yvals

def plotLikelihood(xvals, Pxw1, Pxw2, functionName, colour, no):
    yvals = Pxw1/Pxw2
    plot(xvals, yvals, functionName, colour, no)
    plt.ylim(-0.1, 50.0)
    plot(xvals, yvals, functionName, colour, 4)
    plt.ylim(-0.1, 2.0)
    return

def main1():
    figure = plt.subplots(2,2)[0]
    figure.tight_layout(pad = 2.5)

    x = np.linspace(-3, 10, 130)
    Pxw1 = plotGaussian(x, 2, 1, "P(x|w1)", "blue", 1)
    Pxw2 = plotGaussian(x, 5, 1, "P(x|w2)", "red", 2)
    plotLikelihood(x, Pxw1, Pxw2, "P(x|w1)/P(x|w2)", "green", 3)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.xlabel("x", labelpad = 0)
        plt.grid()
        plt.legend()
    plt.show()
    return

def main3():
    x = np.linspace(-26, 34, 600)
    a1 = 3
    a2 = 5
    b = 1
    Pxw1 = (1/(np.pi*b))*(1/(1+((x-a1)/b)**2))
    Pxw2 = (1/(np.pi*b))*(1/(1+((x-a2)/b)**2))
    Pw1 = 0.5
    Pw2 = 0.5
    Px = Pxw1*Pw1 + Pxw2*Pw2
    Pw1x = (Pxw1*Pw1)/Px
    plt.plot(x, Pw1x, color = "red", label = "P(w1|x)")
    plt.xlabel("x")
    plt.grid()
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main1() #For question 1
    main3() #For question 3

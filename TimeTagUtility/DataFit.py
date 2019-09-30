import numpy as np
from scipy import optimize

class Parameter:
    def __init__(self, value):
            self.value = value

    def set(self, value):
            self.value = value

    def __call__(self):
            return self.value

def fit(function, parameters, y, x = None):
    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return y - function(x)

    if x is None: x = np.arange(y.shape[0])
    p = [param() for param in parameters]
    return optimize.leastsq(f, p)

def GaussianFunc(x,params):
    return params[2] * np.exp(-((x-params[0])/params[1])**2)


def GaussianFit(gauInitialParams, xData, yData):
    # giving initial parameters
    mu = Parameter(gauInitialParams[0])
    sigma = Parameter(gauInitialParams[1])
    height = Parameter(gauInitialParams[2])

    # define your function:
    def f(x): return height() * np.exp(-((x-mu())/sigma())**2)
    output = fit(f, [mu, sigma, height], yData, xData)
    print(output)
    return output

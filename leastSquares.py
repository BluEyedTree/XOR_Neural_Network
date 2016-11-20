import numpy as np
import lineSearchOptimize as lso



### The data
hours_tmp = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]
hours = np.matrix([[1,x] for x in hours_tmp])
passed_tmp = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
passed = np.matrix([passed_tmp]).T* 1.0

def logistic(t):
    return 1.0/(1.0 + np.exp(t))

numErrorEvals = 0
numGradEvals = 0

### Version 1 ###
def dlog(t):
    lt = logistic(t)
    return np.multiply(lt, (1-lt))
def errorFunc(beta):
    global numErrorEvals
    numErrorEvals+=1
    errorVec = passed - logistic(-hours*beta)
    return np.inner(errorVec.T, errorVec.T)
def baseDeriv(beta):
    return -np.multiply(2*(passed - logistic(-hours*beta)),dlog(-hours*beta))
def dError(beta):
    global numGradEvals
    numGradEvals+=1
    return hours.T*baseDeriv(beta)


startPoint = np.array([[0],[0]])
print("--------- GRADIENT DESCENT: Least Square ---------")
betas = lso.gradDesc(startPoint, 8, 10**(-4), 10**(-1), 10**(-12), errorFunc, dError, lambda x: -dError(x))
print("Computed in:              {c} iterations".format(c=lso.numIters))
print("Number of Line Searches:  {c}".format(c=lso.numLineSearch))
if (lso.numIters > 0):
    print("Line Searches/iterations: {c}".format(c=round(lso.numLineSearch/lso.numIters, 3)))
else:
    print("Line Searches/iterations: Infinite")
print("Error Function Evaluated: {ne} times".format(ne=numErrorEvals))
print("Gradient Evaluated:       {ge} times".format(ge=numGradEvals))
print("Beta = ({x}, {y})".format(x=betas[0,0],y=betas[1,0]))
print("---------------------- DONE ----------------------")
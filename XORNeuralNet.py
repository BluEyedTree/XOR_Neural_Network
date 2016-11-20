
import numpy as np
import matplotlib.pyplot as plt
import lineSearchOptimize as lso



## Some constants we will use to count the different function calls
logit_function_calls = 0
forwardPropCalls = 0
numGradEvals = 0

# Specific to this problem (with leading 1s in the x matrix)
x = np.matrix([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.matrix([[0], [1], [1], [0]])

# Where we'll store the logit function evaluations
#A00 = 0
#A01 = 0
#A0  = 0

## Setting an intial beta value for developement purposes
## we first set the seed so that the random numbers come out the same every time
np.random.seed(1234)
randomBetaVals = [4 * (np.random.random() - 0.5)  for tmp in range(9)]
initialBetas = np.matrix([randomBetaVals]).T

## The general logit function
def logit(arg):
	global logit_function_calls
	logit_function_calls += 1
	return 1.0 / (1 + np.exp(-arg))

## The whole forward propogation thing.
def yHat(beta):
	global forwardPropCalls
	forwardPropCalls += 1
	A00 = logit(x * beta[range(0,3),0])
	A01 = logit(x * beta[range(3,6),0])
	tmp = np.concatenate((A00, A01, np.matrix([[1,1,1,1]]).T), axis=1)
	A0 = logit(tmp * beta[range(6,9), 0])
	return A00, A01, A0

def xorMSE(beta):
	tmp = y - yHat(beta)[2]
	return (tmp.T * tmp)[0,0]

def xorGrad(beta):
	"""
	Just remember to return a numpy column matrix
	"""
	global numGradEvals
	numGradEvals += 1
	A00, A01, A0 = yHat(beta)
	da0  = -2 * np.multiply((y - A0), np.multiply(A0, (1 - A0)))
	db0  = np.multiply(da0, A00)
	db1  = np.multiply(da0, A01)
	da00 = np.multiply(db0, (1 - A00)) * beta[6,0]
	db00 = np.multiply(da00, x[:,0])
	db01 = np.multiply(da00, x[:,1])
	da10 = np.multiply(db1, (1 - A01)) * beta[7,0]
	db10 = np.multiply(da10, x[:,0])
	db11 = np.multiply(da10, x[:,1])
	gradVal = [db00, db01, da00, db10, db11, da10, db0, db1, da0]
	gradVal = np.matrix([[sum(arg)[0,0] for arg in gradVal]]).T
	return gradVal


print("----------------- XOR Neural Net -----------------")
## This code below will be executable once xorGrad is defined
smarterBetas = np.matrix([[1.0, 1.0, -0.5, 1.0, 1.0, -1.5, 1, -1, -1.1]]).T
beta = lso.gradDesc(smarterBetas, 64, 10**(-4), 10**(-1), 10**(-8), xorMSE, xorGrad, lambda arg: -xorGrad(arg))
plt.plot([-beta[2,0]/beta[1,0], -beta[0,0]/beta[1,0] - beta[2,0]/beta[1,0]])
plt.plot([-beta[5,0]/beta[4,0], -beta[3,0]/beta[4,0] - beta[5,0]/beta[4,0]])
plt.title("Final beta values")
plt.xlabel("x0")
plt.ylabel("x1")
plt.show()
print("Forward Prop Evaluated:     {ne} times".format(ne=forwardPropCalls))
print("Logit Function Calls:       {ne} times".format(ne=logit_function_calls))
print("Gradient Evaluated:         {ge} times".format(ge=numGradEvals))
print("Computed in:                {c} iterations".format(c=lso.numIters))
print("Number of Line Searches:    {c}".format(c=lso.numLineSearch))
if (lso.numIters > 0):
    print("Line Searches/iterations:   {c}".format(c=round(lso.numLineSearch/lso.numIters, 3)))
else:
    print("Line Searches/iterations:   Infinite")
np.set_printoptions(precision=3)
print("Beta = {x}".format(x=initialBetas.T[0]))
print("Total MSE for this Beta:    {ne} times".format(ne=xorMSE(beta)))
tmp = xorGrad(beta)
print("Gradient Norm:              {ne}".format(ne=(tmp.T*tmp)[0,0]))
print("---------------------- DONE ----------------------")
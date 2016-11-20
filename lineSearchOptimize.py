
import numpy as np
import matplotlib.pyplot as plt

"""
The main function in this file, `gradDesc`, is unfortunately named.
It is more general than a gradient descent function. It is a general
purpose optimizer with line search. But we'd already started collaborating
and it's not worth the cost to rename it.
"""

numLineSearch = 0
numIters = 0
beta_array = []

def lineSearch(alpha_max, phi, dphi, c_1, c_2):
	"""
	argument list
	-----------
	alpha_max => real number maximum for alpha
	phi       => should take a real number as argument,
					return the value of the minimizing function at x_0 + tp,
					where p is the descent direction
	dphi      => derivative of phi
	c_1, c_2  => constants for Wolfe condidtions
	"""

	phi0, dphi0 = phi(0), dphi(0)
	def zoom(alpha_lo, alpha_hi):
		while alpha_lo != alpha_hi:
			global numLineSearch
			numLineSearch += 1
			alpha_j = (alpha_hi + alpha_lo) / 2.
			a = phi(alpha_j)
			wolf_1 = (a <= phi0 + c_1*alpha_j*dphi0)
			if not (wolf_1) or (a >= phi(alpha_lo)):
				alpha_hi = alpha_j
			else:
				b = dphi(alpha_j)
				if abs(b) <= -c_2*dphi0:
					return alpha_j
				if b*(alpha_hi - alpha_lo) >= 0:
					alpha_hi = alpha_lo
				alpha_lo = alpha_j
		raise ValueError("The zoom while-loop has failed with alpha_lo = alpha_hi = {ah}, with {ni} iterations.".format(ah=alpha_lo, ni=numIters))
	alpha_0, alpha_1 = 0.0, (1.0 * alpha_max / 2)
	i = 1
	while True:
		global numLineSearch
		numLineSearch +=1
		a_1 = phi(alpha_1)
		wolf_1 = (a_1 <= phi0 + c_1*alpha_1*dphi0)
		if (not wolf_1) or (i > 1 and a_1 >= phi(alpha_0)):
			return zoom(alpha_0, alpha_1)
		b = dphi(alpha_1)
		if abs(b) <= - c_2*dphi0:
			return alpha_1
		if b >= 0:
			return zoom(alpha_1, alpha_0)
		alpha_0, alpha_1 = alpha_1, (alpha_1 + alpha_max)/2.0
		i = i + 1

def gradDesc(guess, alpha_max, c_1, c_2, tol, func, grad, pFunc):
	"""
	argument list
	-----------
	guess => column matrix guess as to the location of the minimizer
	alpha_max => real number maximum for alpha
	c_1, c_2  => constants for Wolfe condidtions
	tol       => The greatest acceptable value for the norm of the gradient
	func      => The function you want to minimize.
					Should only require arguments of the dimensions of the guess.
	grad      => Gradient of the above function
	pFunc     => Should take arguments like our guess and return a descent direction
	"""
	x = guess
	global numIters
	## This is unfortunate, but because the functions we'll be optimizing have arguments generally referred to as beta, it makes most sense to call this beta_array. This is unfortunate.
	global beta_array
	beta_array = [x]
	numIters = 0
	currentGradVal = grad(x)
	prevP = oldGradVal = 0*currentGradVal
	domain_dim = max(currentGradVal.shape)
	beta = 0
	beta_counter = domain_dim
	while (currentGradVal.T * currentGradVal > tol) and numIters < 1500:
		if (beta_counter >= domain_dim):
			beta = 0
			beta_counter = 1
		else:
			beta_counter += 1
			beta = (currentGradVal.T * currentGradVal) / (oldGradVal.T * oldGradVal)
			beta = beta[0,0]
		numIters = numIters + 1
		def phi(t):
			return func(x + t * (pFunc(x) + beta * prevP))
		def dphi(t):
			return (grad(x + t*(pFunc(x) + beta * prevP)).T * (pFunc(x) + beta * prevP))[0,0]

		alpha = lineSearch(alpha_max, phi, dphi, c_1, c_2)
		prevP = pFunc(x) + beta * prevP
		x = x + alpha * prevP
		beta_array.append(x)
		currentGradVal, oldGradVal = grad(x), currentGradVal
	return x

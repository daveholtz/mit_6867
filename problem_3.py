import numpy as np
import pylab
import homework1

l = 0
M = 10

def createFeatures(X=np.zeros(1), M=1):
	features = X**0
	for i in range(1, M+1):
		features = np.hstack([features, X**i])
	return features

def ridgeRegression(X, Y, l=0, M=1):
	features = createFeatures(X, M)

	firstTerm = np.linalg.inv(np.dot(np.transpose(features), features) + l*np.identity(M+1))
	secondTerm = np.transpose(features)
	thirdTerm = Y

	theta = np.dot(np.dot(firstTerm, secondTerm), thirdTerm)

	return theta

def predictRidge(X, theta):
	features = createFeatures(X, M)
	predictions = np.dot(features, theta)

	return predictions

X, Y = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/curvefitting.txt')
features = createFeatures(X, M)
theta = ridgeRegression(X, Y, l=l, M=M)

print theta
print predictRidge(X, theta)

pylab.plot(X, Y, X, predictRidge(X, theta))
pylab.show()
import numpy as np
import pylab
import homework1
import sys

def createFeatures(X=np.zeros(1), M=1):
	features = X**0
	for i in range(1, M+1):
		features = np.hstack([features, X**i])
	return features

def ridgeRegression(X, Y, l=0, M=1):
	# Closed form solution from http://www.hongliangjie.com/notes/lr.pdf
	features = createFeatures(X, M)

	firstTerm = np.linalg.inv(np.dot(np.transpose(features), features) + l*np.identity(M+1))
	secondTerm = np.transpose(features)
	thirdTerm = Y

	theta = np.dot(firstTerm, np.dot(secondTerm, thirdTerm))

	return theta

def predictRidge(X, theta):
	features = createFeatures(X, M=theta.size-1)
	predictions = np.dot(features, theta)

	return predictions

def MSE(predictions, actuals):
	mse = np.dot(np.transpose(predictions-actuals), (predictions-actuals))/float(predictions.size)

	return mse 

lambdas = np.array([0, .01, .05, .1, .5, 1, 5, 10, 50, 100, 500])
Ms = np.array([1, 2, 3, 5, 10, 25, 50])

X_train, Y_train = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/regress_train.txt')
X_test, Y_test = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/regress_test.txt')
X_validate, Y_validate = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/regress_validate.txt')

def gridSearch(lambdas, Ms):
	bestLambda = lambdas[0]
	bestM = Ms[0]
	bestTheta = ridgeRegression(X_train, Y_train, l=lambdas[0], M=Ms[0])
	bestPredictions = predictRidge(X_test, bestTheta)
	minMSE = MSE(bestPredictions, Y_test)

	for i in lambdas:
		for j in Ms:
			theta = ridgeRegression(X_train, Y_train, l=i, M=j)
			predictions = predictRidge(X_test, theta)
			mse = MSE(predictions, Y_test)
			print 'Lambda = %f, M = %d, Test MSE = %f, Train MSE = %f' % (i, j, mse, MSE(predictions, Y_train))
			if mse < minMSE:
				minMSE = mse
				bestLambda = i
				bestM = j 
				bestTheta = theta 
				bestPredictions = predictions

	result = 'Lambda = %d, M = %d, MSE = %d ' % (bestLambda, bestM, minMSE)
	print result

	return bestTheta


X, Y = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/curvefitting.txt')
theta = ridgeRegression(X, Y, l=0, M=10)	

predictions = predictRidge(X, theta)

pylab.plot(X, Y, 'ro',
	X, predictRidge(X, theta), 'bo')
pylab.show()

#bestTheta = gridSearch(lambdas, Ms)

order = np.argsort(X_test, axis=0)

theta = ridgeRegression(X_train, Y_train, M=5)



#pylab.plot(list(X_train[order].flatten()), list(Y_train[order].flatten()), 'ro',
#	X_train[order].flatten(), predictRidge(X_train, bestTheta)[order].flatten(), 'bo')
#pylab.show()
#
#pylab.plot(list(X_test[order].flatten()), list(Y_test[order].flatten()), 'ro',
#	X_test[order].flatten(), predictRidge(X_test, bestTheta)[order].flatten(), 'bo')
#pylab.show()
#
#pylab.plot(list(X_validate[order].flatten()), list(Y_validate[order].flatten()), 'ro',
#	X_validate[order].flatten(), predictRidge(X_validate, bestTheta)[order].flatten(), 'bo')
#pylab.show()

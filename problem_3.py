import numpy as np
import pylab
import homework1
import sys
import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *

def createFeatures(X=np.zeros(1), M=1):
	features = X**0
	for i in range(1, M+1):
		features = np.hstack([features, X**i])
	return features

def ridgeRegression(X, Y, l=0, M=1, features=False):
	# Closed form solution from http://www.hongliangjie.com/notes/lr.pdf
	if not features:
		features = createFeatures(X, M)
	else:
		features = X
		M = features.shape[1]-1

	firstTerm = np.linalg.inv(np.dot(np.transpose(features), features) + l*np.identity(M+1))
	secondTerm = np.transpose(features)
	thirdTerm = Y

	theta = np.dot(firstTerm, np.dot(secondTerm, thirdTerm))

	return theta

def predictRidge(X, theta, features=False):
	if not features:
		features = createFeatures(X, M=theta.size-1)
	else:
		features = X
	
	predictions = np.dot(features, theta)

	return predictions

def MSE(predictions, actuals):
	mse = np.dot(np.transpose(predictions-actuals), (predictions-actuals))/float(predictions.size)

	return mse 

def gridSearch(lambdas, Ms):
	results = np.empty([4,1])
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
			results = np.hstack([results, np.array([i, j, mse, MSE(predictions, Y_train)]).reshape(4,1)])
			if mse < minMSE:
				minMSE = mse
				bestLambda = i
				bestM = j 
				bestTheta = theta 
				bestPredictions = predictions

	result = 'Lambda = %d, M = %d, MSE = %d ' % (bestLambda, bestM, minMSE)
	print result

	return bestTheta, np.transpose(results)

def addInterceptTerm(X_array):
	data = np.concatenate((np.ones(X_array.shape[0]).reshape(-1, 1), X_array), axis=1)

	return data

def scaleFeatures(X, mean = None, sigma= None):
	if mean == None and sigma == None:
		mean = np.mean(X, axis=0)
		sigma = np.std(X, axis=0)

	meanArray = np.repeat(mean.reshape(1, -1), X.shape[0], axis=0)
	sigmaArray = np.repeat(sigma.reshape(1, -1), X.shape[0], axis=0)

	scaledFeatures = (X - meanArray)/sigmaArray 
	
	return scaledFeatures, mean, sigma

##########

X, Y = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/curvefitting.txt')
theta = ridgeRegression(X, Y, l=0, M=10)	

lambdas = np.array([0, .01, .05, .1, .5, 1, 5, 10, 50, 100, 500])
Ms = np.array([1, 2, 3, 5])

X_train, Y_train = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/regress_train.txt')
X_test, Y_test = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/regress_test.txt')
X_validate, Y_validate = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/regress_validate.txt')

thetaOLS = ridgeRegression(X_train, Y_train, l=0, M=10)
predictions = predictRidge(X_train, thetaOLS)
print predictions

print X_train
print Y_train
pylab.plot(X_train, Y_train, 'ro',
	X_train, predictRidge(X_train, thetaOLS), 'bo')
pylab.show()

predictions = predictRidge(X, theta)

pylab.plot(X, Y, 'ro',
	X, predictRidge(X, theta), 'bo')
pylab.show()

bestTheta, resultsArray = gridSearch(lambdas, Ms)

resultsArray = pd.DataFrame(resultsArray, columns=np.array(['lambda', 'M', 'MSE_test', 'MSE_train']))
resultsArray = resultsArray[1:len(resultsArray.index)]
print resultsArray.head(n=10)

resultsArray['lambda'] = resultsArray['lambda'].astype('category')

print ggplot(aes(x='M', y='MSE_train', colour=str('lambda')), data=resultsArray) + geom_line() + scale_y_log() + \
	ylab('MSE (test data)') + xlab('M (polynomial order)') + ggtitle('MSE dependence on M, lambda')

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

###
### Blog Feedback Data
###


X_train_blog = np.genfromtxt('/Users/dholtz/Downloads/6867_hw1_data/BlogFeedback_data/x_train.csv', delimiter=",")
Y_train_blog = np.genfromtxt('/Users/dholtz/Downloads/6867_hw1_data/BlogFeedback_data/y_train.csv', delimiter=",")

X_train_blog = addInterceptTerm(X_train_blog)

thetaBlog = ridgeRegression(X_train_blog, Y_train_blog, l=.01, features=True)
predictBlog = predictRidge(X_train_blog, thetaBlog, features=True)
mseBlog = MSE(predictBlog, Y_train_blog)

print scaleFeatures(X_train_blog)

#print thetaBlog
#print mseBlog
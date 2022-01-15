import numpy as np
import computeCost as cc

# theta = gradientDescent(X, y, theta, alpha, iterations)
def gradientDescent(X, y, theta, alpha, iterations):

	m = X.shape[0]
	print(m)

	for i in range(iterations):
		# 苦戦...sum()の結果がベクトルを返すので、
		# 1列複数行のyと計算できない
		# → 一行の転置にはreshape()を使うべし

		"""
		誤答: T(=trnspose())は一行の配列には働かない
		error = (X*theta).sum(axis=1)-y.T
		theta = theta - alpha*(((error)*X.T).sum(axis=1)).T/X.shape[0]
		"""

		theta = theta - alpha*(1/m)*(((X*theta).sum(axis=1).reshape(m,1)-y)*X).sum(axis=0)

		if i%100 == 0:
			print(cc.computeCost(X, y, theta))

	print ("gradient descent finish!")
	return theta







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from costFunction import costFunction
from predict import predict

def main():
	df = pd.read_csv("ex2data1.txt", sep=",", header=None)
	m = df.shape[0]

	X = df.values[:,0:2]
	X = np.concatenate((np.ones((m,1)),X),axis=1)
	y = df.values[:,2:3]
	initial_theta = np.zeros(())

	[cost, grad] = costFunction(initial_theta,X,y)
	theta = fminunc(costFunction, initial_theta, 300, 0.01, X, y)

# 重ね合わせる散布図の表示
	color = ["y","b"]
	for index, rec in df.iterrows():
		plt.scatter(rec[0],rec[1],c=color[int(rec[2])])
	plt.show()


def fminunc(func, theta, max_iter, alpha, X, y):
	lastCost = 1000000
	for i in range(max_iter):
		[cost, grad] = costFunction(theta, X, y)
		#if i%10 == 0:
		print("iter,cost:{},{}".format(i,cost))
		if lastCost - cost < 0.01:
			break
		else:
			lastCost = cost
		theta = theta - alpha*grad
	return theta

if __name__ == "__main__":
	main()

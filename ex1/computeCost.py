import numpy as np

def computeCost(X, y, theta):
	
	return sum(((X * theta) - y)**2)/(2*X.size)

# TEST
if __name__ == "__main__":
	X,y = np.loadtxt('ex1data1.txt',delimiter=",",unpack=True)
	theta = 0
	J = computeCost(X,y,theta)
	print(J)



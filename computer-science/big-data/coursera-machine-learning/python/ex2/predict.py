import numpy as np
from sigmoid import sigmoid

def predict(X,theta):
	m = X.shape[0]
	z = np.sum(theta*X,axis=1).reshape(m,1) 
	return sigmoid(z)

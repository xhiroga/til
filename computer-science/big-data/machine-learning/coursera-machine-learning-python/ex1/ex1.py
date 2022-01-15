import numpy as np
import matplotlib.pyplot as plt
import gradientDescent as gd

data = np.loadtxt("ex1data1.txt",delimiter=",")

m = data.shape[0]
X = np.concatenate((np.ones((m,1)),data[:,:1]), axis=1)
# np.r_[]とも書けるが、concatenameの方が性能で勝る

y = data[:,1:]
theta = np.zeros((1,2))

iterations = 1500;
alpha = 0.01;

print("Default Theta")
print(theta)

theta = gd.gradientDescent(X, y, theta, alpha, iterations)

print("Theta found by gradient descent")
print(theta)
# Theta found by gradient descent: -3.630291 1.166362
_y = (X*theta).sum(axis=1)


plt.scatter(X[:,1:],y)
plt.plot(X[:,1:],_y)
plt.show()



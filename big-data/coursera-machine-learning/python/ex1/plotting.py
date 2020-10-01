# データをロードして
# グラフ化する

import numpy as np
from matplotlib import pyplot as plt
import time

x,y = np.loadtxt("ex1data1.txt", delimiter=",", unpack=True)

fig = plt.figure() #新しいウィンドウを描画
ax = fig.add_subplot(1,1,1)

ax.scatter(x,y)
fig.show()

while(True):
	time.sleep(10)


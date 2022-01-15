import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# import datasets
df = pd.read_csv("ex2data1.txt", sep=",", header=None)
m,n = df.shape
X = df.values[:,:2]
Y = df.values[:,2:]

logreg = linear_model.LogisticRegression(C=1)
logreg.fit(X,Y)
# 本来なら目的関数の定義と最急降下法のための準備が必要
# fitメソッドを呼ぶだけで良い。


# 予測モデルを用いて散布図の背景に色を塗る。
# メッシュを作成して各点ごとに予測を適用すればよい。
x_min, x_max = X[:,0].min(), X[:,0].max()
y_min, y_max = X[:,1].min(), X[:,1].max()
xx, yy = np.meshgrid(np.arange(x_min,x_max,.02), np.arange(y_min, y_max, .02))
# 行方向に数字が進むxxと列方向のyyを生成

Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4,3)) #これインチで指定らしい
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired) # カラーメッシュ=要するにモザイク画と一緒

plt.scatter(X[:,0], X[:,1], c=Y.ravel(), edgecolors='k', cmap=plt.cm.Paired)
plt.show()



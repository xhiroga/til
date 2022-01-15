import numpy as np
import math
from logging import basicConfig, getLogger, INFO, DEBUG

import sigmoid as sg

basicConfig(level=INFO)
logger = getLogger(__name__)

# theta, X, yともにm行方向に長い想定
def costFunction(theta, X, y):
	logger.debug("cost compute start")
	m = X.shape[0]
	n = X.shape[1]

	# まずthetaとxの計算結果をシグモイド関数に入れて確率を算出
	# 欲しいコスト関数は、
	# y=0の時は1を予測すると怒り、y=1の時は0を予測すると怒るコスト
	# 怒る = 予測した確率が0に近いほど負の対数が増えること

	z = np.sum(theta*X,axis=1).reshape(m,1) # m行1列
	logger.debug("z: {}".format(z.shape))
	
	logz = np.array([math.log(el) for el in sg.sigmoid(z)]).reshape(m,1)

	J = (1/m)*sum(-y*logz-(1-y)*logz )
	logger.debug("cost J:{}".format(J.shape))

	grad = (1/m)*np.sum((sg.sigmoid(z)-y)*X,axis=0)# 1行特徴数列
	logger.debug("grad:".format(grad.shape))
	#sig(z)-y の絶対値じゃないのが腹落ちしない どういうことだ

	return J, grad


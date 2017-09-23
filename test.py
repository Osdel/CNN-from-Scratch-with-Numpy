import numpy as np 
from cnn import conv3d
from utils import merge_layers

"""
	x0,x1,d3 are the 3 depth layers of the image.
	w0,w1,w2 are the 3 depth layers of the first weigth matrix.
	w3,w4,w5 are the 3 depth layers of the first weigth matrix.
	b is the bias
	Example extracted from Andrej Karpathy CS231n CNN Lecture
"""
x0 = np.array(
	[
	[0,0,0,0,0,0,0],
	[0,0,0,0,2,1,0],
	[0,2,2,2,0,2,0],
	[0,1,0,0,0,1,0],
	[0,0,0,2,2,0,0],
	[0,0,0,1,1,1,0],
	[0,0,0,0,0,0,0],
	]
	)

x1 = np.array(
	[
	[0,0,0,0,0,0,0],
	[0,0,2,2,1,2,0],
	[0,0,0,2,0,2,0],
	[0,1,0,2,0,0,0],
	[0,2,0,2,0,2,0],
	[0,2,0,2,0,2,0],
	[0,0,0,0,0,0,0],
	]
	)
x2 = np.array(
	[
	[0,0,0,0,0,0,0],
	[0,2,2,0,2,2,0],
	[0,2,0,2,0,0,0],
	[0,0,2,2,2,2,0],
	[0,2,1,0,1,1,0],
	[0,0,2,0,1,0,0],
	[0,0,0,0,0,0,0],
	]
	)
w0 = np.array(
	[
	[0,0,1],
	[1,-1,-1],
	[1,1,1]
	]
	)

w1 = np.array(
	[
	[1,0,-1],
	[1,-1,-1],
	[0,0,0]
	]
	)

w2 = np.array(
	[
	[1,-1,-1],
	[0,1,-1],
	[0,1,-1]
	]
	)
w3 = np.array(
	[
	[0,0,-1],
	[1,1,-1],
	[0,-1,0]
	]
	)
w4 = np.array(
	[
	[0,0,1],
	[1,1,0],
	[-1,-1,1]
	]
	)
w5 = np.array(
	[
	[-1,1,0],
	[-1,0,0],
	[1,1,1]
	]
	)

b = np.ndarray(shape=(3,3,2), buffer=np.array([1,0]*9), offset=0, dtype=int)
	

X = merge_layers([x0,x1,x2])
w = merge_layers([w0,w1,w2])
ww = merge_layers([w3,w4,w5])
W = np.array([w,ww])

cnnl = conv3d(X,W,stride=2,padding=0) + b
print(cnnl[:,:,0])
print(cnnl[:,:,1])





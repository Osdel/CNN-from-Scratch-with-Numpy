import numpy as np 

class conv3d:
	"""
		Basic conv3d implementation.
	"""
	def __init__(self, volume3d, W, stride, padding):
		self.volume3d = volume3d
		self.W = W
		self.stride = stride
		self.padding = padding
		self.k = W.shape[0]
		shape = self.volume3d.shape
		self.width = (shape[0]-self.k+2*padding)/stride + 1
		self.height = (shape[1]-self.k+2*padding)/stride + 1
		self.depth = self.k
		self.output = np.zeros((int(self.width),int(self.height),int(self.k)))
		self.shape = self.output.shape
		self.convolve()

	def __call__(self):
		return self.output

	def __add__(self,other):
		return self.output + other	

	def __getitem__(self,key):
		return self.output[key]	

	"""
		Calculate the output element at [i,j,d] 
	"""	
	def calculate(self,i,j,d):
		f = self.W.shape[1]
		x = []
		for a in range(int(self.W.shape[1])):
			x.append(np.sum(self.volume3d[i:i+f,j:j+f,a] * self.W[d,:,:,a]))
		return np.sum(x)

    
	def convolve_layer(self, depth):
		rows=0
		cols=0
		output = np.zeros((int(self.width),int(self.height)))
		for i in range(0,int(self.volume3d.shape[0]-self.stride),int(self.stride)):
			cols = 0
			for j in range(0,int(self.volume3d.shape[1]-self.stride),int(self.stride)):
				output[rows,cols] = self.calculate(i,j,depth)
				cols += 1
			rows += 1
		self.output[:,:,depth]=output

	"""
		Convolve the entire output
	"""	
	def convolve(self):
		for k in range(int(self.k)):
			self.convolve_layer(k)
		return self.output
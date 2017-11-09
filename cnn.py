import numpy as np 

class conv2d:
	"""
		Basic conv2d implementation.
		volume: The input volume to convolve.
		W: weights volume
		stride: how much the fiter slide over the input volume
		padding: zero padding
	"""
	def __init__(self, volume, W, stride, padding):
		self.volume = volume
		if padding:
			self.volume = self.pad(padding)
		self.W = W
		self.stride = stride
		self.padding = padding
		self.k = W.shape[0]
		shape = self.volume.shape
		self.width = (shape[0]-self.k+2*padding)/stride + 1
		self.height = (shape[1]-self.k+2*padding)/stride + 1
		self.depth = self.k
		self.output = np.zeros((int(self.width),int(self.height),int(self.k)))
		self.shape = self.output.shape
		self.convolve()

	"""
	Get the output of a conv2d object calling it as a method.
	"""	
	def __call__(self):
		return self.output

	
	"""
	Allow the conv2d to be summed with a scalar value, sums the output with the value
	"""	
	def __add__(self,other):
		return self.output + other	
	
	"""
	Allow the conv2d to be multiplied with a scalar value, multiply the output with the value
	"""	
	def __mul__(self,other):
		return self.output * other

	def __getitem__(self,key):
		return self.output[key]	

	"""
	Pad the volume[:,:,i] with 'padding' zeros simetrically
	"""	
	def pad(self,padding):
		x = self.volume
		newshape = x.shape + np.array([2*padding,2*padding,0])
		result = np.zeros(newshape)
		depth = x.shape[2]
		for i in range(depth):
			temp = x[:,:,i]
			result[:,:,i] = np.lib.pad(temp,padding,'constant',constant_values=(0))
		return result

	"""
		Calculate the output element at [i,j,d] 
	"""	
	def calculate(self,i,j,d):
		f = self.W.shape[1]
		x = []
		for a in range(int(self.W.shape[1])):
			x.append(np.sum(self.volume[i:i+f,j:j+f,a] * self.W[d,:,:,a]))
		return np.sum(x)

	def convolve_layer(self, depth):
		rows=0
		cols=0
		output = np.zeros((int(self.width),int(self.height)))
		for i in range(0,int(self.volume.shape[0]-self.stride),int(self.stride)):
			cols = 0
			for j in range(0,int(self.volume.shape[1]-self.stride),int(self.stride)):
				output[rows,cols] = self.calculate(i,j,depth)
				cols += 1
			rows += 1
		self.output[:,:,depth]=output

		
	"""
		Convolve with the basic algorithm the entire output
	"""	
	def convolve_basic(self):
		for k in range(int(self.k)):
			self.convolve_layer(k)
		return self.output

	"""
	Another convolution method, memory use expensive, but faster algorithm than convolve_basic, 
	because is a simple matrix multiplication and can be computed with gpu
	"""	
	def convolve_im2col(self):
		w_dim = np.prod(self.W.shape[1:])
		cols = []
		weights = []
		#creating the volumes matrix
		s = self.stride
		f = self.W.shape[1]
		w = self.width*self.height
		for i in range(0, int(self.volume.shape[0]-s), s):
			for j in range(0, int(self.volume.shape[1]-s), s):
				region = self.volume[i:i+f, j:j+f, :]
				region = region.reshape((w_dim,))
				cols.append(region)
		cols = np.array(cols).T
		#creating the weights matrix
		for k in range(self.k):
			current = self.W[k,:,:,:]
			current = current.reshape((w_dim,))
			weights.append(current)
		weights = np.array(weights)
		#compute the matrix dot multipication
		result = np.dot(weights,cols)
		#reshape the output
		result = result.reshape(self.k, self.W.shape[1], self.W.shape[1])
		temp = result
		result = np.zeros(self.output.shape)
		for k in range(self.k):
			result[:,:,k] = temp[k,:,:]
		self.output = result
		return result

	"""
	Main convolution method.
	method: The convolution method to use, 'convolve_basic' for default, to use im2col method
	call it with method='convolve_im2col'
	"""	
	def convolve(self,method="convolve_basic"):
		if method == "convolve_im2col":
			return self.convolve_im2col()
		else:
			return self.convolve_basic()	



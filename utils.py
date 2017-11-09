import numpy as np

"""
	Merge the layers, Ex: merge_layers([7x7,7x7,7x7])=7x7x3
	Element at [:,:,d] follows the order declared. 
"""
def merge_layers(layers):
	shape = np.shape(layers[0])
	width = shape[0]
	height = shape[1]
	result = np.zeros((width,height,len(layers))) 
	for w in range(width):
		for h in range(height):
			temp = []
			for l in layers:
				temp.append(l[w,h])
			result[w,h] = np.array(temp)
	return result		
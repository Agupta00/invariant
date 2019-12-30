#! /usr/bin/env python3
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict

import pprint
import collections
import numpy as np
import operator
import cv2


xRange=100
yRange=100

def cycle(input, i=1):
	while True:
		for x in input[::i]:
			yield x


class Cm:
	def __init__(self,x,y):
		self.x=x
		self.y=y


	def neighbors(self):
		return [[x_,y_] for x_ in range(max(self.x-1,0),min(self.x+2,xRange)) for y_ in range(max(self.y-1,0),min(self.y+2,yRange)) ]

class Tensor:
	def __init__(self,cm=None, inputs=None, N=(0,0,0)):

		#unit_tensor
		if inputs==None :
			self.cm = cm
			self.N=N
			self.basis=None
			self.length=1
			self.inputs=inputs
			self.relLen_dict={}
			self.pointer_dict={}
		else:
			# self.cm = cm
			self.inputs=inputs
			self.cm=self.get_cm()
			self.N=(0,0,1)
			self.basis=None
			# self.length=1
			self.length=self.get_length()
			self.relLen_dict=self.get_relLen(inputs)
			self.pointer_dict=self.get_pointers(inputs)

	def __repr__(self, plist=["cm","pointer_dict", "length"]):
		if(self.N==(0,0,0)):
			plist=["cm"]
		rep= "\nTensor: " + str(self.N) +"\n"
		for atr in plist:
			rep+=atr + ": "+ str(getattr(self, atr))+"\n"
		return rep

	#runtime activation of this tensor given inputs
	def activate(self,inputs):
		
		s1=collections.Counter(x.N for x in self.inputs)
		s2=collections.Counter(x.N for x in inputs)

		if(s1!=s2):
			return False
		
		#TODO, if missing inputs allow for reduced activation

		#correct relative lengths
		if(self.relLen_dict!=self.get_relLen(inputs)):
			return False

		#correct pointers
		if not np.array_equal(self.pointer_dict,self.get_pointers(inputs)):
			# print(self.pointer_dict)
			# print(self.get_pointers(inputs))
			return False

		#correct basis

		return True


	def get_length(self):
		return sum(input.length for input in self.inputs)

	def get_cm(self):
		# print(self.inputs)
		cm = sum(np.array([[input.cm[0],input.cm[1]] for input in self.inputs]))/len(self.inputs)

		# self.cm = np.array([ np.array(sum(input.cm[i]) for input in self.inputs) for i in [1,2]])
		return cm
		# self.cm=self.cm/len(self.inputs)

		#relative lengths
	def get_relLen(self,inputs):
		inputs.sort(key= lambda x: x.cm[0] + x.cm[1]*yRange)
		relLen_dict=defaultdict(list)  #list backed multidic

		for i in range(len(inputs)):
			relLen_dict[inputs[i].N].append(inputs[i].length/self.length)
		return relLen_dict	


	def get_pointers(self,inputs):
		inputs.sort(key= lambda x: x.cm[0] + x.cm[1]*yRange)
		pointer_dict = defaultdict(list)  #list backed multidict
		
		if(len(inputs)<2): 
			print("error only one input, for input", inputs)
			return
		#[a,b,c,d]
		for i in range(len(inputs)):
			# delta x,y normalized by length
			pointer = lambda a,b: (np.array([inputs[a].cm[0]-inputs[b].cm[0], inputs[a].cm[1]-inputs[b].cm[1]])/self.length).tolist()
			if i==0:
				# which ti input,pointer to tj input -- where Neighboorhood determines identity
				#a->b
				pointer_dict[(inputs[i].N,inputs[i+1].N)].append(pointer(i,i+1))
				#a->d
				# pointer_dict[(inputs[i].N,inputs[-1].N)]=pointer(i,-1)
			elif i==len(inputs)-1:
				#d->c
				pointer_dict[(inputs[i].N,inputs[i-1].N)].append(pointer(i,i-1))
				#d->a
				# pointer_dict[(inputs[i].N,inputs[0].N)]=pointer(i,0)
			else:
				#b->a
				pointer_dict[(inputs[i].N,inputs[i-1].N)].append(pointer(i,i-1))
				#b->c
				pointer_dict[(inputs[i].N,inputs[i+1].N)].append(pointer(i,i+1))
		# print("pointers",pointer_dict)
		return pointer_dict


def inputs_fromImg(x,y,img):
	cm = Cm(x,y)

	#list of unit tensors which are not 0, aka 1 in boolean img input case
	unit_tensors=[Tensor(cm_) for cm_ in cm.neighbors() if img[cm_[0],cm_[1]]!=0]
	if(unit_tensors!=[]): 
		return unit_tensors
	else:
		#null tensor
		return [Tensor(N=(-1,-1,-1))]

def fowardPass(x,y,img):

	unit_tensors=inputs_fromImg(x,y,img)


	newTensor= Tensor(cm=[x,y], inputs=unit_tensors)
	return newTensor

def main():


	img = np.eye(20, dtype=np.float32)
	cv2.imshow("Image",img)

	tensor = fowardPass(3,3,img)
	inputs = inputs_fromImg(3,3,img)
	inputs2 = inputs_fromImg(6,5,img)


	# print(tensor.__repr__(plist=["relLen_dict"]))
	print(tensor.activate(inputs))
	print(tensor.activate(inputs2))


	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

main()



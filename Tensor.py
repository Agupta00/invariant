#! /usr/bin/env python3
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict

import pprint
import collections
import numpy as np
import operator
import cv2


key=1

xRange=yRange=512

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
	def __init__(self,cm=None, inputs=[], N=(0,0,1)):

		#unit_tensor
		# if(N==(-1,-1,-1)):
		# 	self.type="null tensor"
		if inputs==[]:
			self.cm = cm
			self.N=(0,0,0)
			self.basis=None
			self.length=1
			self.inputs=inputs
			self.relLen_dict={}
			self.pointer_dict={}
		else:
			# self.cm = cm
			self.inputs=inputs
			self.types=self.get_types(inputs)

			self.cm=self.get_cm()
			self.N=N
			self.basis=None
			# self.length=1
			self.length=self.get_length()
			self.relLen_dict=self.get_relLen(inputs)
			self.pointer_dict=self.get_pointers(inputs)

	def __eq__(self, other): 
		if(self.types==other.types and
			self.relLen_dict==other.relLen_dict and
			self.pointer_dict==other.pointer_dict): 
			return True
		else:
			return False
	def __hash__(self):
		return 1

	def __repr__(self, plist=["cm","pointer_dict", "length", "inputs"]):
		if(self.N==(0,0,0)):
			plist=["cm"]
		rep= "\nTensor: " + str(self.N) +"\n"
		for atr in plist:
			rep+=atr + ": "+ str(getattr(self, atr))+"\n"
		return rep

	#runtime activation of this tensor given inputs
	def activate(self,inputs):
		
		if(self.get_types(inputs)!=self.types):
			return False
		
		#TODO, if missing inputs allow for reduced activation

		#correct relative lengths
		if(self.relLen_dict!=self.get_relLen(inputs)):
			return False

		#correct pointers
		if (self.pointer_dict!=self.get_pointers(inputs)):
			# print(self.pointer_dict)
			# print(self.get_pointers(inputs))
			return False

		#correct basis

		return True

	@staticmethod
	def get_types(inputs):
		return collections.Counter(x.N for x in inputs)

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
			# print("error only one input, for input", inputs)
			return pointer_dict
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
	# test = [img[cm_[0]][cm_[1]] for cm_ in cm.neighbors() if img[cm_[0]][cm_[1]]!=999999]
	unit_tensors=[Tensor(cm_) for cm_ in cm.neighbors() if img[cm_[0],cm_[1]]!=0]
	if(unit_tensors!=[]): 
		return unit_tensors
	else:
		#null tensor
		# return [Tensor(N=(-1,-1,-1))]
		return None

def fowardPass(x,y,img):

	unit_tensors=inputs_fromImg(x,y,img)
	newTensor= Tensor(cm=[x,y], inputs=unit_tensors)
	return newTensor

def main():


	img = np.eye(xRange, dtype=np.float32)
	cv2.imshow("Image",img)

	tensor = fowardPass(6,6,img)
	# inputs = inputs_fromImg(6,6,img)



	# print(tensor.__repr__(plist=["relLen_dict"]))
	print(tensor)


	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

def smt():
	key=5
	# xRange=yRange=512

	# img = np.eye(xRange, dtype=np.float32)
	img = np.zeros((512,512,3), np.float32)
	# cv2.rectangle(img,(384,0),(510,128),(0,255,0),100)
	cv2.circle(img,(150,150), 100, (0,0,255), -1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	print(img.shape)
	cv2.imshow("lalala", img)	
	# k = cv2.waitKey(0) # 0==wait forever
	#no sliding for now
	delta=1

	#all z0 tensors
	#dict of tensors
	z0=set()
	z=dict()
	for x in range(0,xRange,delta):	
		for y in range(0,yRange,delta):
			unit_tensors=inputs_fromImg(x+1,y+1,img)
			# if(x==6 and y==6):
			# 	print(unit_tensors)
			if unit_tensors==None:
				continue
			tensor= Tensor(cm=[x,y], inputs=unit_tensors, N=(0,0,key))
			
			#todo implement hash function
			if tensor not in z0:
				key+=3
				z0.add(tensor)
				z[tensor]=tensor.N
	print("found {} tensors".format(len(z0)))


	#color picture using tensor types
	img2=np.zeros((512,512,), np.uint8)
	for x in range(0,xRange,delta):	
		for y in range(0,yRange,delta):
			unit_tensors=inputs_fromImg(x+1,y+1,img)
			if unit_tensors==None:
				continue
			tensor= Tensor(cm=[x,y], inputs=unit_tensors, N=(0,0,key))

			# print(z[tensor])
			img2[x][y]=z[tensor][2]
	cv2.imshow("new img",img2)
	k = cv2.waitKey(0) # 0==wait forever



	# z0.pop()
	# print("here is one of the tensors {}".format(z0.pop()))

	# pprint.pprint(z0)




smt()
# main()


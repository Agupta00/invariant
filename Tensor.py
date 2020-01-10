#! /usr/bin/env python3
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict

import pprint
import collections
import numpy as np
import operator
import cv2
import networkx as nx
import math


key=1

xRange=yRange=510


def neighbors(x, y, xRange, yRange, i=1,limit=1):
	# i=2, limit=2, where a is the returned values
	#     aaaaa
	#     a!!!a
	#     a!T!a
	#     a!!!a
	#	  aaaaa

	while (limit>0):
		#top row
		list=[[x_,y-i] for x_ in range(max(x-i,0),min(x+1+i,xRange)) if y-i <yRange]
		# print(list,"---")
		#left side and right side
		list+=[[x_,y_] for x_ in [x-i] for y_ in range(max(y+1-i,0),min(y+i,yRange)) if x-i>=0 ]
		list+=[[x_,y_] for x_ in [x+i] for y_ in range(max(y+1-i,0),min(y+i,yRange)) if x+i<xRange    ]

		# print(list,"---")
		#bottom row
		list+=[[x_,y+i] for x_ in range(max(x-i,0),min(x+1+i,xRange)) if y+i <yRange]
		# print(list,"---")

		for a in list:
			yield a
		limit-=1
		i+=1


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
			self.length=1
			self.inputs=inputs
			self.relLen_dict={}
			self.pointer_dict={}
			self.input_dict=dict()
		else:
			# self.cm = cm
			self.inputs=inputs
			self.input_dict=dict()
			self.input_dict=self.init_types(inputs)
			self.cm=self.get_cm()
			self.N=N
			# self.length=1
			self.length=self.get_length()
			self.relLen_dict=self.get_relLen(inputs)
			self.pointer_dict=self.init_pointer_dict(inputs)

	def init_types(self,inputs):
		#returns updated_dict since help=1 is on, with w=p(0)
		input_dict=self.activate_component(self.get_inputs(self,inputs),{},help=1)
		# print("input_dict",input_dict)
		return input_dict

	def init_pointer_dict(self,inputs):
		inputs=self.get_pointers(inputs)
		pointer_dict=self.activate_component(inputs, {}, help=1)
		return pointer_dict

	def __eq__(self, other): 
		if(self.input_dict==other.input_dict and
			self.relLen_dict==other.relLen_dict and
			self.pointer_dict==other.pointer_dict): 
			return True
		else:
			return False
	def __hash__(self):
		return hash(frozenset(self.input_dict).union(self.relLen_dict).union(self.pointer_dict))

	def __repr__(self, plist=["cm","pointer_dict", "length", "inputs"]):
		if(self.N==(0,0,0)):
			plist=["cm"]
		rep= "\nTensor: " + str(self.N) +"\n"
		for atr in plist:
			rep+=atr + ": "+ str(getattr(self, atr))+"\n"
		return rep

	def accumulate(self, input):
		if self.inputs==None:
			self.inputs=[]
		self.inputs+=inputs

	def clean(self):
		self.inputs=[]
		self.cm=None
		self.length=None

	#runtime activation of this tensor given inputs

	def activate_component(self,inputs,input_dict, norm=1/15, threshold=3/4, help=0):
	    # input_dict=self.input_dict
	    act=help
	    # left side of the gaussian curve, from domain 0->1
	    p = lambda w: min(math.pow(math.e, (math.pi * -math.pow((w-1),2))),1)
	    dw = lambda w: -2*math.pi*(w-1)*p(w)

	    #return updated if activated, else keep old one
	    updated_input_dict=input_dict.copy()
	    #inputs is a set of type counter ie (type:number)
	    #input_dict=( (type:number),weight)
	    for key,w in input_dict.items():

	        if(present(key,inputs)):
	            print(key,present(key,inputs),"   ",w)
	            act+=present(key,inputs)*w
	            #probability of weight w
	            
	            #w=w+1/norm*dw
	            updated_input_dict[key]=round(updated_input_dict[key]+norm*dw(w),4)
	        else:
	            print(key, norm*dw(w))
	            #decay weight
	            #w-= todo
	            updated_input_dict[key]=round(updated_input_dict[key]-norm*dw(w),4)


	    #normalize over avg weights and number of inputs
	    # print("sum act",act)
	    # act=act*sum(input_dict.values())/(int(len(input_dict))**2)
	    try:
	        act=round(act/sum(input_dict.values()),5)
	    except:
	    	pass


	    #add weights for present inputs who were not in input_dict to updated dict
	    if act>threshold:
	        for item in inputs:
	            if(item not in input_dict):
	                # (a:2)=0.005 #(type:number)=initial weight
	                updated_input_dict[item]=round(p(0),4)
	                # self.input_dict=updated_input_dict
	        return updated_input_dict
	    else:
	        return None



	def activate(self,inputs):


		# if(self.get_input_dict(inputs)!=self.input_dict):
		# 	return False

		if activate_component(self,get_inputs(inputs),self.input_dict)==None:
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
	def get_inputs(self,inputs):
		# return collections.Counter(x.N for x in inputs)
		counter = collections.Counter(x.N for x in inputs)
		#makes counter hashable goes from a:2 -> (a,2) where a is x.N
		inputs = frozenset(counter.items())
		return inputs


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
	    # pointer_dict = defaultdict(list)  #list backed multidict
	    pointer_counter = collections.Counter()

	    if(len(inputs)<2): 
	        # print("error only one input, for input", inputs)
	        return pointer_counter
	    #[a,b,c,d]
	    for i in range(len(inputs)):
	        # delta x,y normalized by length
	        # =>( (a,b)(x,y) )
	        # (inputs[a].N,inputs[b].N)
	        pointer = lambda a,b: ( (inputs[a].N,inputs[b].N),
	            tuple(np.array([inputs[a].cm[0]-inputs[b].cm[0], inputs[a].cm[1]-inputs[b].cm[1]])/self.length) )

	        if i==0:
	            # which ti input,pointer to tj input -- where Neighboorhood determines identity
	            #a->b
	            # pointer_dict[(inputs[i].N,inputs[i+1].N)].append(pointer(i,i+1))
	            pointer_counter.update([pointer(i,i+1)])
	            #a->d
	            # pointer_dict[(inputs[i].N,inputs[-1].N)]=pointer(i,-1)
	        elif i==len(inputs)-1:
	            #d->c
	            # pointer_dict[(inputs[i].N,inputs[i-1].N)].append(pointer(i,i-1))
	            pointer_counter.update([pointer(i,i-1)])
	            #d->a
	            # pointer_dict[(inputs[i].N,inputs[0].N)]=pointer(i,0)
	        else:
	            #a<-b
	            # pointer_dict[(inputs[i].N,inputs[i-1].N)].append(pointer(i,i-1))
	            pointer_counter.update([pointer(i,i-1)])
	            #b->c
	            # pointer_dict[(inputs[i].N,inputs[i+1].N)].append(pointer(i,i+1))
	            pointer_counter.update([pointer(i,i+1)])
	    # print("pointers",pointer_dict)
	    return frozenset(pointer_counter)


def inputs_fromImg(x,y,img,unit_tensor=True):
	cm = Cm(x,y)
	cm=[x,y]
	sizeX,sizeY=img.shape[0],img.shape[1]
	#list of unit tensors which are not 0, aka 1 in boolean img input case
	# test = [img[cm_[0]][cm_[1]] for cm_ in cm.neighbors() if img[cm_[0]][cm_[1]]!=999999]
	if unit_tensor:
		tensors=[Tensor(cm_) for cm_ in neighbors(x,y, xRange=sizeX,yRange=sizeY ) if img[cm_[0],cm_[1]]!=0]
	# else:
	# 	tensors=[img[] for cm_ in cm.neighbors() if img[cm_[0],cm_[1]]!=0]

	if(tensors!=[]): 
		return tensors
	else:
		#null tensor
		# return [Tensor(N=(-1,-1,-1))]
		return None

def fowardPass(x,y,img):

	unit_tensors=inputs_fromImg(x,y,img)
	newTensor= Tensor(cm=[x,y], inputs=unit_tensors)
	return newTensor


#for each active tensor get the nearest K tensors, corresponds to delta-1 sliding kernel
#given an input of an array of tensors
def learn_z0(img, x=0,y=0,delta=1):
	key=50
	# xRange=yRange=512

	#all z0 tensors
	#dict of tensors
	sizeX,sizeY=img.shape[0],img.shape[1]
	for x in range(1,sizeX,delta):	
		for y in range(1,sizeY,delta):
			unit_tensors=inputs_fromImg(x,y,img)
			# if(x==6 and y==6):
			# 	print(unit_tensors)
			if unit_tensors==None:
				continue
			tensor= Tensor(cm=[x,y], inputs=unit_tensors, N=(x,y,key))
			if tensor not in z:
				key+=10
				z[tensor]=tensor.N
	print("found {} tensors".format(len(z)))
	return z


def convertToZ0(img, delta=1):
	#color picture using tensor types
	img2=np.zeros((img.shape[0]//delta,img.shape[1]//delta,), np.uint8)

	print("img shape",img.shape)
	for x in range(1,img.shape[0],delta):	
		for y in range(1,img.shape[1],delta):
			unit_tensors=inputs_fromImg(x,y,img)
			if unit_tensors==None:
				continue
			tensor= Tensor(cm=[x,y], inputs=unit_tensors, N=(0,0,0))

			# print(z[tensor])
			if z.get(tensor)==None:
				img2[int(x/delta)][int(y/delta)]=0
			else:
				img2[int(x/delta)][int(y/delta)]=z[tensor][2]

	# print(z.values())
	# print([img2[x][y] for x in range(512) for y in range(512) if img2[x][y]!=0])
	# cv2.imshow("Z0",img2)
	# k = cv2.waitKey(0) # 0==wait forever
	return img2



z=dict()
def main2():


	#circle
	# img = np.eye(xRange, dtype=np.float32)
	img = np.zeros((xRange,xRange,3), np.uint8)
	# cv2.rectangle(img,(384,0),(510,128),(0,255,0),100)
	cv2.circle(img,(250,250), 200, (0,0,255), -1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imshow("lalala", img)	

	#update z0 columb
	z.update(learn_z0(img))
	actImg=convertToZ0(img,delta=3)

	cv2.imshow("Z0",actImg)
	k = cv2.waitKey(0) # 0==wait forever


	# z.update(learn_z0(actImg))
	# newImg=convertToZ0(actImg)
	# cv2.imshow("Z0",newImg)
	# k = cv2.waitKey(0) # 0==wait forever

	for i in range(1,10):
		print(i,"---")
		img = test(img,i)



def test(img,i):
	cv2.namedWindow("z",cv2.WINDOW_NORMAL)
	cv2.resizeWindow("z", 600,600)

	learn_z0(img,i,i,delta=3)
	outImg=convertToZ0(img,delta=3)
	cv2.imshow("z",outImg)
	k = cv2.waitKey(0) # 0==wait forever

	return outImg






# main2()


def smt():
	key=50
	# xRange=yRange=512

	# img = np.eye(xRange, dtype=np.float32)
	img = np.zeros((xRange,xRange,3), np.uint8)
	# cv2.rectangle(img,(384,0),(510,128),(0,255,0),100)
	cv2.circle(img,(250,250), 200, (0,0,255), -1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	print(img.shape)
	cv2.imshow("lalala", img)	
	# k = cv2.waitKey(0) # 0==wait forever
	#no sliding for now
	delta=3

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
				key+=10
				z0.add(tensor)
				z[tensor]=tensor.N
	print("found {} tensors".format(len(z0)))


	img2=np.zeros((xRange,yRange,), np.uint8)
	for x in range(0,xRange,1):	
		for y in range(0,yRange,1):
			unit_tensors=inputs_fromImg(x+1,y+1,img)
			if unit_tensors==None:
				continue
			tensor= Tensor(cm=[x,y], inputs=unit_tensors, N=(0,0,key))

			# print(z[tensor])
			if z.get(tensor)==None:
				img2[x][y]=0
			else:
				img2[x][y]=z[tensor][2]

	# print(z.values())
	# print([img2[x][y] for x in range(512) for y in range(512) if img2[x][y]!=0])
	cv2.imshow("new img",img2)
	k = cv2.waitKey(0) # 0==wait forever



	# z0.pop()
	# print("here is one of the tensors {}".format(z0.pop()))

	# pprint.pprint(z0)




# smt()
# main()


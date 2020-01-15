#! /usr/bin/env python3
from typing import List, Set, Dict, Tuple, Optional
import collections
from collections import defaultdict

import pprint

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
		neighbors=[[x,y]]
		neighbors+=[[x_,y-i] for x_ in range(max(x-i,0),min(x+1+i,xRange)) if y-i <yRange]
		# print(list,"---")
		#left side and right side
		neighbors+=[[x_,y_] for x_ in [x-i] for y_ in range(max(y+1-i,0),min(y+i,yRange)) if x-i>=0 ]
		neighbors+=[[x_,y_] for x_ in [x+i] for y_ in range(max(y+1-i,0),min(y+i,yRange)) if x+i<xRange    ]

		# print(list,"---")
		#bottom row
		neighbors+=[[x_,y+i] for x_ in range(max(x-i,0),min(x+1+i,xRange)) if y+i <yRange]
		# print(list,"---")

		for a in neighbors:
			yield a
		limit-=1
		i+=1


class Tensor:
	edges_dict=defaultdict(list)

	def __init__(self,cm=None, inputs=[], N=(0,0,1),unit_tensor=False):
		#key: value = N(x,y,z):[edges]
		if unit_tensor:
			self.cm = cm
			self.N=(0,0,0)
			self.length=1
			self.inputs=None
			self.relLen_dict={}
			self.pointer_dict={}
			self.input_dict=dict()
			self.neighbors=[]
		else:
			# self.cm = cm
			self.inputs=inputs
			self.input_dict=dict()
			self.init_types(inputs)
			self.cm=self.get_cm()
			self.N=N
			# self.length=1
			self.length=self.get_length()
			self.relLen_dict=self.get_relLen(inputs)
			self.init_pointer_dict(inputs)
			self.neighbors=lambda: Tensor.edges_dict[self.N]
			self.clean()

	def init_types(self,inputs):
		#returns updated_dict since help=1 is on, with w=p(0)
		self.activate_component(self.get_inputs(inputs),{},help=1, atribute="input_dict")
		# print("input_dict",input_dict)
		return 

	def init_pointer_dict(self,inputs):
		inputs=self.get_pointers(inputs)
		self.activate_component(inputs, {}, help=1, atribute="pointer_dict")
		return 

	def __eq__(self, other): 
		if(frozenset(self.input_dict)==frozenset(other.input_dict) and
			frozenset(self.relLen_dict)==frozenset(other.relLen_dict) and
			frozenset(self.pointer_dict)==frozenset(other.pointer_dict)): 
			return True
		else:
			return False
	def __hash__(self):
		return hash(frozenset(self.input_dict).union(self.relLen_dict).union(self.pointer_dict))

	def __repr__(self, plist=["cm","neighbors","inputs","pointer_dict"]):
		if(self.N==(0,0,0)):
			plist=["cm"]
		rep= "\nTensor: " + str(self.N) +"\n"
		for atr in plist:
			if atr=="neighbors":
				rep+=atr + ": "+ str(getattr(self, atr)())+"\n"
			else:
				rep+=atr + ": "+ str(getattr(self, atr))+"\n"
		return rep

	def accumulate(self, input):
		self.inputs.append(input)

	def clean(self):
		# return
		self.inputs=[]
		# self.cm=None
		# self.length=None

	#runtime activation of this tensor given inputs

	#updates atribute_dict and returns activation (int), 0 if no activation
	def activate_component(self,inputs,input_dict,atribute, norm=1/15, threshold=3/4, help=0):
	    # input_dict=self.input_dict
	    act=help
	    present = lambda key, inputs: 1 if key in inputs else 0

	    # left side of the gaussian curve, from domain 0->1
	    p = lambda w: min(math.pow(math.e, (math.pi * -math.pow((w-1),2))),1)
	    dw = lambda w: -2*math.pi*(w-1)*p(w)

	    #return updated if activated, else keep old one
	    updated_input_dict=input_dict.copy()
	    #inputs is a set of type counter ie (type:number)
	    #input_dict=( (type:number),weight)
	    for key,w in input_dict.items():

	        if(present(key,inputs)):
	            # print(key,present(key,inputs),"   ",w)
	            act+=present(key,inputs)*w
	            #probability of weight w
	            
	            #w=w+1/norm*dw
	            updated_input_dict[key]=round(updated_input_dict[key]+norm*dw(w),4)
	        else:
	            # print(key, norm*dw(w))
	            #decay weight
	            #w-= todo
	            updated_input_dict[key]=round(updated_input_dict[key]-norm*dw(w),4)


	    #normalize over avg weights and number of inputs
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
	                setattr(self,atribute,updated_input_dict)
	        return act
	    else:
	        return 0

	def activate(self):
		# if self.activate_component(self.get_inputs(self.inputs),self.input_dict)==False:
		# 	self.clean()
		# 	return False
		act=0
		
		#TODO, if missing inputs allow for reduced activation

		#correct relative lengths
		if(self.relLen_dict!=self.get_relLen(self.inputs)):
			self.clean()
			return 0


		act=self.activate_component(self.get_pointers(self.inputs), self.pointer_dict, atribute="pointer_dict")
		if act==0:
			self.clean()
			return 0

		# if (self.pointer_dict!=self.get_pointers(self.inputs)):
		# 	self.clean()
		# 	# print(self.pointer_dict)
		# 	# print(self.get_pointers(inputs))
		# 	return False

		#correct basis
		self.clean()
		return act


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


def inputs_fromImg(x,y,img, sizeX=None,sizeY=None):
	if sizeX==None:
		sizeX=img.shape[0]
		sizeY=img.shape[1]

	cm =[x,y]
	input_tensors=[img[cm_[0]][cm_[1]] for cm_ in neighbors(x,y, xRange=sizeX,yRange=sizeY ) if type(img[cm_[0],cm_[1]])!=int]

	if(len(input_tensors)>1): 
		return input_tensors
	else:
		return None

#todo optimize
#pixel->unit_tensor
def transformImg(img):
	tensorImg=np.zeros((img.shape[0],img.shape[1]), Tensor) 
	for x in range(0,img.shape[0]):
		for y in range(0,img.shape[1]):
			if(img[x][y]!=0):
				tensorImg[x][y]=Tensor(cm=[x,y],unit_tensor=True)
	return tensorImg

#for each active tensor get the nearest K tensors, corresponds to delta-1 sliding kernel
#given an input of an array of tensors
def learn_z0(img,z, Nx=0,Ny=0,delta=1):
	key=50
	sizeX,sizeY=img.shape[0],img.shape[1]
	for x in range(1,sizeX,delta):	
		for y in range(1,sizeY,delta):
			input_tensors=inputs_fromImg(x,y,img)
			# if(x==6 and y==6):
			# 	print(unit_tensors)
			if input_tensors==None:
				continue
			# print(unit_tensors,"x,y: ",(x,y),"-----\n")
			tensor= Tensor(cm=[x,y], inputs=input_tensors, N=(Nx,Ny,key))
			if tensor not in z:
				key+=10
				z[tensor]=tensor.N
	print("found {} tensors".format(len(z)))


def accumlate_neighbors(self,inputs):
	for neighbor in self.neighbors():
		#for now edge is just a Parent tensor, todo update with weight

		#provide region around the given tensor to learn relations
		neighbor.accumulate(inputs)


def learn(img,z, Nx=0,Ny=0,delta=1):
	img2=np.zeros((img.shape[0]//delta,img.shape[1]//delta), Tensor)

	newParentTensors=0
	key=150
	sizeX,sizeY=img.shape[0],img.shape[1]
	for x in range(1,sizeX,delta):	
		for y in range(1,sizeY,delta):			
			#tensors in the ROI centered around x,y
			input_tensors=inputs_fromImg(x,y,img)
			if input_tensors==None:
				continue

			#accumulate current inputs to parent tensors
			parentTensors=[]
			for tensor in input_tensors:
				for neighbor in tensor.neighbors():
					parentTensors.append(neighbor)
					#gets neighboring tensors around the tensor, if they are within the ROI
					neighbor.accumulate(tensor)
					# neighbor.accumulate([tensor for tensor in input_tensors if tensor.cm[0]<=x+1 and tensor.cm[1]<=y+1])
				
			# activate each parent tensor
			# for parentTensor in parentTensors:
			# 	parentTensor.activate()

			if parentTensors!=[]:
				activations= np.array([parentTensor.activate() for parentTensor in parentTensors])
				index_max=np.argmax(activations)

			if parentTensors==[] or activations[index_max]==0:
				img2[int(x//delta)-1][int(y//delta)-1]=0
				#if no outgoing edges
				#adds a new tensor to the
				parentTensor=Tensor(inputs=input_tensors,N=(Nx,Ny,key))
				key+=10

				if parentTensor not in z:
					z[parentTensor]=parentTensor.N
					newParentTensors+=1
				else:
					pass
			

				#add a connection to the parentTensor
				for input in input_tensors:
					# print("out edges")
					# print(Tensor.edges_dict[input.N])
					Tensor.edges_dict[input.N].append(parentTensor)
					# print("these are the inputs that i am adding a connection to",input)

			else:
				# print(len(activations))
				img2[int(x/delta)][int(y/delta)]=parentTensors[index_max]

			# if parentTensors!=[]:
			# 	activations= np.array([parentTensor.activate() for parentTensor in parentTensors])
			# 	index_max=np.argmax(activations)

			# 	if activations[index_max]==0:
			# 		img2[int(x/delta)][int(y/delta)]=0
			# 	else:
			# 		img2[int(x/delta)][int(y/delta)]=parentTensors[index_max]

			# else:
			# 	#if no outgoing edges
			# 	#adds a new tensor to the
			# 	parentTensor=Tensor(inputs=input_tensors,N=(Nx,Ny,key))
			# 	key+=10

			# 	if parentTensor not in z:
			# 		z[parentTensor]=parentTensor.N
			# 		newParentTensors+=1
			# 	else:
			# 		pass
			

			# 	#add a connection to the parentTensor
			# 	for input in input_tensors:
			# 		Tensor.edges_dict[input.N].append(parentTensor)
			# 		# print("these are the inputs that i am adding a connection to",input)


	print("found {} new parent tensors".format(newParentTensors))
	return img2

#converts to Big tensor given img of tensor inputs
def showNeighborhood(img,z, delta=1):
	img2=np.zeros((img.shape[0]//delta,img.shape[1]//delta,), np.uint8)

	print("img shape",img.shape)
	for x in range(1,img.shape[0],delta):	
		for y in range(1,img.shape[1],delta):
			input_tensors=inputs_fromImg(x,y,img)
			# print(unit_tensors)
			if input_tensors==None:
				continue
			tensor= Tensor(cm=[x,y], inputs=input_tensors, N=(0,0,0))

			# print(z[tensor])
			if z.get(tensor)==None:
				img2[int(x/delta)][int(y/delta)]=0
			else:
				img2[int(x/delta)][int(y/delta)]=z[tensor][2]
	return img2


#given tensor img of layer i gives tensor img of i+1
def fowardPass(img,z, delta=1):
	img2=np.zeros((img.shape[0]//delta,img.shape[1]//delta,), Tensor)

	print("img shape",img.shape)
	for x in range(1,img.shape[0],delta):	
		for y in range(1,img.shape[1],delta):
			input_tensors=inputs_fromImg(x,y,img)
			# print(unit_tensors)
			if input_tensors==None:
				continue
			tensor= Tensor(cm=[x,y], inputs=input_tensors, N=(0,0,0))

			# print(z[tensor])
			if z.get(tensor)==None:
				img2[int(x/delta)][int(y/delta)]=0
			else:
				tensor.N=z.get(tensor)
				img2[int(x/delta)][int(y/delta)]=tensor

	# print(z.values())
	# print([img2[x][y] for x in range(512) for y in range(512) if img2[x][y]!=0])
	# cv2.imshow("Z0",img2)
	# k = cv2.waitKey(0) # 0==wait forever
	return img2




def main():
		img=np.eye(3)
		img=transformImg(img)
		z=dict()
		# z0=showNeighborhood0(img)

		# unit_tensors=inputs_fromImg(2,1,img,True)
		# print(unit_tensors)


		learn_z0(img,z)
		print(z)

# main()

def main2():

	z=dict()
	#circle
	# img = np.eye(27, dtype=np.uint8)
	img = np.zeros((xRange,xRange,3), np.uint8)
	# cv2.rectangle(img,(384,0),(510,128),(0,255,0),100)
	cv2.circle(img,(250,500), 200, (0,0,255), -1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.Laplacian(img,cv2.CV_64F)
	# cv2.imshow("laplazian filter", img)	
	# k=cv2.waitKey(0)

	#tensor array
	img=transformImg(img)
	#z0 img
	learn_z0(img,z,delta=1)

	# picture =showNeighborhood(img,z,delta=3)
	# cv2.imshow("z0",picture)
	# k=cv2.waitKey(0)


	z0Img=fowardPass(img,z,delta=1)
	print(z0Img.shape,"shape")

	z1=dict()
	for i in range(1):
		z1Img=learn(z0Img,z1,Nx=1,Ny=1,delta=3)	
		z1Img_pixels=convertTensorToPixel(z1Img)
		show(z1Img_pixels)

	# for key,value in z1.items():
	# 	print(key)

	# for i in range(1):
	# 	img=recursiveZ(img,z,i)

def convertTensorToPixel(img):
	count=0
	img2=np.zeros((img.shape[0],img.shape[1]), np.uint8)
	sizeX,sizeY=img.shape[0],img.shape[1]
	for x in range(0,sizeX):	
		for y in range(0,sizeY):	
			if type(img[x][y])!=int:
				count+=1
				img2[x][y]=img[x][y].N[2]	
	print(count)
	return img2	
			#tenso
def show(img):
	cv2.namedWindow("z",cv2.WINDOW_NORMAL)
	cv2.resizeWindow("z", 600,600)
	cv2.imshow("z",img)
	k = cv2.waitKey(0) # 0==wait forever


def recursiveZ(img,z,i):
	cv2.namedWindow("z",cv2.WINDOW_NORMAL)
	cv2.resizeWindow("z", 600,600)
	learn_z0(img,z,i,i,delta=3)
	Z1=showNeighborhood(img,z,delta=3)
	cv2.imshow("z",Z1)
	k = cv2.waitKey(0) # 0==wait forever
	return Z1

main2()


def test(img,z,i):
	cv2.namedWindow("z",cv2.WINDOW_NORMAL)
	cv2.resizeWindow("z", 600,600)

	learn_z0(img,z,i,i,delta=3,inputs_fromImg=False)
	outImg=showNeighborhood(img,z,delta=3)
	cv2.imshow("z",outImg)
	k = cv2.waitKey(0) # 0==wait forever

	return outImg








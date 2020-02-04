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

class Tensor:
	def __init__(self,cm, inputs=[], N=(0,0,1),unit_tensors=False):

		#unit_tensor
		# if(N==(-1,-1,-1)):
		# 	self.type="null tensor"
		if inputs==[] or unit_tensors:
			self.cm = cm
			self.N=(0,0,0)
			self.length=1
			self.inputs=inputs
			self.relLen_dict={}
			self.pointer_dict={}
			self.input_dict=dict()
			self.weights=[]
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
			self.weights=[]



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

xRange=yRange=510

import Tensor
def main():

	img = np.eye(xRange, dtype=np.float32)
	# img = np.zeros((xRange,xRange,3), np.uint8)
	# # cv2.rectangle(img,(384,0),(510,128),(0,255,0),100)
	# cv2.circle(img,(5,5), 2, (0,0,255), -1)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("Image",img)

	tensor = Tensor.fowardPass(3,3,img)
	# inputs = inputs_fromImg(6,6,img)

	z=dict()
	Tensor.learn
	return tensor
	print(tensor.__repr__(plist=["pointer_dict"]))
	# print(tensor)


	cv2.waitKey(0)
	cv2.destroyAllWindows()
# print(main().__repr__(plist=["pointer_dict"]))

# Tensor.main2()


for x in range(2):
	y=2
	print(x)
print(y)









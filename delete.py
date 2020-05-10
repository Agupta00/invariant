import cv2
import numpy as np
import copy

class smt:
	def __init__(self):
		self.x=1
		self.y = {1:2,2:3}

	def __deepcopy__(self,memo):
		temp = copy.copy(self)
		temp.x = copy.deepcopy(self.x)
		return temp




def main():
	obj1 = smt()
	obj2 = copy.deepcopy(obj1)
	obj2.x=2
	obj2.y[1]=3

	print(obj1.x)
	print(obj1.y)
	print(obj2.y)

main()
xRange=yRange=5
import numpy as np

def neighbors(x,y,i=1,limit=1):
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

for a in neighbors(1,1,1,1):
	# img = np.zeros((xRange,xRange,3), np.uint8)
	print(a)
# a=neighbors(5,5)
# print(next(a))


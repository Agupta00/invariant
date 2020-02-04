# import cv2
# import numpy as np


# bw_image = np.eye(100, dtype=np.float32)
# cv2.imshow("Color Image",bw_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from skimage.exposure import rescale_intensity
import numpy as np
import argparse
# import cv2
import math
# from collections import defaultdict
import collections
import Tensor

# img = np.zeros((512,512,3), np.uint8)
# cv2.rectangle(img,(0,0),(510,128),(0,255,0),100)
# cv2.circle(img,(0,0), 100, (0,0,255), -1)
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("lalala", grayImage)
# k = cv2.waitKey(0) # 0==wait forever


# # img = np.zeros((512,512,3), np.float32)
# # cv2.rectangle(img,(0,0),(510,128),(0,255,0),100)
# # # cv2.circle(img,(447,63), 63, (0,0,255), -1)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # print(img.shape)

# out = [img[x][y] for x in range(100) for y in range(100) if img[x][y]!=0]
# print(out)

# img2=np.zeros((512,512,), np.uint8)
# img2[0][0]=55
# img2[0][1]=-1
# print(img2[0][0])

import networkx as nx




present = lambda key, inputs: 1 if key in inputs else 0


# def activate(inputs, norm=1/15, threshold=3/4):
#     act=0
#     # left side of the gaussian curve, from domain 0->1
#     p = lambda w: min(math.pow(math.e, (math.pi * -math.pow((w-1),2))),1)
#     dw = lambda w: -2*math.pi*(w-1)*p(w)

#     #return updated if activated, else keep old one
#     updated_input_dict=input_dict.copy()
#     #inputs is a set of type counter ie (type:number)
#     #input_dict=( (type:number),weight)
#     for key,w in input_dict.items():

#         if(present(key,inputs)):
#             print(key,present(key,inputs),"   ",w)
#             act+=present(key,inputs)*w
#             #probability of weight w
            
#             #w=w+1/norm*dw
#             updated_input_dict[key]=round(updated_input_dict[key]+norm*dw(w),4)
#         else:
#             print(key, norm*dw(w))
#             #decay weight
#             #w-= todo
#             updated_input_dict[key]=round(updated_input_dict[key]-norm*dw(w),4)


#     #normalize over avg weights and number of inputs
#     print("sum act",act)
#     # act=act*sum(input_dict.values())/(int(len(input_dict))**2)
#     act=round(act/sum(input_dict.values()),5)
#     print("weighted activation: ", act)



#     #add weights for present inputs who were not in input_dict to updated dict
#     if act>threshold:
#         for item in inputs:
#             if(item not in input_dict):
#                 # (a:2)=0.005 #(type:number)=initial weight
#                 updated_input_dict[item]=round(p(0),4)
#                 # self.input_dict=updated_input_dict
#                 print(updated_input_dict)
#         return True
#     else:
#         return False


def activate(inputs, norm=1/15, threshold=3/4, help=0):
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
        print("weighted activation: ", act)
    except:
        print("0 activation")


    #add weights for present inputs who were not in input_dict to updated dict
    if act>threshold:
        for item in inputs:
            if(item not in input_dict):
                # (a:2)=0.005 #(type:number)=initial weight
                updated_input_dict[item]=round(p(0),4)
                # self.input_dict=updated_input_dict
                print(updated_input_dict)
        return True
    else:
        return False



input_dict=dict()
input_dict2=dict()

counter = collections.Counter(['a','a','b'])
#makes counter hashable goes from a:2 -> (a,2)
inputs = frozenset(counter.items())
input_dict[('a',1)]=.2
inputs2 = frozenset({('a', 2), ('b', 1), ('c',1)})
for i,item in enumerate(inputs):
    input_dict[item]=1
    # input_dict2[item]=2
input_dict=dict()
print("input_dict",input_dict)
print(activate(inputs=inputs2,help=1))


# def get_pointers(inputs):
#     length=1
#     inputs.sort(key= lambda x: x.cm[0] + x.cm[1]*yRange)
#     # pointer_dict = defaultdict(list)  #list backed multidict
    
#     if(len(inputs)<2): 
#         # print("error only one input, for input", inputs)
#         return pointer_dict
#     #[a,b,c,d]
#     pointer_counter = collections.Counter()
#     for i in range(len(inputs)):
#         # delta x,y normalized by length
#         # =>( (a,b)(x,y) )
#         # (inputs[a].N,inputs[b].N)
#         pointer = lambda a,b: ( (inputs[a].N,inputs[b].N),
#             tuple(np.array([inputs[a].cm[0]-inputs[b].cm[0], inputs[a].cm[1]-inputs[b].cm[1]])/length) )

#         if i==0:
#             # which ti input,pointer to tj input -- where Neighboorhood determines identity
#             #a->b
#             # pointer_dict[(inputs[i].N,inputs[i+1].N)].append(pointer(i,i+1))
#             pointer_counter.update([pointer(i,i+1)])
#             #a->d
#             # pointer_dict[(inputs[i].N,inputs[-1].N)]=pointer(i,-1)
#         elif i==len(inputs)-1:
#             #d->c
#             # pointer_dict[(inputs[i].N,inputs[i-1].N)].append(pointer(i,i-1))
#             pointer_counter.update([pointer(i,i-1)])
#             #d->a
#             # pointer_dict[(inputs[i].N,inputs[0].N)]=pointer(i,0)
#         else:
#             #a<-b
#             # pointer_dict[(inputs[i].N,inputs[i-1].N)].append(pointer(i,i-1))
#             pointer_counter.update([pointer(i,i-1)])
#             #b->c
#             # pointer_dict[(inputs[i].N,inputs[i+1].N)].append(pointer(i,i+1))
#             pointer_counter.update([pointer(i,i+1)])
#     # print("pointers",pointer_dict)
#     return pointer_counter

# xRange=yRange=100
# data=Tensor.data
# # print(data.inputs[0])
# # print("\n\n\n\n",get_pointers(data.inputs))
# #for pointers
# input_dict=dict()
# pointer_counter=get_pointers(data.inputs)
# inputs=frozenset(pointer_counter)
# inputs2=frozenset({(((0, 0, 0), (0, 0, 0)), (-2.0, -2.0)), (((0, 0, 0), (0, 0, 0)), (2.0, 2.0))})


# for i,item in enumerate(inputs):
#     input_dict[item]=1

# print(inputs)
# print(activate(inputs=inputs2))


# ##
# # test for pointers
# def get_pointers(self,inputs):
#     inputs.sort(key= lambda x: x.cm[0] + x.cm[1]*yRange)
#     pointer_dict = defaultdict(list)  #list backed multidict
    
#     if(len(inputs)<2): 
#         # print("error only one input, for input", inputs)
#         return pointer_dict
#     #[a,b,c,d]
#     for i in range(len(inputs)):
#         # delta x,y normalized by length
#         pointer = lambda a,b: (np.array([inputs[a].cm[0]-inputs[b].cm[0], inputs[a].cm[1]-inputs[b].cm[1]])/self.length).tolist()
#         if i==0:
#             # which ti input,pointer to tj input -- where Neighboorhood determines identity
#             #a->b
#             pointer_dict[(inputs[i].N,inputs[i+1].N)].append(pointer(i,i+1))
#             #a->d
#             # pointer_dict[(inputs[i].N,inputs[-1].N)]=pointer(i,-1)
#         elif i==len(inputs)-1:
#             #d->c
#             pointer_dict[(inputs[i].N,inputs[i-1].N)].append(pointer(i,i-1))
#             #d->a
#             # pointer_dict[(inputs[i].N,inputs[0].N)]=pointer(i,0)
#         else:
#             #a<-b
#             pointer_dict[(inputs[i].N,inputs[i-1].N)].append(pointer(i,i-1))
#             #b->c
#             pointer_dict[(inputs[i].N,inputs[i+1].N)].append(pointer(i,i+1))
#     # print("pointers",pointer_dict)
#     return pointer_dict


#d is (a:2),w

        














#!/usr/bin/python3.5


import os
import time
import numpy as np
import sys
import pandas as pd







quat1=np.array([[1,2,3,4],[1,2,3,4]])
quat2=np.array([[5,6,7,8],[5,6,7,8]])
quatsec=quat1[:,0][:, None]*quat2[:,1:]+quat2[:,0][:, None]*quat1[:,1:] + np.cross(quat1[:,1:],quat2[:,1:])
quat_mult=np.array([quat1[:,0]*quat2[:,0]-np.sum(quat1[:,1:]*quat2[:,1:],axis=1),quatsec[:,0],quatsec[:,1],quatsec[:,2]]).T
print(quat_mult)

quat1=np.array([1,2,3,4])
quat2=np.array([5,6,7,8])
def Quat_multipl(quat1,quat2):
	vec1=np.array([quat1[1],quat1[2],quat1[3]])
	vec2=np.array([quat2[1],quat2[2],quat2[3]])
	quatsec=quat1[0]*vec2+quat2[0]*vec1+np.cross(vec1,vec2)
	quat_mult=np.array([quat1[0]*quat2[0]-np.dot(vec1,vec2),quatsec[0],quatsec[1],quatsec[2]])
	return(quat_mult)
	
quat_mult=Quat_multipl(quat1,quat2)
print(quat_mult)

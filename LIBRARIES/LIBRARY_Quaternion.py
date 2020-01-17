import sys 
sys.path.append("./LIBRARIES/")
import LIBRARY_Vector_calculus as Vect_calc
import numpy as np
import math
from scipy.spatial import ConvexHull


def Quat_inv(quaternion):
	quaternion_inv=np.array([quaternion[0],-1*quaternion[1],-1*quaternion[2],-1*quaternion[3]])
	return(quaternion_inv)

def Quat_multipl(quat1,quat2):
	vec1=np.array([quat1[1],quat1[2],quat1[3]])
	vec2=np.array([quat2[1],quat2[2],quat2[3]])
	quatsec=quat1[0]*vec2+quat2[0]*vec1+np.cross(vec1,vec2)
	quat_mult=np.array([quat1[0]*quat2[0]-np.dot(vec1,vec2),quatsec[0],quatsec[1],quatsec[2]])
	return(quat_mult)
	
def Quat_triple_prod(quat1,vector,quat2):
	vec_quat=np.array([0,vector[0],vector[1],vector[2]])
	quat_trip=Quat_multipl(quat1,Quat_multipl(vec_quat,quat2))
	vec_trip=np.array([quat_trip[1],quat_trip[2],quat_trip[3]])
	return(vec_trip)

def Quat_inv_arr(quaternion):
	quaternion[:,1]=-1*quaternion[:,1]
	quaternion[:,2]=-1*quaternion[:,2]
	quaternion[:,3]=-1*quaternion[:,3]
	return(quaternion)

#def Quat_multipl_arr(quat1,quat2):
	#vec1=np.array([quat1[1],quat1[2],quat1[3]])
	#vec2=np.array([quat2[1],quat2[2],quat2[3]])
	#quatsec=quat1[:,0]*quat2[:,1:3]+quat2[:,0]*quat1[:,1:3]+np.cross(quat1[:,1:3],quat2[:,1:3])
	#quat_mult=np.array([quat1[0]*quat2[0]-np.dot(vec1,vec2),quatsec[0],quatsec[1],quatsec[2]])
	#return(quat_mult)
	
#def Quat_triple_prod_arr(quat1,vector,quat2):
	#vec_quat=np.array([0,vector[0],vector[1],vector[2]])
	#quat_trip=Quat_multipl(quat1,Quat_multipl(vec_quat,quat2))
	#vec_trip=np.array([quat_trip[1],quat_trip[2],quat_trip[3]])
	#return(vec_trip)

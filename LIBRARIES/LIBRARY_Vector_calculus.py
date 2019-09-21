import numpy as np


def Normalvector_plane(V1,V2):
	norm=np.cross(V1,V2)
	normdist=np.sqrt(norm[0]*norm[0]+norm[1]*norm[1]+norm[2]*norm[2])
	return(norm/normdist)
	
def Length_vector(A):
	length=np.sqrt(A[0]*A[0]+A[1]*A[1]+A[2]*A[2])
	return(length)

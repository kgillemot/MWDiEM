#!/usr/bin/python3.5
#GJK algorithm
import sys 
import numpy as np
import random
import math
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from matplotlib import pyplot
from mpl_toolkits import mplot3d
import functions as func
import time


def Convex_polygon_maker_2D():
	####nNOT USED IN OBJECT ORIENTED VERSION
	numsides=np.random.randint(3,10)
	polyhedra=np.random.rand(numsides,3)*100
	polygon=Shape_to_plain(polyhedra)
	hull= ConvexHull(polygon)
	hull_indices = hull.vertices
	A=[]
	for i in hull.vertices:
		A.append([polygon[i,0],polygon[i,1],0])
	A=np.matrix(A)
	return(A)

def Convex_polygon_maker_3D(numcornersmin,numcornersmax,sizebox,maxparticlesize):
	####nNOT USED IN OBJECT ORIENTED VERSION
	numcorners=np.random.randint(numcornersmin,numcornersmax)
	radius_int=np.random.randint(sizemin,sixemax)
	radius_float=np.random.rand(1)
	radius=radius_int+radius_float
	fi=np.random.rand(numcorners)*180
	teta=np.random.rand(numcorners)*360*0.9999999999999
	fi_rad=fi*math.pi/180
	teta_rad=teta*math.pi/180
	polyhedra=[radius*math.sin(fi_rad)*math.cos(teta_rad),radius*math.sin(fi_rad)*math.sin(teta_rad),radius*math.cos(fi_rad)]
	
	
	#polyhedra=np.random.rand(numcorners,3)*maxparticlesize
	transl=np.random.rand(3)*(sizebox-maxparticlesize)
	polyhedra=polyhedra+transl
	hull= ConvexHull(polyhedra)
	volume = ConvexHull(polyhedra).volume
	numvertices=len(hull.vertices)
	numsimplices=len(hull.simplices)
	hull_indices = hull.vertices
	A=[]
	#ebben haromszogek sarkai vannak 3-as kupacokban
	Asimp=[]
	for i in hull.vertices:
		A.append([polyhedra[i,0],polyhedra[i,1],polyhedra[i,2]])
	for i in hull.simplices:
		Asimp.append(polyhedra[i])
	A=np.matrix(A)
	Asimp=np.array(Asimp)
	return(A,Asimp,hull.simplices,volume,numvertices,numsimplices)
	

#valeszeg ez is tok lassu
def Order_polygon(A):
	base=[1,0,0]
	N=A.shape[0]
	angle=np.zeros((N))
	A_sorted=np.zeros(A.shape)
	for i in range(0,N):
		norm_A=math.sqrt(A[i,0]**2+A[i,1]**2+A[i,2]**2)
		angle[i]=np.dot(A[i],base)/norm_A
	indices_sorted=np.lexsort((angle,))
	for i in range(0,N):
		A_sorted[i]=A[indices_sorted[i]]
	return(A_sorted)
	
	

def Shape_to_plain(A):
	N=A.shape[0]
	A_plain=np.zeros((N,2))
	for i in range(0,N):
		A_plain[i,0]=A[i,0]
		A_plain[i,1]=A[i,1]
	return(A_plain)

def Center_of_mass(A):
	N=len(A)
	#N=A.shape[0]
	Cm=np.zeros((3))
	Asum0=0
	Asum1=0
	Asum2=0
	for i in range(0,N):
		#Asum0=Asum0+A[i,0]
		Asum0=Asum0+A[i].vertex_coo[0]
		Asum1=Asum1+A[i].vertex_coo[1]
		Asum2=Asum2+A[i].vertex_coo[2]
	Cm[0]=Asum0/N
	Cm[1]=Asum1/N
	Cm[2]=Asum2/N
	return(Cm)

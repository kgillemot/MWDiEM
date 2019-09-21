import sys 
sys.path.append("./LIBRARIES/")
import LIBRARY_Fine_contact_detection as Fine_contact

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import LIBRARY_Fine_contact_detection as Fine_contact

#from matplotlib import pyplot
#from mpl_toolkits import mplot3d
import time


def Minkowski_difference_plot_2D(A,B):
	mink=Minkowski_difference(A,B)
	mink_2D=Shape_to_plain(mink)
	hull= ConvexHull(mink_2D)
	plt.plot(mink_2D[:,0], mink_2D[:,1], 'o')
	for simplex in hull.simplices:
		plt.plot(mink_2D[simplex, 0], mink_2D[simplex, 1], 'k-')
	plt.show()
	
def Minkowski_difference_plot_3D(A,B,P): #,points):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	mink=Fine_contact.Minkowski_difference(A,B)
	hullA= ConvexHull(mink)
	xA=[]
	yA=[]
	zA=[]
	for i in hullA.vertices:
		xA.append(mink[i,0])
		yA.append(mink[i,1])
		zA.append(mink[i,2])
	ax.plot(xA,yA,zA,"ko")
	mink=np.array(mink)
	for s in hullA.simplices:
		s = np.append(s, s[0])  # Here we cycle back to the first coordinate		
		ax.plot(mink[s, 0], mink[s, 1], mink[s, 2], "r-")
	#hullB= ConvexHull(points)
	#xA2=[]
	#yA2=[]
	#zA2=[]
	#for i in hullB.vertices:
		#xA2.append(points[i,0])
		#yA2.append(points[i,1])
		#zA2.append(points[i,2])
	#ax.plot(xA2,yA2,zA2,"ko")

	#for s in hullB.simplices:
		#s = np.append(s, s[0])  # Here we cycle back to the first coordinate		
		#ax.plot(points[s, 0], points[s, 1], points[s, 2], "b-")
	origin=[0,0,0]
	ax.plot([0],[0],[0],"ko")
	ax.plot([P[0]],[P[1]],[P[2]],"ko")
	#ax.plot([S1[0],S2[0],S3[0],S4[0]],[S1[1],S2[1],S3[1],S4[1]],[S1[2],S2[2],S3[2],S4[2]],"go")
	pontocskak = np.linspace(-3, 3, 1000)
	nulla= [0] * 1000
	ax.plot(pontocskak, nulla, nulla,"b-")
	ax.plot(nulla,pontocskak, nulla,"b-")
	ax.plot(nulla,nulla,pontocskak, "b-")
	plt.xlabel('XXXXXXXXXX')
	plt.ylabel('YYYYYYYYYY')
	plt.show()
	
	#hull= ConvexHull(mink)
	#x=[]
	#y=[]
	#z=[]
	#for i in range(0,mink.shape[0]):
		#x.append(mink[i,0])
		#y.append(mink[i,1])
		#z.append(mink[i,2])
	#tupleList = list(zip(x, y, z))
	#verts=[]
	#for ix in range(0,hull.simplices.shape[0]):
		#tmpverts=[]
		#for iy in range(0,hull.simplices.shape[1]):
			#tmpverts.append(tupleList[hull.simplices[ix][iy]])
		#verts.append(tmpverts)	
	#ax.plot([0],[0],[0],"ko")
	#poly = Poly3DCollection(verts,alpha=1)
	#ax.add_collection3d(poly)
	#ax.set_xlim([-5,5])
	#ax.set_ylim([-5,5])
	#ax.set_zlim([-5,5])

	#plt.show()

	
	
	
def Polygons_plot_2D(A,B,fout):
	#az eredeti polygonok plottolasa
	A_2D=Shape_to_plain(A)
	B_2D=Shape_to_plain(B)
	fig, ax = plt.subplots()
	polygon1=Polygon(A_2D,True)
	polygon2=Polygon(B_2D,True)
	patches=[]
	patches.append(polygon1)
	patches.append(polygon2)
	p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
	colors = 100*np.random.rand(len(patches))
	p.set_array(np.array(colors))
	ax.add_collection(p)
	ax.autoscale_view()
	plt.show()
	#plt.savefig(fout)
	#time.sleep(1)

def Polygons_plot_3D(A,B,point):
	point=np.array(point)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	hullA= ConvexHull(A)
	hullB= ConvexHull(B)
	xA=[]
	yA=[]
	zA=[]
	xB=[]
	yB=[]
	zB=[]
	for i in hullA.vertices:
		xA.append(A[i,0])
		yA.append(A[i,1])
		zA.append(A[i,2])
	for i in hullB.vertices:
		xB.append(B[i,0])
		yB.append(B[i,1])
		zB.append(B[i,2])
	ax.plot(xA,yA,zA,"ko")
	ax.plot(xB,yB,zB,"ko")
	xx=[point[0,0]]
	yy=[point[0,1]]
	zz=[point[0,2]]
	ax.plot(xx,yy,zz,"ko")
	A=np.array(A)
	B=np.array(B)
	for s in hullA.simplices:
		s = np.append(s, s[0])  # Here we cycle back to the first coordinate		
		ax.plot(A[s, 0], A[s, 1], A[s, 2], "r-")
	for s in hullB.simplices:
		s = np.append(s, s[0])  # Here we cycle back to the first coordinate		
		ax.plot(B[s, 0], B[s, 1], B[s, 2], "b-")
	#plt.title(fout)
	plt.show()
	
	
def Polygon_plot_3D(A,fout):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	hull= ConvexHull(A)
	x=[]
	y=[]
	z=[]
	for i in range(0,A.shape[0]):
		x.append(A[i,0])
		y.append(A[i,1])
		z.append(A[i,2])
	tupleList = list(zip(x, y, z))
	verts=[]
	for ix in range(0,hull.simplices.shape[0]):
		tmpverts=[]
		for iy in range(0,hull.simplices.shape[1]):
			tmpverts.append(tupleList[hull.simplices[ix][iy]])
		verts.append(tmpverts)	
	poly = Poly3DCollection(verts)
	ax.set_xlim([-5,5])
	ax.set_ylim([-5,5])
	ax.set_zlim([-5,5])
	plt.xlabel('XXXXXXXXXX')
	plt.ylabel('YYYYYYYYYY')
	ax.spines['left'].set_position('center')
	ax.spines['bottom'].set_position('center')
	pontocskak = np.linspace(-100, 100, 1000)
	nulla= [0] * 1000
	ax.plot(pontocskak, nulla, nulla,"r-")
	ax.plot(nulla,pontocskak, nulla,"r-")
	ax.plot(nulla,nulla,pontocskak, "r-")
	
	
	ax.add_collection3d(poly)
	ax.view_init(elev=0, azim=0)
	plt.title(fout+"A.png")
	plt.savefig(fout+"A.png")
	ax.view_init(elev=0, azim=90)
	plt.title(fout+"B.png")
	plt.savefig(fout+"B.png")
	ax.view_init(elev=90, azim=0)
	plt.title(fout+"C.png")
	plt.savefig(fout+"C.png")
	ax.view_init(elev=45, azim=45)
	plt.title(fout+"D.png")
	plt.savefig(fout+"D.png")
	ax.view_init(elev=0, azim=45)
	plt.title(fout+"E.png")
	plt.savefig(fout+"E.png")
	ax.view_init(elev=45, azim=0)
	plt.title(fout+"F.png")
	plt.savefig(fout+"F.png")
	ax.view_init(elev=0, azim=0)
	plt.title(fout+"AA.png")
	plt.savefig(fout+"AA.png")
	ax.view_init(elev=0, azim=-90)
	plt.title(fout+"BB.png")
	plt.savefig(fout+"BB.png")
	ax.view_init(elev=-90, azim=0)
	plt.title(fout+"CC.png")
	plt.savefig(fout+"CC.png")
	ax.view_init(elev=-45, azim=-45)
	plt.title(fout+"DD.png")
	plt.savefig(fout+"DD.png")
	ax.view_init(elev=0, azim=-45)
	plt.title(fout+"EE.png")
	plt.savefig(fout+"EE.png")
	ax.view_init(elev=-45, azim=0)
	plt.title(fout+"FF.png")
	plt.savefig(fout+"FF.png")
	time.sleep(1)
	plt.show()

#tobb poligont 	
def Polygons_plot_3D_multi(A,point1, point2):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	#print(A)
	for i in range(0,len(A)):
		vertices2=[]
		for j in range(0,A[i].num_vertices):
			vertices2.append(A[i].vertices[j].vertex_coo)
		hull=ConvexHull(vertices2)
		x=[]
		y=[]
		z=[]		
		for j in range(0,A[i].num_vertices):
			#print(A[i][j,0])
			x.append(A[i].vertices[j].vertex_coo[0])
			y.append(A[i].vertices[j].vertex_coo[1])
			z.append(A[i].vertices[j].vertex_coo[2])
		tupleList = list(zip(x, y, z))
		verts=[]
		for ix in range(0,hull.simplices.shape[0]):
			tmpverts=[]
			for iy in range(0,hull.simplices.shape[1]):
				tmpverts.append(tupleList[hull.simplices[ix][iy]])
			verts.append(tmpverts)	
		poly = Poly3DCollection(verts, cmap=matplotlib.cm.jet, alpha=0.4)
	
		ax.add_collection3d(poly)
		colors = 100*np.random.rand(len(A))
		poly.set_array(np.array(colors))
		ax.add_collection(poly)
	ax.plot([point1[0]],[point1[1]],[point1[2]],"ko")
	ax.plot([point2[0]],[point2[1]],[point2[2]],"ko")
	ax.set_xlim([-5,5])
	ax.set_ylim([-5,5])
	ax.set_zlim([-5,5])
	ax.view_init(elev=0, azim=0)
	plt.xlabel('XXXXXXXXXX')
	plt.ylabel('YYYYYYYYYY')
	plt.show()
	
def Plot_particles(xcell,ycell,zcell,partnumcell):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	pontocskak = np.linspace(-3, 3, 1000)
	for i in range(0,3):
		for j in range(0,3):
			for k in range(0,3):
				ii=[i]*1000
				jj=[j]*1000
				kk=[k]*1000
				ax.plot(pontocskak, jj, kk,"b-")
				ax.plot(ii,pontocskak, kk,"b-")
				ax.plot(ii,jj,pontocskak, "b-")
	ax.set_xlim([0,3])
	ax.set_ylim([0,3])
	ax.set_zlim([0,3])
	for i in range(0,len(xcell)):
		ax.text(xcell[i],ycell[i],zcell[i], partnumcell[i], color="black") 
	
	ax.scatter(xcell, ycell, zcell, c='r', marker='o')
	plt.show()
	
	

	
def Plot_vectors(Aorig,A):
	#Aorig is the position of the vector, and A is the vector itself.
	from mpl_toolkits.mplot3d import Axes3D

	origin = [0,0,0]
	X=Aorig[:,0]
	Y=Aorig[:,1]
	Z=Aorig[:,2] 
	#X, Y, Z = zip(origin,origin,origin)
	A=np.matrix(A)
	
	U=A[:,0]
	V=A[:,1]
	W=A[:,2]
	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')
	ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.1,)
	plt.show()


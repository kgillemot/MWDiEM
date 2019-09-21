import sys 
sys.path.append("./LIBRARIES/")
import LIBRARY_Fine_contact_detection as Fine_contact
import LIBRARY_Misc as Misc
import LIBRARY_Vector_calculus as Vect_calc
import LIBRARY_Plotting as Plot

import numpy as np
from scipy.spatial import ConvexHull



def Support_function(particle_A,particle_B,d):
	#### WORKS AND IS OBJECT ORIENTED
	N=particle_A.num_vertices
	M=particle_B.num_vertices
	dist1=np.zeros((N))
	dist2=np.zeros((M))
	for i in range(0,N):
		dist1[i]=d[0]*particle_A.vertices[i].vertex_coo[0]+d[1]*particle_A.vertices[i].vertex_coo[1]+d[2]*particle_A.vertices[i].vertex_coo[2]
	a=Misc.array_min_max_finder(dist1)[3]
	for i in range(0,M):
		dist2[i]=-d[0]*particle_B.vertices[i].vertex_coo[0]-d[1]*particle_B.vertices[i].vertex_coo[1]-d[2]*particle_B.vertices[i].vertex_coo[2]
	b=Misc.array_min_max_finder(dist2)[3]
	diff=np.array([particle_A.vertices[a].vertex_coo[0]-particle_B.vertices[b].vertex_coo[0],particle_A.vertices[a].vertex_coo[1]-particle_B.vertices[b].vertex_coo[1],particle_A.vertices[a].vertex_coo[2]-particle_B.vertices[b].vertex_coo[2]])
	return(diff,a,b)
	
def Minkowski_difference(particle_A,particle_B):
	#WORKS AND IS OBJECT ORIENTED
	N=particle_A.num_vertices
	M=particle_B.num_vertices
	mink=np.zeros((N*M,3))
	k=0
	for i in range(0,N):
		for j in range(0,M):
			mink[k]=particle_A.vertices[i].vertex_coo-particle_B.vertices[j].vertex_coo
			k=k+1
	return(mink)



def Check_linesegment(P1,P2):
	#P2 is closer to the origin
	#check if the Voronoi region outside the slab has the origin
	coll=-1
	d=0
	V21=P2-P1
	#check which Voronoi region has the origin
	if np.dot(V21,P2)<0:
		coll=0
		#print("Linesegment Voronoi Region 1, No Collision","\n")
		return(coll,d)	
		############if np.dot(V21,P1)>0:
		############coll=0
		#############print("Linesegment Voronoi Region 3, No Collision","\n")
		############return(coll,d)	
	#check if the line itself contains the origin
	test=np.cross(V21,P2)
	if test[0]==0 and test[1]==0 and test[2]==0:
		coll=1
		#print("Linesegment, the line itself contains the origin, Collision\n")
		return(coll,d)
	d=np.cross(np.cross(V21,-P2),V21) 
	#the origin is in Region 2, but not on the line
	#print("Linesegment, the origin in in Voronoi Region 2, TBC\n")
	return(coll,d)
		
		
def Check_triangle(P1,P2,P3):
	#P3 is closest to the origin
	coll=-1
	d=0
	#check if it's outside of the possibble Voronoi region
	#####(coll1,d)=Check_linesegment(P1,P2)
	#####if coll1==1:
		#####coll=1
		#####return(coll,d)
	(coll2,d)=Check_linesegment(P2,P3)
	if coll2==1:
		coll=1
		return(coll,d)
	(coll3,d)=Check_linesegment(P1,P3)    #########(P3,P1)
	if coll3==1:
		coll=1
		return(coll,d)
	if coll2+coll3>-1:            ######## >-2   +coll1
		coll=0
		return(coll,d)	
	V21=P2-P1
	V32=P3-P2
	V13=P1-P3
	n=np.cross(V32,V13)
	######n1=np.cross(V21,n)
	n2=np.cross(V32,n)
	n3=np.cross(V13,n)
	#check which Voronoi region is it
	#Region 1 outsode of line 1
	##########if np.dot(n1,P1)<0:
		###########d=np.cross(np.cross(V13,P3),-V13)
		##########d=n1
		###########print("Triangle Voronoi regio 1")
		##########return(coll,d)
	#Region 2 outside of line 2
	if np.dot(n2,P2)<0:
		#d=np.cross(np.cross(V32,P3),-V32)
		d=n2
		#print("Triangle Voronoi Region 2")
		return(coll,d)	
	#Region 3 outside of line 3
	if np.dot(n3,P3)<0:
		#d=np.cross(np.cross(V32,P3),-V32)
		d=n3
		#print("Triangle Voronoi Region 3")
		return(coll,d)	
	#Inside the triangle
	if np.dot(P3,n)==0:
		coll=1
		#print("Triangle, inside the Triangle")
		return(coll,d)	
	#Above or below the triangle
	if np.dot(n,P3)>0:
		d=-n
	else:
		d=n
	#print("Triangle Voronoi regions above or below the triangle")
	return(coll,d)
	
	
	
def Check_Tetrahedron(P1,P2,P3,P4):
	#print("starttetra")
	#P4 is closest to the origin
	coll=-1
	d=0
	todel=0
	(coll1,d)=Check_triangle(P1,P2,P4)
	if coll1==1:
		coll=1
		return(coll,d,todel)
	(coll2,d)=Check_triangle(P2,P3,P4)
	if coll2==1:
		coll=1
		return(coll,d,todel)
	(coll3,d)=Check_triangle(P3,P1,P4)
	if coll3==1:
		coll=1
		return(coll,d,todel)
		##########(coll4,d)=Check_triangle(P1,P2,P3)
		##########if coll4==1:
		##########coll=1
		##########return(coll,d,todel)
	if coll1+coll2+coll3>-2:     ####coll 4 >-3
		coll=0
		return(coll,d,todel)	
	V13=P1-P3
	V21=P2-P1
	V32=P3-P2	
	V41=P4-P1
	n2=np.cross(V21,V41)
	V42=P4-P2
	n3=np.cross(V32,V42)
	V43=P4-P3	
	n4=np.cross(V13,V43)
	#check which Voronoi region is it in
	if np.dot(n2,P4)<0:
		d=n2
		todel=3
	elif np.dot(n3,P4)<0:
		d=n3
		todel=1
	elif np.dot(n4,P4)<0:
		d=n4
		todel=2
	else:
		coll=1
	return(coll,d,todel)		


def GJKuj_algorithm(particle_A,particle_B,aa,bb):
	#WORKS AND IS OBJECT ORIENTED but needs to be further checked, and maybe the "while" cycle is useless in it
	#ha tesztelni akarom csak ki kell plotolni a Minkovski differencet es az origot
	#https://caseymuratori.com/blog_0003
	#https://pybullet.org/Bullet/phpBB3/search.php?st=0&sk=t&sd=d&sr=posts&keywords=gjk&start=30
	A=particle_A
	B=particle_B
	#is there a collision or not
	coll=-1

	#Center of mass
	CmA=particle_A.center_of_mass
	CmB=particle_B.center_of_mass



	#1. Single point
	#print("Single point")
	d1=CmB-CmA
	(S1,aa1,bb1)=Fine_contact.Support_function(A,B,d1)
	
	#check if the new point crosses the origin or not
	if np.dot(d1,S1)<0:
		#ez nagy kerdes, hogy benne maradhat e:
		#print(str(aa)+" "+str(bb)+" NO COLLISION Point SAT\n	")
		S2=[0,0,0]
		aa2=0
		bb2=0
		S3=[0,0,0]
		aa3=0
		bb3=0
		S4=[0,0,0]
		aa4=0
		bb4=0
		coll=0
		return(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,coll)
		#check if the new point is the origin
		#########if S1[0]==0 and S1[1]==0 and S1[2]==0:
		#########print(str(aa)+" "+str(bb)+" COLLISION Point Hitting Origin\n	")	
		#########S2=[0,0,0]
		#########S3=[0,0,0]
		#########S4=[0,0,0]
		#########coll=1
		#########return(S1,S2,S3,S4,coll)

	

	#2. Line segment	
	#print("Line segment")
	d2=-1*d1	

	(S2,aa2,bb2)=Fine_contact.Support_function(A,B,d2)	
	#check if the new point crosses the origin or not
	if np.dot(d2,S2)<0:
		#print(str(aa)+" "+str(bb)+" NO COLLISION Line SAT\n	")
		S3=[0,0,0]
		aa3=0
		bb3=0
		S4=[0,0,0]
		aa4=0
		bb4=0
		coll=0
		return(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,coll)
		#########check if the new point is the origin
		########if S2[0]==0 and S2[1]==0 and S2[2]==0:
		########print(str(aa)+" "+str(bb)+" COLLISION Line Hitting Origin\n	")
		########S3=[0,0,0]
		########S4=[0,0,0]
		########coll=1
		########return(S1,S2,S3,S4,coll)
	(coll,d3)=Check_linesegment(S1,S2)
	if coll==0:
		#print(str(aa)+" "+str(bb)+" NO COLLISION Line Voronoi\n	")
		S3=[0,0,0]
		aa3=0
		bb3=0
		S4=[0,0,0]
		aa4=0
		bb4=0
		coll=0
		return(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,coll)
		#########if coll==1:
		#########print(str(aa)+" "+str(bb)+" COLLISION Line Voronoi\n	")
		#########S3=[0,0,0]
		#########S4=[0,0,0]
		#########coll=1
		#########return(S1,S2,S3,S4,coll)
	
	
	#3. Triangle
	(S3,aa3,bb3)=Fine_contact.Support_function(A,B,d3)
	#check if the new point crosses the origin or not
	if np.dot(d3,S3)<0:
		#print(str(aa)+" "+str(bb)+" NO COLLISION Triangle SAT\n")
		S4=[0,0,0]
		aa4=0
		bb4=0
		coll=0
		return(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,coll)
	#check if the new point is the origin
		############if S3[0]==0 and S3[1]==0 and S3[2]==0:
		############print(str(aa)+" "+str(bb)+" COLLISION Triangle Hitting origin\n	")
		############S4=[0,0,0]
		############coll=1
		############return(S1,S2,S3,S4,coll)
	(coll,d4)=Check_triangle(S1,S2,S3)
	if coll==0:
		#print(str(aa)+" "+str(bb)+" NO COLLISION Triangele Voronoi\n	")
		S4=[0,0,0]
		coll=0
		return(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,coll)
		################if coll==1:
		################print(str(aa)+" "+str(bb)+" COLLISION Triangle Voronoi\n	")
		################S4=[0,0,0]
		################coll=1
		################return(S1,S2,S3,S4,coll)

	

	#4. Tetrahedron
	zz=0
	while (zz<100):
		#print("Iter Tetrahedron ",zz,"\n")
		(S4,aa4,bb4)=Fine_contact.Support_function(A,B,d4)

		#check if the new point crosses the origin or not
		if np.dot(d4,S4)<0:
			#print(str(aa)+" "+str(bb)+" NO COLLISION Tetra SAT\n")
			coll=0
			return(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,coll)
		#check if the new point is the origin
		if S4[0]==0 and S4[1]==0 and S4[2]==0:
			#print(str(aa)+" "+str(bb)+" COLLISION Tetra Hitting origin\n	")
			coll=1
			change=CorrectFlat(S1,S2,S3,S4)
			if change==1:
				d4=np.cross(S1,S2)
				(S4,aa4,bb4)=Fine_contact.Support_function(A,B,d4)
			return(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,coll)
		(coll,d4,todel)=Check_Tetrahedron(S1,S2,S3,S4)
		if coll==0:
			#print(str(aa)+" "+str(bb)+" NO COLLISION Tetra Voronoi\n	")
			coll=0
			return(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,coll)
		if coll==1:
			#print(str(aa)+" "+str(bb)+" COLLISION Tetra Vornoi\n	")
			coll=1
			change=CorrectFlat(S1,S2,S3,S4)
			if change==1:
				d4=np.cross(S1,S2)
				(S4,aa4,bb4)=Fine_contact.Support_function(A,B,d4)
			return(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,coll)
		if todel==1:
			S1uj=S2
			aa1uj=aa2
			bb1uj=bb2
			S2uj=S3
			aa2uj=aa3
			bb2uj=bb3
			S3uj=S4
			aa3uj=aa4
			bb3uj=bb4
		if todel==2:
			S1uj=S3
			aa1uj=aa3
			bb1uj=bb3
			S2uj=S1
			aa2uj=aa1
			bb2uj=bb1
			S3uj=S4
			aa3uj=aa4
			bb3uj=bb4
		if todel==3:
			S1uj=S1
			aa1uj=aa1
			bb1uj=bb1
			S2uj=S2
			aa2uj=aa2
			bb2uj=bb2
			S3uj=S4
			aa3uj=aa4
			bb3uj=bb4
		if todel==4:
			S1uj=S1
			aa1uj=aa1
			bb1uj=bb1
			S2uj=S2
			aa2uj=aa2
			bb2uj=bb2
			S3uj=S3
			aa3uj=aa3
			bb3uj=bb3
		del S1,S2,S3, aa1, aa2, aa3, bb1, bb2, bb3
		S1=S1uj
		aa1=aa1uj
		bb1=bb1uj
		S2=S2uj
		aa2=aa2uj
		bb2=bb2uj
		S3=S3uj
		aa3=aa3uj
		bb3=bb3uj
		zz=zz+1	
	return(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,coll)






def EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,particle_A,particle_B):
	#ugy fest JO es OBJECT ORIENTED MAR
	#meg hianyzik egy ket biztonsagi step: https://pybullet.org/Bullet/phpBB3/viewtopic.php?f=4&t=2931
	#ezen kivul nincs benne asszem, h mi tortenik, ha a tetrahedron nem teljes
	#valamint valszeg csak polihedronokra mukszik
	#https://pybullet.org/Bullet/phpBB3/viewtopic.php?f=4&t=2931
	#https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=3322
	#http://allenchou.net/2013/12/game-physics-contact-generation-epa/
	#https://stackoverflow.com/questions/31764305/im-implementing-the-expanding-polytope-algorithm-and-i-am-unsure-how-to-deduce
	#https://groups.google.com/forum/#!msg/comp.graphics.algorithms/tjFsExG3JqA/yYTOvSEF_NgJ
	#https://books.google.hu/books?id=bbqdWHCZ9IYC&pg=PA197&lpg=PA197&dq=how+to+get+support+point+in+minkowsky+difference&source=bl&ots=XmBPDTxwcY&sig=ACfU3U3X0JtFpVPB-onNXBnqJl-8Su2HbQ&hl=en&sa=X&ved=2ahUKEwir5p3oy-3gAhXtposKHeQOAwwQ6AEwAXoECAgQAQ#v=onepage&q=how%20to%20get%20support%20point%20in%20minkowsky%20difference&f=false
	#http://hacktank.net/blog/?p=119
	#http://www.dyn4j.org/2010/05/epa-expanding-polytope-algorithm/
	#https://wildbunny.co.uk/blog/2011/04/20/collision-detection-for-dummies/
	#http://mathworld.wolfram.com/BarycentricCoordinates.html
	#https://code.google.com/archive/p/box2d/downloads
	#http://allenchou.net/2013/12/game-physics-contact-generation-epa/
	#Book: Real-Time Collision Detection
	xmin_old=1000000000
	Suj_proj_len=xmin_old+1
	counter=0
	
	vertices=np.array([S1,S2,S3,S4])
	verticesnum=np.array([[aa1,bb1],[aa2,bb2],[aa3,bb3],[aa4,bb4]])
	while(xmin_old<Suj_proj_len):	
		hull= ConvexHull(vertices)	

		verticeshull=[]
		numverticeshull=[]
		for j in range(0,len(hull.vertices)):
			verticeshull.append(vertices[hull.vertices[j]])
			numverticeshull.append(verticesnum[hull.vertices[j]])
			

		normalvectors=[]
		distances=[]
		onepoints=[]
		for i in range(0,len(hull.simplices)):
			V1=vertices[hull.simplices[i][1]]-vertices[hull.simplices[i][0]]
			V2=vertices[hull.simplices[i][2]]-vertices[hull.simplices[i][1]]
			V3=vertices[hull.simplices[i][0]]-vertices[hull.simplices[i][2]]
			normtemp=Vect_calc.Normalvector_plane(V1,V2)
			dist=np.dot(normtemp,vertices[hull.simplices[i][1]])
			if dist<0:
				normtemp=-normtemp
			normalvectors.append(normtemp)
			onepoints.append([vertices[hull.simplices[i][0]],vertices[hull.simplices[i][1]],vertices[hull.simplices[i][2]]])
			
			#check if the projection of the normal vector is inside the triangle or not
			#http://blackpawn.com/texts/pointinpoly/default.html
			normpoint=abs(dist)*normtemp
			Vn1=normpoint-vertices[hull.simplices[i][0]]
			Vn2=normpoint-vertices[hull.simplices[i][1]]
			Vn3=normpoint-vertices[hull.simplices[i][2]]
			if np.dot(np.cross(V1,Vn1),np.cross(V1,-V3))>0:
				if np.dot(np.cross(V2,Vn2),np.cross(V2,-V1))>0:
					if np.dot(np.cross(V3,Vn3),np.cross(V3,-V2))>0:
						dist=dist
					else:
						dist=1000000000
				else:
					dist=1000000000
			else:
				dist=1000000000
			distances.append(abs(dist))	

		(xmin_new,xmax_new,indmin,indmax)=Misc.array_min_max_finder(distances)
		xmin_old=xmin_new	


		(Suj,aauj,bbuj)=Fine_contact.Support_function(particle_A,particle_B,normalvectors[indmin])  #uj pont
		Suj_proj_len=np.dot(normalvectors[indmin],Suj)

		vertices=np.concatenate((verticeshull,[Suj]),axis=0)
		verticesnum=np.concatenate((numverticeshull,[[aauj,bbuj]]),axis=0)
		if xmin_old>Suj_proj_len or (xmin_old*1.000000000001>Suj_proj_len and xmin_old*0.99999999999<Suj_proj_len):
			break
		else:
			counter=counter+1
	pen_depth=xmin_new
	#print(pen_depth)
	
	
	##Testing part
	##here I directly check on the Minkovsky difference the closes point
	#mink=Minkowski_difference(particle_A,particle_B)
	#hull= ConvexHull(mink)
	#verticeshull=[]
	#for j in range(0,len(hull.vertices)):
		#verticeshull.append(mink[hull.vertices[j]])

		#normalvectors2=[]
		#distances=[]
		#onepoints=[]
		#for i in range(0,len(hull.simplices)):
			#V1=mink[hull.simplices[i][1]]-mink[hull.simplices[i][0]]
			#V2=mink[hull.simplices[i][2]]-mink[hull.simplices[i][1]]
			#normtemp=Vect_calc.Normalvector_plane(V1,V2)
			#dist=np.dot(normtemp,mink[hull.simplices[i][1]])
			#if dist<0:
				#normtemp=-normtemp
			#normalvectors2.append(normtemp)
			#onepoints.append(mink[hull.simplices[i][1]])
			#distances.append(abs(dist))	
	#(xmin_new2,xmax_new2,indmin2,indmax2)=Misc.array_min_max_finder(distances)
	##print(xmin_new2)
	##print(normalvectors2[indmin2])	
	#if pen_depth==xmin_new2:
		#print("Yupppi, it works, the minimal distance is correct")


	#the corners of the closest triangle on the Minkowsky difference
	a=hull.simplices[indmin][0]
	b=hull.simplices[indmin][1]
	c=hull.simplices[indmin][2]

	
	#Now get the contact information, project the normalvector onto the triangle
	P=pen_depth*normalvectors[indmin]

	A=onepoints[indmin][0]
	B=onepoints[indmin][1]
	C=onepoints[indmin][2]
	
	v0=B-A
	v1=C-A
	v2=P-A
	
	if np.dot(np.cross(v0,v1),normalvectors[indmin])<0:
		v0=C-A
		v1=B-A

	
	
	v00=np.dot(v0,v0)
	v01=np.dot(v0,v1)
	v11=np.dot(v1,v1)
	v20=np.dot(v2,v0)
	v21=np.dot(v2,v1)
	denom=v00*v11-v01*v01
	v=(v11*v20-v01*v21)/denom
	w=(v00*v21-v01*v20)/denom
	u=1-w-v
	
	
	#print(a,b,c)
	#print(verticesnum)
	#print(verticesnum[a],verticesnum[b],verticesnum[c])
	
	cont_point_A=u*particle_A.vertices[verticesnum[a][0]].vertex_coo+v*particle_A.vertices[verticesnum[b][0]].vertex_coo+w*particle_A.vertices[verticesnum[c][0]].vertex_coo
	cont_point_B=u*particle_B.vertices[verticesnum[a][1]].vertex_coo+v*particle_B.vertices[verticesnum[b][1]].vertex_coo+w*particle_B.vertices[verticesnum[c][1]].vertex_coo

	#print(cont_point_A,cont_point_B)
	
	#v0=vertices2-S1
	#v1=S3-S1
	#v2=
	
	
	return(pen_depth,normalvectors[indmin],cont_point_A,cont_point_B)


def CorrectFlat(S1,S2,S3,S4):
	d1=np.array([S1,S2,S3])
	change=0
	if np.linalg.det(d1)<0.00000000001 and np.linalg.det(d1)>-0.00000000001:
		d2=np.array([S1,S2,S4])
		if np.linalg.det(d2)<0.00000000001 and np.linalg.det(d2)>-0.00000000001:
			change=1
	return(change)

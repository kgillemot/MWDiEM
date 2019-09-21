import sys 
sys.path.append("./LIBRARIES/")
import LIBRARY_Fine_contact_detection as Fine_contact
import LIBRARY_Misc as Misc
import LIBRARY_Vector_calculus as Vect_calc
import LIBRARY_Plotting as Plot

import numpy as np
from scipy.spatial import ConvexHull
from numba import guvectorize, float64, njit, float32, vectorize, int32, jit, cuda
import numba


	
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


@cuda.jit(device=True)
def array_min_max_finder(x):
	xmax=x[0]
	xmin=x[0]
	indmax=0
	indmin=0
	for i in range(0,len(x)):
		if x[i]<xmin:
			xmin=x[i]
			indmin=i
		if x[i]>xmax:
			xmax=x[i]
			indmax=i
	return (xmin,xmax,indmin,indmax)

@cuda.jit(device=True)
def Normalvector_plane(V1,V2):
	norm=np.cross(V1,V2)
	normdist=np.sqrt(norm[0]*norm[0]+norm[1]*norm[1]+norm[2]*norm[2])
	return(norm/normdist)

@cuda.jit(device=True)
def Length_vector(A):
	length=np.sqrt(A[0]*A[0]+A[1]*A[1]+A[2]*A[2])
	return(length)

@cuda.jit(device=True)
def Support_function(particle_A_vert,particle_B_vert,d):
	#### WORKS AND IS OBJECT ORIENTED
	#print(particle_A_vert.shape)
	N=particle_A_vert.shape[0]
	M=particle_B_vert.shape[0]
	#print(particle_A_vert)
	dist1=np.zeros((N))
	dist2=np.zeros((M))
	for i in range(0,N):
		dist1[i]=d[0]*particle_A_vert[i][0]+d[1]*particle_A_vert[i][1]+d[2]*particle_A_vert[i][2]
	a=array_min_max_finder(dist1)[3]
	for i in range(0,M):
		dist2[i]=-d[0]*particle_B_vert[i][0]-d[1]*particle_B_vert[i][1]-d[2]*particle_B_vert[i][2]
	b=array_min_max_finder(dist2)[3]
	diff=np.array([particle_A_vert[a][0]-particle_B_vert[b][0],particle_A_vert[a][1]-particle_B_vert[b][1],particle_A_vert[a][2]-particle_B_vert[b][2]])
	return(diff,a,b)

@cuda.jit(device=True)
def Check_linesegment(P1,P2):
	#P2 is closer to the origin
	#check if the Voronoi region outside the slab has the origin
	coll=-1
	d=0
	V21=P2-P1
	#check which Voronoi region has the origin
	if dotp(V21,P2)<0:
		coll=0
		return(coll,d)	
	test=np.cross(V21,P2)
	if test[0]==0 and test[1]==0 and test[2]==0:
		coll=1
		return(coll,d)
	d=np.cross(np.cross(V21,-P2),V21) 
	return(coll,d)


@cuda.jit(device=True)
def Check_triangle(P1,P2,P3):
	#P3 is closest to the origin
	coll=-1
	d=0
	#check if it's outside of the possibble Voronoi region
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
	#Region 2 outside of line 2
	if dotp(n2,P2)<0:
		#d=np.cross(np.cross(V32,P3),-V32)
		d=n2
		#print("Triangle Voronoi Region 2")
		return(coll,d)	
	#Region 3 outside of line 3
	if dotp(n3,P3)<0:
		#d=np.cross(np.cross(V32,P3),-V32)
		d=n3
		#print("Triangle Voronoi Region 3")
		return(coll,d)	
	#Inside the triangle
	if dotp(P3,n)==0:
		coll=1
		#print("Triangle, inside the Triangle")
		return(coll,d)	
	#Above or below the triangle
	if dotp(n,P3)>0:
		d=-n
	else:
		d=n
	#print("Triangle Voronoi regions above or below the triangle")
	return(coll,d)


@cuda.jit(device=True)
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
	if dotp(n2,P4)<0:
		d=n2
		todel=3
	elif dotp(n3,P4)<0:
		d=n3
		todel=1
	elif dotp(n4,P4)<0:
		d=n4
		todel=2
	else:
		coll=1
	return(coll,d,todel)		

@cuda.jit(device=True)
def CorrectFlat(S1,S2,S3,S4):
	d1=np.array([S1,S2,S3])
	change=0
	if np.linalg.det(d1)<0.00000000001 and np.linalg.det(d1)>-0.00000000001:
		d2=np.array([S1,S2,S4])
		if np.linalg.det(d2)<0.00000000001 and np.linalg.det(d2)>-0.00000000001:
			change=1
	return(change)

@cuda.jit(device=True)	
def EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,particle_A_vert,particle_B_vert):
	xmin_old=1000000000
	Suj_proj_len=xmin_old+1
	counter=0	
	vertices=np.array([S1,S2,S3,S4])
	verticesnum=np.array([[aa1,bb1],[aa2,bb2],[aa3,bb3],[aa4,bb4]])
	while(xmin_old<Suj_proj_len):	
		hull= ConvexHull(vertices)	
		verticeshull=[np.float64(x) for x in range(0)]
		numverticeshull=[np.int64(x) for x in range(0)]
		for j in range(0,len(hull.vertices)):
			verticeshull.append(vertices[hull.vertices[j]])
			numverticeshull.append(verticesnum[hull.vertices[j]])			
		normalvectors=[np.flaot64(x) for x in range(0)]
		distances=[np.float64(x) for x in range(0)]
		onepoints=[np.float64(x) for x in range(0)]
		for i in range(0,len(hull.simplices)):
			V1=vertices[hull.simplices[i][1]]-vertices[hull.simplices[i][0]]
			V2=vertices[hull.simplices[i][2]]-vertices[hull.simplices[i][1]]
			V3=vertices[hull.simplices[i][0]]-vertices[hull.simplices[i][2]]
			normtemp=Normalvector_plane(V1,V2)
			dist=dotp(normtemp,vertices[hull.simplices[i][1]])
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
			if dotp(np.cross(V1,Vn1),np.cross(V1,-V3))>0:
				if dotp(np.cross(V2,Vn2),np.cross(V2,-V1))>0:
					if dotp(np.cross(V3,Vn3),np.cross(V3,-V2))>0:
						dist=dist
					else:
						dist=1000000000
				else:
					dist=1000000000
			else:
				dist=1000000000
			distances.append(abs(dist))	
		(xmin_new,xmax_new,indmin,indmax)=array_min_max_finder(distances)
		xmin_old=xmin_new	
		(Suj,aauj,bbuj)=Support_function(particle_A_vert,particle_B_vert,normalvectors[indmin])  #uj pont
		Suj_proj_len=dotp(normalvectors[indmin],Suj)
		vertices=np.concatenate((verticeshull,[Suj]),axis=0)
		verticesnum=np.concatenate((numverticeshull,[[aauj,bbuj]]),axis=0)
		if xmin_old>Suj_proj_len or (xmin_old*1.000000000001>Suj_proj_len and xmin_old*0.99999999999<Suj_proj_len):
			break
		else:
			counter=counter+1
	pen_depth=xmin_new
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
	if dotp(np.cross(v0,v1),normalvectors[indmin])<0:
		v0=C-A
		v1=B-A
	v00=dotp(v0,v0)
	v01=dotp(v0,v1)
	v11=dotp(v1,v1)
	v20=dotp(v2,v0)
	v21=dotp(v2,v1)
	denom=v00*v11-v01*v01
	v=(v11*v20-v01*v21)/denom
	w=(v00*v21-v01*v20)/denom
	u=1-w-v
	cont_point_A=u*particle_A_vert[verticesnum[a]][0]+v*particle_A_vert[verticesnum[b]][0]+w*particle_A_vert[verticesnum[c]][0]
	cont_point_B=u*particle_B_vert[verticesnum[a]][1]+v*particle_B_vert[verticesnum[b]][1]+w*particle_B_vert[verticesnum[c]][1]
	return(pen_depth,normalvectors[indmin],cont_point_A,cont_point_B)


@cuda.jit(device=True)
def Force(particle_A_cm,particle_B_cm,particle_A_vel,particle_B_vel,penetration_depth,penetration_normal,contact_point_A,contact_point_B,Ks,Kd,mu):	
	######### NORMAL FORCE ##############
	# Ez valoszinuleg jo, de azert meg tesztelni kell	
	if dotp(particle_B_cm-particle_A_cm,penetration_normal)<0:
		penetration_normal=-penetration_normal		
	#spring force
	Fn_As=(Ks*penetration_depth**(3/2))*penetration_normal
	if dotp(particle_B_cm-particle_A_cm,Fn_As)>0:
		Fn_As=-Fn_As
	#damping force	
	v_rel=particle_A_vel-particle_B_vel
	v_rel_n=dotp(v_rel,penetration_normal)*penetration_normal
	Fn_Ad=-(Kd*penetration_depth**(1/2))*v_rel_n
	#full normal force	
	Fn_A=Fn_As+Fn_Ad	
	########## TANGENTIAL FORCE ################
	#print(particle_A.velocity)
	v_t_rel=v_rel-v_rel_n
	if np.linalg.norm(v_t_rel)!=0:
		Ft_A=-min(mu*np.linalg.norm(Fn_A),Ks*np.linalg.norm(v_t_rel))*v_t_rel/np.linalg.norm(v_t_rel)
	else:
		Ft_A=np.array([0,0,0])	
	Ft_B=-Ft_A	
	########### TORQUE #########################
	TorqueA=np.cross(contact_point_A-particle_A_cm,Ft_A)
	TorqueB=np.cross(contact_point_B-particle_B_cm,Ft_B)
	return(Fn_A,Fn_As,Fn_Ad,TorqueA,TorqueB)

@cuda.jit(device=True)
def dotp(vectorA,vectorB):
	dotti=vectorA[0]*vectorB[0]+vectorA[1]*vectorB[1]+vectorA[2]*vectorB[2]
	return(dotti)

@cuda.jit
#@guvectorize([(float64[:], float64[:],float64[:],float64[:],float64[:],float64[:],float64,float64,float64,float64,float64,float64[:,:])], '(n,m),(n,m),(n,o,m),(n,o,m),(n,m),(n,m),(),(),(),(),()->(n,m)', target='cuda')
def Contact_detection_GPU(particle_A_center_of_mass,particle_B_center_of_mass,particle_A_vertices,particle_B_vertices,particle_A_velocity,particle_B_velocity,aa,bb,Ks,Kd,mu,force):
	
	
	
	for i in range(0,len(particle_A_center_of_mass)):
	
		
		
		
		Fn_A=[0,0,0]
		Fn_As=[0,0,0]
		Fn_Ad=[0,0,0]
		Torque_A=[0,0,0]
		Torque_B=[0,0,0]

		print(numba.typeof(Fn_A))

		#force[i]=Fn_A
		force[i]=[Fn_A,Fn_As,Fn_Ad,Torque_A,Torque_B]
		#force[i]=[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
		
		
		A=particle_A_vertices[i]
		B=particle_B_vertices[i]
		

		#is there a collision or not
		coll=-1

		#Center of mass
		CmA=particle_A_center_of_mass[i]
		CmB=particle_B_center_of_mass[i]

		#1. Single point
		#print("Single point")
		d1=CmB-CmA
		(S1,aa1,bb1)=Support_function(A,B,d1)
		
		#check if the new point crosses the origin or not
		if dotp(d1,S1)<0:
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
			return(force)
			#check if the new point is the origin

		#2. Line segment	
		#print("Line segment")
		d2=-1*d1	
		(S2,aa2,bb2)=Support_function(A,B,d2)	
		#check if the new point crosses the origin or not
		if dotp(d2,S2)<0:
			#print(str(aa)+" "+str(bb)+" NO COLLISION Line SAT\n	")
			S3=[0,0,0]
			aa3=0
			bb3=0
			S4=[0,0,0]
			aa4=0
			bb4=0
			coll=0
			return(force)

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
			return(force)

		#3. Triangle
		(S3,aa3,bb3)=Support_function(A,B,d3)
		#check if the new point crosses the origin or not
		if dotp(d3,S3)<0:
			#print(str(aa)+" "+str(bb)+" NO COLLISION Triangle SAT\n")
			S4=[0,0,0]
			aa4=0
			bb4=0
			coll=0
			return(force)
		#check if the new point is the origin

		(coll,d4)=Check_triangle(S1,S2,S3)
		if coll==0:
			#print(str(aa)+" "+str(bb)+" NO COLLISION Triangele Voronoi\n	")
			S4=[0,0,0]
			coll=0
			return(force)

		#4. Tetrahedron
		zz=0
		while (zz<100):
			#print("Iter Tetrahedron ",zz,"\n")
			(S4,aa4,bb4)=Support_function(A,B,d4)
			
			#check if the new point crosses the origin or not
			if dotp(d4,S4)<0:
				#print(str(aa)+" "+str(bb)+" NO COLLISION Tetra SAT\n")
				coll=0
				return(force)
			#check if the new point is the origin
			if S4[0]==0 and S4[1]==0 and S4[2]==0:
				#print(str(aa)+" "+str(bb)+" COLLISION Tetra Hitting origin\n	")
				coll=1
				change=CorrectFlat(S1,S2,S3,S4)
				if change==1:
					d4=np.cross(S1,S2)
					(S4,aa4,bb4)=Support_function(A,B,d4)
				(penetration_depth,penetration_normal,contact_point_A,contact_point_B)=Fine_contact.EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,A,B)	
				
				(Fn_A,Fn_As,Fn_Ad,TorqueA,TorqueB)=Force(particle_A_center_of_mass[i],particle_B_center_of_mass[i],particle_A_velocity[i],particle_B_velocity[i],penetration_depth,penetration_normal,contact_point_A,contact_point_B,Ks,Kd,mu)
				force[i][0]=Fn_A
				force[i][1]=Fn_As
				force[i][2]=Fn_Ad
				force[i][3]=np.array(Torque_A)
				force[i][4]=Torque_B
				return(force)
			(coll,d4,todel)=Check_Tetrahedron(S1,S2,S3,S4)
			if coll==0:
				#print(str(aa)+" "+str(bb)+" NO COLLISION Tetra Voronoi\n	")
				coll=0
				return(force)
			if coll==1:
				#print(str(aa)+" "+str(bb)+" COLLISION Tetra Vornoi\n	")
				coll=1
				change=CorrectFlat(S1,S2,S3,S4)
				if change==1:
					d4=np.cross(S1,S2)
					(S4,aa4,bb4)=Support_function(A,B,d4)
				(penetration_depth,penetration_normal,contact_point_A,contact_point_B)=Fine_contact.EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,A,B)	
				(Fn_A,Fn_As,Fn_Ad,TorqueA,TorqueB)=Force(particle_A_center_of_mass[i],particle_B_center_of_mass[i],particle_A_velocity[i],particle_B_velocity[i],penetration_depth,penetration_normal,contact_point_A,contact_point_B,Ks,Kd,mu)
				force[i]=[Fn_A,Fn_As,Fn_Ad,Torque_A,Torque_B]
				return(force)	
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
			#del S1,S2,S3, aa1, aa2, aa3, bb1, bb2, bb3
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
		return(force)











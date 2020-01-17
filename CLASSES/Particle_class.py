import sys 
sys.path.append("./LIBRARIES/")
import LIBRARY_Vector_calculus as Vect_calc
import numpy as np
import math
from scipy.spatial import ConvexHull
import LIBRARY_Quaternion as Quater

class Particle(object):
	
	def __init__(self,particle_id):
		
		self.particle_id=particle_id
		

		#MATERIAL PROPERTIES
		#self.density=0
		#self.youngmodulus=0
		#self.poissonratio=0
		
		#TRANSLATIONAL MOVEMENT 
		#self.mass=mass
		#self.mass=self.density*self.volume
		self.velocity=np.array([0,0,0])
		self.velocity_12=np.array([0,0,0])
		#self.velocity_old=np.array([0,0,0])
		self.force_normal=np.array([0,0,0])
		#self.force_normal_old=np.array([0,0,0])
		self.force_n_spring=np.array([0,0,0])
		self.force_n_damping=np.array([0,0,0])
		#self.acceleration=np.array([0,0,0])
		self.force_external=np.array([0,0,0])
		
		#ROTATIONAL MOVEMENT

		self.torque=np.array([0,0,0])
		self.torque_bf=np.array([0,0,0])
		#self.torque_old=np.array([0,0,0])
		self.torque_external=np.array([0,0,0])
		self.angular_velocity_bf=np.array([0,0,0])
		self.angular_velocity_12_bf=np.array([0,0,0])
		self.angular_velocity=np.array([0,0,0])
		self.angular_velocity_12=np.array([0,0,0])
		self.angular_acceleration_bf=np.array([0,0,0])
		#self.alfa_old=np.array([0,0,0])
		#self.alfa=np.array([0,0,0])
		
		#self.moment_of_inertia=Calc_inertia()
		
		#CREATE QUATERNION
		self.quaternion=np.array([1,0,0,0])
		self.quaternion_12=np.array([1,0,0,0])
		self.quaternion_all=np.array([1,0,0,0])
		
	
	
	########################### PARTICLE GEOMETRICAL PROPERTIES #####################################
		
	#csucspontok
	class Vertex():
		def __init__(self,vertex_id,hull):
			self.vertex_id=vertex_id
			self.vertex_id_old=hull.vertices[vertex_id]
			self.vertex_coo=np.array([hull.points[self.vertex_id_old][0],hull.points[self.vertex_id_old][1],hull.points[self.vertex_id_old][2]])

			self.vertex_edges_id=[]
			self.vertex_faces_id=[]

	#oldalak
	class Face():
		#the faces are currently triangles, but should work with other features as well	
		def __init__(self,face_id,hull,center_of_mass,vertex):                  
			self.face_id=face_id
			
			#face vertices
			self.face_num_vertices=len(hull.simplices[face_id])
			self.face_vertices=[]
			self.face_vertices_id=[]
			for i in range(0,self.face_num_vertices):
				self.face_vertices.append(hull.points[hull.simplices[face_id][i]])
				for j in range(0,len(hull.vertices)):
					if vertex[j].vertex_id_old==hull.simplices[face_id][i]:
						self.face_vertices_id.append(j)						
			self.face_vertices=np.array(self.face_vertices)
			#print(self.face_vertices)
			self.face_vertices_id=np.array(self.face_vertices_id)
			
			


			
			##face edges
			##currently only works with either triangles or ordered lists
			#self.face_num_edges=self.face_num_vertices-1
			#self.face_edges=[]
			#self.face_edges_id=[]
			#for i in range(0,self.face_num_vertices-1):
				#self.face_edges.append(self.face_vertices[i+1]-self.face_vertices[i])
				#self.face_edges_id.append([self.face_vertices_id[i+1],self.face_vertices_id[i]])
			#self.face_edges.append(self.face_vertices[0]-self.face_vertices[self.face_num_vertices-1])
			#self.face_edges_id.append([self.face_vertices_id[0],self.face_vertices_id[self.face_num_vertices-1]])
			#self.face_edges=np.array(self.face_edges)
			#self.face_edges_id=np.array(self.face_edges_id)


			#face normal pointing outward
			crossproduct=np.cross(self.face_vertices[1]-self.face_vertices[0],self.face_vertices[2]-self.face_vertices[1])
			self.face_normal=crossproduct/Vect_calc.Length_vector(crossproduct)
			dirvector=self.face_vertices[0]-center_of_mass
			if np.dot(self.face_normal,dirvector)<0:
				self.face_normal=-1*self.face_normal
			self.face_normal=np.array(self.face_normal)	

			
	#class Edge():
		#def __init__(self,edge_id)
			#self.edge_id=edge_id
			#self.edge_dir=np.zeros(3)
			#self.edge_vertices=np.zeros((2,3))
			#self.edge_vertices_id=np.zeros(2)
			#self.edge_faces_id=np.zeros(2)
			
	def Calc_inertia(self,density):
		#Inertia of Any Polyhedron, A NTHONY R. D OBROVOLSKIS. ICARUS 124, 698–704 (1996) 0243
		Pxx=0
		Pxy=0
		Pxz=0
		Pyy=0
		Pyz=0
		Pzz=0
		for i in range(0,self.num_faces):
			D=self.faces[i].face_vertices[0]
			E=self.faces[i].face_vertices[1]
			F=self.faces[i].face_vertices[2]
			deltaV=np.dot(D,self.faces[i].face_normal)/6
			#print("deltav",deltaV)
			Pxx=Pxx+density*deltaV/20*(2*D[0]*D[0]+2*E[0]*E[0]+2*F[0]*F[0]+D[0]*E[0]+D[0]*E[0]+D[0]*F[0]+D[0]*F[0]+E[0]*F[0]+E[0]*F[0])
			Pxy=Pxy+density*deltaV/20*(2*D[0]*D[1]+2*E[0]*E[1]+2*F[0]*F[1]+D[0]*E[1]+D[1]*E[0]+D[0]*F[1]+D[1]*F[0]+E[0]*F[1]+E[1]*F[0])
			Pxz=Pxz+density*deltaV/20*(2*D[0]*D[2]+2*E[0]*E[2]+2*F[0]*F[2]+D[0]*E[2]+D[2]*E[0]+D[0]*F[2]+D[2]*F[0]+E[0]*F[2]+E[2]*F[0])
			Pyy=Pyy+density*deltaV/20*(2*D[1]*D[1]+2*E[1]*E[1]+2*F[1]*F[1]+D[1]*E[1]+D[1]*E[1]+D[1]*F[1]+D[1]*F[1]+E[1]*F[1]+E[1]*F[1])
			Pyz=Pyz+density*deltaV/20*(2*D[1]*D[2]+2*E[1]*E[2]+2*F[1]*F[2]+D[1]*E[2]+D[2]*E[1]+D[1]*F[2]+D[2]*F[1]+E[1]*F[2]+E[2]*F[1])
			Pzz=Pzz+density*deltaV/20*(2*D[2]*D[2]+2*E[2]*E[2]+2*F[2]*F[2]+D[2]*E[2]+D[2]*E[2]+D[2]*F[2]+D[2]*F[2]+E[2]*F[2]+E[2]*F[2])
		Ixx=Pyy+Pzz
		Iyy=Pxx+Pzz
		Izz=Pxx+Pyy
		Iyz=-Pyz
		Ixz=-Pxz
		Ixy=-Pxy	
		I=np.array([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])	
		if np.isfinite(np.linalg.cond(I))==False:
			print("Singular matrix, ohoh!")

		return(I)	
	
	#################### PARTICLE CREATORS ################################################################
	def Translate(self,transl):
		translx=transl[0]
		transly=transl[1]
		translz=transl[2]
		for i in range(0,self.num_vertices):
			self.vertices[i].vertex_coo[0]=self.vertices[i].vertex_coo[0]+translx
			self.vertices[i].vertex_coo[1]=self.vertices[i].vertex_coo[1]+transly
			self.vertices[i].vertex_coo[2]=self.vertices[i].vertex_coo[2]+translz
		self.center_of_mass[0]=self.center_of_mass[0]+translx
		self.center_of_mass[1]=self.center_of_mass[1]+transly
		self.center_of_mass[2]=self.center_of_mass[2]+translz
		for i in range(0,self.num_faces):
			self.faces[i].face_vertices[0][0]+=translx
			self.faces[i].face_vertices[0][1]+=transly
			self.faces[i].face_vertices[0][2]+=translz
			
			self.faces[i].face_vertices[1][0]+=translx
			self.faces[i].face_vertices[1][1]+=transly
			self.faces[i].face_vertices[1][2]+=translz
		
			self.faces[i].face_vertices[2][0]+=translx
			self.faces[i].face_vertices[2][1]+=transly
			self.faces[i].face_vertices[2][2]+=translz
	
		
	def Random_convex_polygon_maker_3D(self,num_vertices_min,num_vertices_max,min_radius,max_radius):
		
		#CREATE POLYGON
		num_vertices=np.random.randint(num_vertices_min,num_vertices_max)		
		max_particlesize=max_radius*2	
		radius_int=np.random.randint(min_radius,max_radius)
		radius_float=np.random.rand()
		radius=radius_int+radius_float
		fi=np.random.rand(num_vertices)*180*math.pi/180
		teta=np.random.rand(num_vertices)*360*0.9999999999999*math.pi/180
		polyhedra=[]
		for i in range(0,num_vertices):
			polyhedra.append([radius*math.sin(fi[i])*math.cos(teta[i]),radius*math.sin(fi[i])*math.sin(teta[i]),radius*math.cos(fi[i])])
		polyhedra=np.array(polyhedra)	
		polyhedra=np.random.rand(num_vertices,3)*max_particlesize
		hull= ConvexHull(polyhedra)
		
		#DETERMINE NUMBER OF TOPOLOGICAL FEATURES
		self.num_vertices=len(hull.vertices)
		self.num_faces=len(hull.simplices)
		self.num_edges=self.num_vertices+self.num_faces-2
		self.volume = ConvexHull(polyhedra).volume
		
		#CREATE VERTICES
		self.vertices=[]
		for i in range(0,self.num_vertices):
			self.vertices.append(self.Vertex(i,hull))

		#CENTER OF MASS	
		self.center_of_mass=[0,0,0]
		for i in range(0,self.num_vertices):
			self.center_of_mass=self.center_of_mass+self.vertices[i].vertex_coo
		self.center_of_mass=self.center_of_mass/self.num_vertices
			
		#CREATE FACES
		self.faces=[]
		for i in range(0,self.num_faces):
			self.faces.append(self.Face(i,hull,self.center_of_mass,self.vertices))
		
		##CREATE EDGES
		#self.edges=[]
		#for i in self.num_edges:
			#self.edges.append(Edge(i))

		#DENSITY
		self.density=density
			
		#MOMENT OF INERTIA
		self.moment_of_inertia=self.Calc_inertia(self.density)

		#CALCULATE MASS
		self.mass=self.density*self.volume

	def Spherical_convex_polygon_maker_3D(self,num_vertices_min,num_vertices_max,radius,variation,density,typee,acc_ext_def,torq_ext_def):
		
		#CREATE POLYGON
		num_vertices=np.random.randint(num_vertices_min,num_vertices_max)		
		radius=np.random.normal(radius,variation)
		fi=np.random.rand(num_vertices)*180*math.pi/180
		teta=np.random.rand(num_vertices)*360*0.9999999999999*math.pi/180
		polyhedra=[]
		for i in range(0,num_vertices):
			polyhedra.append([radius*math.sin(fi[i])*math.cos(teta[i]),radius*math.sin(fi[i])*math.sin(teta[i]),radius*math.cos(fi[i])])
		polyhedra=np.array(polyhedra)	
		#polyhedra=np.random.rand(num_vertices,3)*max_particlesize
		hull= ConvexHull(polyhedra)
		
		#DETERMINE NUMBER OF TOPOLOGICAL FEATURES
		self.num_vertices=len(hull.vertices)
		self.num_faces=len(hull.simplices)
		self.num_edges=self.num_vertices+self.num_faces-2
		self.volume = ConvexHull(polyhedra).volume
		
		#CREATE VERTICES
		self.vertices=[]
		for i in range(0,self.num_vertices):
			self.vertices.append(self.Vertex(i,hull))

		#CENTER OF MASS	
		self.center_of_mass=[0,0,0]
		for i in range(0,self.num_vertices):
			self.center_of_mass=self.center_of_mass+self.vertices[i].vertex_coo
		self.center_of_mass=self.center_of_mass/self.num_vertices

		
		#CREATE FACES
		self.faces=[]
		for i in range(0,self.num_faces):
			self.faces.append(self.Face(i,hull,self.center_of_mass,self.vertices))
		
		##CREATE EDGES
		#self.edges=[]
		#for i in self.num_edges:
			#self.edges.append(Edge(i))

		
		self.Translate(-self.center_of_mass)

		#DENSITY
		self.density=density
			
		#MOMENT OF INERTIA
		self.moment_of_inertia=self.Calc_inertia(self.density)
		self.eig=np.linalg.eig(self.moment_of_inertia)
		self.moment_of_inertia_bf=np.array([[self.eig[0][0],0,0],[0,self.eig[0][1],0],[0,0,self.eig[0][2]]])
		self.moment_of_inertia_bf_inv=np.linalg.inv(self.moment_of_inertia_bf)
		
		#CALCULATE MASS
		self.mass=self.density*self.volume
		
		self.typee=typee
		
		#print(self.mass)
		self.force_external=self.mass*acc_ext_def
		self.torque_external=torq_ext_def
		

	def Cube_maker_3D(self,side_length):
		
		#CREATE POLYGON
		polyhedra=[]
		polyhedra.append([0,0,0])
		polyhedra.append([side_length,0,0])
		polyhedra.append([0,side_length,0])
		polyhedra.append([side_length,side_length,0])
		polyhedra.append([0,0,side_length])
		polyhedra.append([side_length,0,side_length])
		polyhedra.append([0,side_length,side_length])
		polyhedra.append([side_length,side_length,side_length])
		polyhedra=np.array(polyhedra)	
		hull= ConvexHull(polyhedra)
		
		#DETERMINE NUMBER OF TOPOLOGICAL FEATURES
		self.num_vertices=len(hull.vertices)
		self.num_faces=len(hull.simplices)
		self.num_edges=self.num_vertices+self.num_faces-2
		self.volume = ConvexHull(polyhedra).volume
		
		#CREATE VERTICES
		self.vertices=[]
		for i in range(0,self.num_vertices):
			self.vertices.append(self.Vertex(i,hull))
			
		#CREATE QUATERNION
		self.quaternion=[1,0,0,0]

		#CENTER OF MASS	
		self.center_of_mass=[0,0,0]
		for i in range(0,self.num_vertices):
			self.center_of_mass=self.center_of_mass+self.vertices[i].vertex_coo
		self.center_of_mass=self.center_of_mass/self.num_vertices
			
		#CREATE FACES
		self.faces=[]
		for i in range(0,self.num_faces):
			self.faces.append(self.Face(i,hull,self.center_of_mass,self.vertices))
		
		##CREATE EDGES
		#self.edges=[]
		#for i in self.num_edges:
			#self.edges.append(Edge(i))
		
		self.Translate(-self.center_of_mass)
		
		#DENSITY
		self.density=density
			
		#MOMENT OF INERTIA
		self.moment_of_inertia=self.Calc_inertia(self.density)
		
		#CALCULATE MASS
		self.mass=self.density*self.volume
		
		
	def Plane_maker_3D(self,size_plane_x,size_plane_y,size_plane_z,density,max_particlesize,typee,acc_ext_def,torq_ext_def):
		
		#CREATE POLYGON
		polyhedra=[]
		normalvector=[0,0,1]
		
		#if normalvector[0]==0 and normalvector[1]==0:
		p1=[size_plane_x/2,size_plane_y/2,size_plane_z/2]
		p2=[size_plane_x/2,size_plane_y/2,-size_plane_z/2]
		p3=[size_plane_x/2,-size_plane_y/2,size_plane_z/2]
		p4=[size_plane_x/2,-size_plane_y/2,-size_plane_z/2]
		p5=[-size_plane_x/2,size_plane_y/2,size_plane_z/2]
		p6=[-size_plane_x/2,size_plane_y/2,-size_plane_z/2]
		p7=[-size_plane_x/2,-size_plane_y/2,size_plane_z/2]
		p8=[-size_plane_x/2,-size_plane_y/2,-size_plane_z/2]
		
		#if normalvector[0]==0 and normalvector[2]==0:
		#	p1=[size,0,size]
		#	p2=[-size,0,size]
		#	p3=[size,0,-size]
		#	p4=[-size,0,-size]
			
		#if normalvector[1]==0 and normalvector[2]==0:
		#	p1=[0,size,size]
		#	p2=[0,size,-size]
		#	p3=[0,-size,size]
		#	p4=[0,-size,-size]
			
		#p5=p1-np.multiply(normalvector,[size,size,size])
		#p6=p2-np.multiply(normalvector,[size,size,size])
		#p7=p3-np.multiply(normalvector,[size,size,size])
		#p8=p4-np.multiply(normalvector,[size,size,size])

		
		polyhedra=np.array([p1,p2,p3,p4,p5,p6,p7,p8])
		hull= ConvexHull(polyhedra)
		
		#DETERMINE NUMBER OF TOPOLOGICAL FEATURES
		self.num_vertices=len(hull.vertices)
		self.num_faces=len(hull.simplices)
		self.num_edges=self.num_vertices+self.num_faces-2
		self.volume = ConvexHull(polyhedra).volume
		
		#CREATE VERTICES
		self.vertices=[]
		for i in range(0,self.num_vertices):
			self.vertices.append(self.Vertex(i,hull))

		#CENTER OF MASS	
		self.center_of_mass=[0,0,0]
		for i in range(0,self.num_vertices):
			self.center_of_mass=self.center_of_mass+self.vertices[i].vertex_coo
		self.center_of_mass=self.center_of_mass/self.num_vertices
			
		#CREATE FACES
		self.faces=[]
		for i in range(0,self.num_faces):
			self.faces.append(self.Face(i,hull,self.center_of_mass,self.vertices))
		
		##CREATE EDGES
		#self.edges=[]
		#for i in self.num_edges:
			#self.edges.append(Edge(i))
		
		#DENSITY
		self.density=density
		
		#MOMENT OF INERTIA
		#MOMENT OF INERTIA
		self.moment_of_inertia=self.Calc_inertia(self.density)
		self.eig=np.linalg.eig(self.moment_of_inertia)
		self.moment_of_inertia_bf=np.array([[self.eig[0][0],0,0],[0,self.eig[0][1],0],[0,0,self.eig[0][2]]])
		self.moment_of_inertia_bf_inv=np.linalg.inv(self.moment_of_inertia_bf)

		
		#CALCULATE MASS
		self.mass=self.density*self.volume
		
		#PLANE NORMAL VECTOR
		slope_angle = 19.1 #[deg]
		slope_angle_rad=slope_angle*np.pi/180
		#self.normvec=[0,0,1]
		self.normvec=np.array([-math.sin(slope_angle_rad),0,math.cos(slope_angle_rad)]) 
		
		self.maxsize=math.sqrt(math.sqrt(size_plane_x**2/4+size_plane_y**2/4)+size_plane_z**2/4)
		
		self.typee=typee
		
				
		self.force_external=self.mass*acc_ext_def
		self.torque_external=torq_ext_def

	################################## PARTICLE MOVERS ######################################


	
			
			
	def Translate_wall(self,transl):
		translx=transl[0]
		transly=transl[1]
		translz=transl[2]
		for i in range(0,self.num_vertices):
			self.vertices[i].vertex_coo[0]=self.vertices[i].vertex_coo[0]+translx
			self.vertices[i].vertex_coo[1]=self.vertices[i].vertex_coo[1]+transly
			self.vertices[i].vertex_coo[2]=self.vertices[i].vertex_coo[2]+translz
		self.center_of_mass[0]=self.center_of_mass[0]+translx
		self.center_of_mass[1]=self.center_of_mass[1]+transly
		self.center_of_mass[2]=self.center_of_mass[2]+translz
		#for i in range(0,self.num_faces):
			#self.faces[i].face_vertices[0][0]+=translx
			#self.faces[i].face_vertices[0][1]+=transly
			#self.faces[i].face_vertices[0][2]+=translz
	
			#self.faces[i].face_vertices[1][0]+=translx
			#self.faces[i].face_vertices[1][1]+=transly
			#self.faces[i].face_vertices[1][2]+=translz
		
			#self.faces[i].face_vertices[2][0]+=translx
			#self.faces[i].face_vertices[2][1]+=transly
			#self.faces[i].face_vertices[2][2]+=translz
			
			
			

	def Particles_to_cells_hash(self,cellsize_x,cellsize_y,cellsize_z,numcells_x,numcells_y):
		self.xc=int(self.center_of_mass[0]/cellsize_x)
		self.yc=int(self.center_of_mass[1]/cellsize_y)
		self.zc=int(self.center_of_mass[2]/cellsize_z)
		self.hashh=self.xc+	self.yc*numcells_x+self.zc*numcells_x*numcells_y
		
		
	#def Rotator(self,alfa):
		##A description of rotations for DEM models of particle systems Authors Authors and affiliations Eduardo M. B. CampelloEmail author
		#def Rot(alfa,vector):
			#alfa=np.array(alfa,dtype=np.float64)
			#alfa_len=np.linalg.norm(alfa)
			#I=np.identity(3,dtype=np.float64)
			#skew=np.array([[0,-alfa[2],alfa[1]],[alfa[2],0,-alfa[0]],[-alfa[1],alfa[0],0]])
			#Q=I+4/(4+alfa_len**2)*(skew+1/2*np.matmul(skew,skew))
			#newvector=np.matmul(Q,vector)
			#return(newvector)
		
		#for i in range(0,self.num_vertices):
			#self.vertices[i].vertex_coo=Rot(alfa,(self.vertices[i].vertex_coo-self.center_of_mass))+self.center_of_mass
		#for i in range(0,self.num_faces):
			#self.faces[i].face_vertices[0]=Rot(alfa,self.faces[i].face_vertices[0]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_vertices[1]=Rot(alfa,self.faces[i].face_vertices[1]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_vertices[2]=Rot(alfa,self.faces[i].face_vertices[2]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_normal=Rot(alfa,self.faces[i].face_normal)
		#self.moment_of_inertia=self.Calc_inertia(self.density)
	
	def Rotator_quaternion_initial(self,angle_deg,axis):
		def Rota(angle,axis,vector):
			newvector=vector*np.cos(angle)+np.cross(axis,vector)*np.sin(angle)+np.dot(vector,axis)*(1-np.cos(angle))*axis
			return(newvector)
		angle=angle_deg*np.pi/180
		for i in range(0,self.num_vertices):
			self.vertices[i].vertex_coo=Rota(angle,axis,(self.vertices[i].vertex_coo-self.center_of_mass))+self.center_of_mass
		for i in range(0,self.num_faces):
			self.faces[i].face_vertices[0]=Rota(angle,axis,self.faces[i].face_vertices[0]-self.center_of_mass)+self.center_of_mass
			self.faces[i].face_vertices[1]=Rota(angle,axis,self.faces[i].face_vertices[1]-self.center_of_mass)+self.center_of_mass
			self.faces[i].face_vertices[2]=Rota(angle,axis,self.faces[i].face_vertices[2]-self.center_of_mass)+self.center_of_mass
			self.faces[i].face_normal=Rota(angle,axis,self.faces[i].face_normal)
			
		#self.quaternion=[np.cos(angle/2),np.sin(angle/2)*axis[0],np.sin(angle/2)*axis[1],np.sin(angle/2)*axis[2]]
		
	def Rotator_quaternion(self,quaternion):
		def Rotaa(quaternion,vector):
			newvec=Quater.Quat_triple_prod(quaternion,vector,Quater.Quat_inv(quaternion))
			return(newvec)
		
		for i in range(0,self.num_vertices):
			self.vertices[i].vertex_coo=Rotaa(quaternion,(self.vertices[i].vertex_coo-self.center_of_mass))+self.center_of_mass
		#for i in range(0,self.num_faces):
			#self.faces[i].face_vertices[0]=Rotaa(quaternion,self.faces[i].face_vertices[0]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_vertices[1]=Rotaa(quaternion,self.faces[i].face_vertices[1]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_vertices[2]=Rotaa(quaternion,self.faces[i].face_vertices[2]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_normal=Rotaa(quaternion,self.faces[i].face_normal)
	
	def Rotator_quaternion_all(self):
		def Rotaa(quaternion,vector):
			newvec=Quater.Quat_triple_prod(quaternion,vector,Quater.Quat_inv(quaternion))
			return(newvec)
		
		for i in range(0,self.num_vertices):
			self.vertices[i].vertex_coo=Rotaa(self.quaternion_all,(self.vertices[i].vertex_coo-self.center_of_mass))+self.center_of_mass
		for i in range(0,self.num_faces):
			self.faces[i].face_vertices[0]=Rotaa(self.quaternion_all,self.faces[i].face_vertices[0]-self.center_of_mass)+self.center_of_mass
			self.faces[i].face_vertices[1]=Rotaa(self.quaternion_all,self.faces[i].face_vertices[1]-self.center_of_mass)+self.center_of_mass
			self.faces[i].face_vertices[2]=Rotaa(self.quaternion_all,self.faces[i].face_vertices[2]-self.center_of_mass)+self.center_of_mass
			self.faces[i].face_normal=Rotaa(self.quaternion_all,self.faces[i].face_normal)		
		self.quaternion_all=np.array([1,0,0,0])
			
	#def Rotator_initial(self, angle, axis_normalvect):
		#I=np.identity(3)
		#teta=angle*np.pi/180
		#skew=np.array([[0,-axis_normalvect[2],axis_normalvect[1]],[axis_normalvect[2],0,-axis_normalvect[0]],[-axis_normalvect[1],axis_normalvect[0],0]])
		#Q=I+np.sin(teta)*skew+(1-np.cos(teta)**2)*np.matmul(skew,skew)
		##newvector=np.matmul(Q,vector)
		
		#for i in range(0,self.num_vertices):
			#self.vertices[i].vertex_coo=np.matmul(Q,self.vertices[i].vertex_coo-self.center_of_mass)+self.center_of_mass
		#for i in range(0,self.num_faces):
			#self.faces[i].face_vertices[0]=np.matmul(Q,self.faces[i].face_vertices[0]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_vertices[1]=np.matmul(Q,self.faces[i].face_vertices[1]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_vertices[2]=np.matmul(Q,self.faces[i].face_vertices[2]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_normal=np.matmul(Q,self.faces[i].face_normal)
		#self.moment_of_inertia=self.Calc_inertia(self.density)
		
		
	#def Rotator_wall(self, angle, axis_normalvect):
		#I=np.identity(3)
		#teta=angle*np.pi/180
		#skew=np.array([[0,-axis_normalvect[2],axis_normalvect[1]],[axis_normalvect[2],0,-axis_normalvect[0]],[-axis_normalvect[1],axis_normalvect[0],0]])
		#Q=I+np.sin(teta)*skew+(1-np.cos(teta)**2)*np.matmul(skew,skew)
		
		
		#for i in range(0,self.num_vertices):
			#self.vertices[i].vertex_coo=np.matmul(Q,self.vertices[i].vertex_coo-self.center_of_mass)+self.center_of_mass
		#for i in range(0,self.num_faces):
			#self.faces[i].face_vertices[0]=np.matmul(Q,self.faces[i].face_vertices[0]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_vertices[1]=np.matmul(Q,self.faces[i].face_vertices[1]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_vertices[2]=np.matmul(Q,self.faces[i].face_vertices[2]-self.center_of_mass)+self.center_of_mass
			#self.faces[i].face_normal=np.matmul(Q,self.faces[i].face_normal)
		#self.moment_of_inertia=self.Calc_inertia(self.density)
		
		#self.normvec=np.matmul(Q,self.normvec)
		
		
			

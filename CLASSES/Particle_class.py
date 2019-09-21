import sys 
sys.path.append("./LIBRARIES/")
import LIBRARY_Vector_calculus as Vect_calc
import numpy as np
import math
from scipy.spatial import ConvexHull


class Particle(object):
	
	def __init__(self,particle_id,force_external,mass):
		
		self.particle_id=particle_id
		

		#MATERIAL PROPERTIES
		self.atomtype=1
		self.density=0
		
		self.youngmodulus=0
		self.poissonratio=0
		
		#TRANSLATIONAL MOVEMENT 
		self.mass=mass
		#self.mass=self.density*self.volume
		self.velocity=np.array([0,0,0])
		self.velocity_old=np.array([0,0,0])
		self.force_normal=np.array([0,0,0])
		self.force_normal_old=np.array([0,0,0])
		self.force_n_spring=np.array([0,0,0])
		self.force_n_damping=np.array([0,0,0])
		self.acceleration=np.array([0,0,0])
		
		
		#EXTERNAL FORCES
		self.force_external=force_external
		
		#ROTATIONAL MOVEMENT
		self.moment_of_inertia=1
		self.torque=np.array([0,0,0])
		self.torque_old=np.array([0,0,0])
		self.torque_external=np.array([0,0,0])
		self.angular_velocity_old=np.array([0,0,0])
		self.angular_velocity=np.array([0,0,0])
		self.alfa_old=np.array([0,0,0])
		self.alfa=np.array([0,0,0])
		
		
	
	
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
			
	
	#################### PARTICLE CREATORS ################################################################
		
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
			
			
	def Plane_maker_3D(self,normalvector,size_plane):
		
		#CREATE POLYGON
		polyhedra=[]
		size=size_plane/2
		
		if normalvector[0]==0 and normalvector[1]==0:
			p1=[size,size,0]
			p2=[size,-size,0]
			p3=[-size,size,0]
			p4=[-size,-size,0]
		
		if normalvector[0]==0 and normalvector[2]==0:
			p1=[size,0,size]
			p2=[-size,0,size]
			p3=[size,0,-size]
			p4=[-size,0,-size]
			
		if normalvector[1]==0 and normalvector[2]==0:
			p1=[0,size,size]
			p2=[0,size,-size]
			p3=[0,-size,size]
			p4=[0,-size,-size]
			
		p5=p1-np.multiply(normalvector,[size,size,size])
		p6=p2-np.multiply(normalvector,[size,size,size])
		p7=p3-np.multiply(normalvector,[size,size,size])
		p8=p4-np.multiply(normalvector,[size,size,size])

		
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
		
		
		
	################################## PARTICLE MOVERS ######################################


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
			

	def Particles_to_cells_hash(self,cellsize,numcells):
		self.xc=int(self.center_of_mass[0]/cellsize)
		self.yc=int(self.center_of_mass[1]/cellsize)
		self.zc=int(self.center_of_mass[2]/cellsize)
		self.hashh=self.xc+	self.zc*numcells+self.yc*numcells*numcells
		
		
	def Rotator(self,alfa):
		#A description of rotations for DEM models of particle systems Authors Authors and affiliations Eduardo M. B. CampelloEmail author
		def Rot(alfa,vector):
			alfa_len=np.linalg.norm(alfa)
			I=np.identity(3)
			skew=np.array([[0,-alfa[2],alfa[1]],[alfa[2],0,-alfa[0]],[-alfa[1],alfa[0],0]])
			Q=I+4/(1+alfa_len**2)*(skew+1/2*np.matmul(skew,skew))
			newvector=np.matmul(Q,vector)
			return(newvector)
		
		for i in range(0,self.num_vertices):
			self.vertices[i].vertex_coo=Rot(alfa,(self.vertices[i].vertex_coo-self.center_of_mass))+self.center_of_mass
		for i in range(0,self.num_faces):
			self.faces[i].face_vertices[0]=Rot(alfa,self.faces[i].face_vertices[0]-self.center_of_mass)+self.center_of_mass
			self.faces[i].face_vertices[1]=Rot(alfa,self.faces[i].face_vertices[1]-self.center_of_mass)+self.center_of_mass
			self.faces[i].face_vertices[2]=Rot(alfa,self.faces[i].face_vertices[2]-self.center_of_mass)+self.center_of_mass
			
	def Rotator_initial(self, angle, axis_normalvect):
		I=np.identity(3)
		teta=angle*np.pi/180
		skew=np.array([[0,-axis_normalvect[2],axis_normalvect[1]],[axis_normalvect[2],0,-axis_normalvect[0]],[-axis_normalvect[1],axis_normalvect[0],0]])
		Q=I+np.sin(angle)*skew+(1-np.cos(angle)**2)*np.matmul(skew,skew)
		#newvector=np.matmul(Q,vector)
		
		for i in range(0,self.num_vertices):
			self.vertices[i].vertex_coo=np.matmul(Q,self.vertices[i].vertex_coo-self.center_of_mass)+self.center_of_mass
		for i in range(0,self.num_faces):
			self.faces[i].face_vertices[0]=np.matmul(Q,self.faces[i].face_vertices[0]-self.center_of_mass)+self.center_of_mass
			self.faces[i].face_vertices[1]=np.matmul(Q,self.faces[i].face_vertices[1]-self.center_of_mass)+self.center_of_mass
			self.faces[i].face_vertices[2]=np.matmul(Q,self.faces[i].face_vertices[2]-self.center_of_mass)+self.center_of_mass


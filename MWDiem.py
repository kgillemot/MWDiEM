#!/usr/bin/python3.5

import sys 
sys.path.append("./LIBRARIES/")
sys.path.append("./CLASSES/")

import numpy as np
import pandas as pd
import matplotlib as plt
import time
import os
from Particle_class import *
import LIBRARY_Coarse_contact_detection as Coarse_contact
import LIBRARY_Fine_contact_detection as Fine_contact
import LIBRARY_Plotting as Plot
import LIBRARY_Vtk_creator as Vtk
import LIBRARY_Force as Forcy
import LIBRARY_Write_particle_data as Writeout
	
#os.system("rm ./No_collision/*")
#os.system("rm ./Collision/*")

os.system("rm -r /home/kata/Out")
os.makedirs("/home/kata/Out")

############################# Input data ##############################

# number of particles
num_particles=2

# size of the box, where the particles are put into
size_box=1000

#number of timesteps
num_timesteps=1000000

#timestep size
timestep_size=1*10**(-4)

#mass
mass=[1,1]

#external acceleration (m/s^2)
accel_external=np.array([[0,0,-10],[0,0,0]])
force_external=accel_external*mass[0]
torque_external=np.array([[0,0,0],[0,0,0]])

#spring constanct
Ks=1*10**9
Kd=5*10**4
mu=0.3



# MAX PARTICLE SIZE ####################
max_particlesize=40

# RANDOM POLYHEDRA ######

# minimum number of corners for the polyhedra (has to be greater or eq. to 4)
num_vertices_min=4
num_vertices_max=5

# minimum and maximum radius for the polyhedra
max_radius=2
min_radius=1

# CUBE ##################
 
side_length=2

# PLANE ##################

size_plane=10
normalvector=[0,0,1]


############################# Particle generation ############################################

##### Step 1. Generate particle instances

particle=np.ndarray((num_particles), dtype='object')
for i in range(0,num_particles):
	particle[i]=Particle(i,force_external[i],mass[i])

##### Step 2. Generate particles	

particle[0].Cube_maker_3D(side_length)	
particle[1].Plane_maker_3D(normalvector,size_plane)
#for i in range(0,num_particles):
#	particle[i].Random_convex_polygon_maker_3D(num_vertices_min,num_vertices_max,min_radius,max_radius)

##### Step 3. Put particles into hash
numcells=int(size_box/max_particlesize)
cellsize=size_box/numcells
for i in range(0,num_particles):
	particle[i].Particles_to_cells_hash(cellsize,numcells)

##### Step 4. Initial positions

## Randomly translate particle
#for i in range(0,num_particles):
	#transl=[np.random.random()*(size_box-max_radius*2),np.random.random()*(size_box-max_radius*2),np.random.random()*(size_box-max_radius*2)]
	#particle[i].Translate(transl)

transl=[0,0,0.5]
particle[0].Translate(transl)
particle[0].Rotator_initial(45, np.array([0,1,0]))

##### Step 5. Initial velocities 

particle[0].velocity=np.array([0,1,-1])
particle[0].angular_velocity=np.array([0,0,1])



################ ITERATION ######################################################################

filedataout="Data_"

for i in range(0,num_timesteps):

	start = time.time()
	if i%1000==0:
		print("Timestep: "+str(i))
	if i%100==0:
		Vtk.Vtk_creator(particle,"/home/kata/Out/Out_"+str(i)+".vtk")

	#### 0. Velocity Verlet 1st phase #########################
	
	for j in range(0,num_particles):
		
		dx=particle[j].velocity*timestep_size+1/2*particle[j].force_normal/particle[j].mass*timestep_size**2
		particle[j].Translate(dx)
		particle[j].force_normal_old=particle[j].force_normal.copy()
		particle[j].force_normal=np.array([0,0,0])
		particle[j].force_n_spring=np.array([0,0,0])
		particle[j].force_n_damping=np.array([0,0,0])

		
		alfa_delta=(1/2*particle[j].angular_velocity+1/2*particle[j].angular_velocity_old)*timestep_size
		alfa=4/(4-particle[j].alfa_old*alfa_delta)*(particle[j].alfa_old+alfa_delta-1/2*np.cross(particle[j].alfa_old,alfa_delta))
		particle[j].Rotator(alfa)
		alfa_old=alfa.copy()
		particle[j].torque=np.array([0,0,0])
		
		
	##### 1. Coarse contact detection
		particle[j].Particles_to_cells_hash(cellsize,numcells)
		
	pairs_to_check=Coarse_contact.Linked_cell_hash(particle,numcells)

	##### 2. Fine contact detection via GJK an EPA and calculate force ######################################


	for j in range(0,len(pairs_to_check)):
		(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,coll)=Fine_contact.GJKuj_algorithm(particle[pairs_to_check[j][0]],particle[pairs_to_check[j][1]],pairs_to_check[j][0],pairs_to_check[j][1])	
		if coll==1:
			(penetration_depth,penetration_normal,contact_point_A,contact_point_B)=Fine_contact.EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,particle[pairs_to_check[j][0]],particle[pairs_to_check[j][1]])
			
			(Fn_A,Fn_As,Fn_Ad,TorqueA,TorqueB)=Forcy.Force(particle[pairs_to_check[j][0]],particle[pairs_to_check[j][1]],penetration_depth,penetration_normal,contact_point_A,contact_point_B,Ks,Kd,mu)
			
			particle[pairs_to_check[j][0]].force_normal=particle[pairs_to_check[j][0]].force_normal+Fn_A
			particle[pairs_to_check[j][1]].force_normal=particle[pairs_to_check[j][1]].force_normal-Fn_A			
			particle[pairs_to_check[j][0]].force_n_spring=particle[pairs_to_check[j][0]].force_n_spring+Fn_As
			particle[pairs_to_check[j][1]].force_n_spring=particle[pairs_to_check[j][1]].force_n_spring-Fn_As
			particle[pairs_to_check[j][0]].force_n_damping=particle[pairs_to_check[j][0]].force_n_damping+Fn_Ad
			particle[pairs_to_check[j][1]].force_n_damping=particle[pairs_to_check[j][1]].force_n_damping-Fn_Ad
			
			particle[pairs_to_check[j][0]].torque=particle[pairs_to_check[j][0]].torque+TorqueA
			particle[pairs_to_check[j][1]].torque=particle[pairs_to_check[j][1]].torque+TorqueB

	##### KERDES, H a PENETRATION NORMAL EGYSEGNYI HOSSZU E !!!!!!!!!!!!!!!!!!!!##############	
			
	##### 3. Velocity Verlet 2nd phase #############################################################
	
	for j in range(0,num_particles):
		particle[j].force_normal=particle[j].force_normal+particle[j].force_external
		particle[j].torque=particle[j].torque+particle[j].torque_external

		particle[j].velocity=particle[j].velocity+1/2*(particle[j].force_normal/particle[j].mass+particle[j].force_normal_old/particle[j].mass)*timestep_size
		particle[j].angular_velocity=particle[j].angular_velocity_old+(1/2*particle[j].torque_old+1/2*particle[j].torque)*timestep_size/particle[j].moment_of_inertia
		
		if i%10==0:
			Writeout.Particle_data_writer(particle[j],filedataout,i)
	
	###### 4. External constraints 
	
	particle[1].force_normal=np.array([0,0,0])
	particle[1].velocity=np.array([0,0,0])
			
	end = time.time()
	if i%1000==0:
		print("Time for timestep: ",end - start)		























############################# Particle_plotting ##############################################
#particle plotting
#partall=[]
#for i in range(0,len(particle)):
#	partall.append(particle[i])
#if coll==1:
	#partall=[]
	#partall.append(particle[0])
	#Plot.Polygons_plot_3D_multi(partall,contact_point_A)

	#partall=[]
	#partall.append(particle[1])
#if coll==1:
	#particle[0].Translate(-penetration_depth*penetration_normal)
	#Plot.Polygons_plot_3D_multi(partall,contact_point_B,contact_point_A)

#Plot.Minkowski_difference_plot_3D(particle[0],particle[1],contact_point)

##Plot.Polygon_plot_3D(mink,fout)
##Plot.Polygon_plot_3D(A,fout)
##Plot.Polygon_plot_3D(B,fout)
##Plot.Polygons_plot_3D_multi(particle)






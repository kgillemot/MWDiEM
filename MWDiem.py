#!/opt/conda/bin/python3.6

import os
import time
import matplotlib as plt
import numpy as np
import sys
sys.path.append("./LIBRARIES/")
sys.path.append("./CLASSES/")
import pandas as pd

from pathlib import Path
import shutil
import argparse

import LIBRARY_Write_particle_data as Writeout
import LIBRARY_Force as Forcy
import LIBRARY_Vtk_creator as Vtk
import LIBRARY_Plotting as Plot
import gpu_fine_contact_detection as Fine_contact
import gpu_Verlet 
import LIBRARY_Coarse_contact_detection as Coarse_contact
import LIBRARY_Quaternion as Quater
from Particle_class import *
from copy import deepcopy




parser = argparse.ArgumentParser(description='MWDiem', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--force', '-f', default=False, action='store_true',
                    help='Delete and overwrite directories, if necessary.')
parser.add_argument('--out', '-o', dest='dirname', type=os.path.expanduser, default='./Out', action='store', help='output directory')
parser.add_argument('--output', '-O', dest='filedataout', default='Data_', action='store', help='output filename')
parser.add_argument('--num_particles', '-N', type=int, default=2, help='number of particles', )
parser.add_argument('--box_size', '-s', type=int, default=100, help='size of the box, where the particles are put into', )
parser.add_argument('--timesteps', '-t', dest='num_timesteps', type=int, default=1000000, help='number of stimesteps')
parser.add_argument('--timestep_size', '-S', dest='timestep_size', type=float, default=1e-3, help='timestep size')
parser.add_argument('--mass', '-m', dest='mass', action='append', type=float, help='Masses, defined one by one. "None" means 1 for every particle.')
parser.add_argument('--seed', type=int, default=int(time.time()), help='initialize numpy random number generator with given seed. Default: current time', )

#dtype=np.double
dtype = np.float32 # todo dtype as arg
# ~ parser.add_argument('--cuda', '-f', default=False, action='store_true',
                    # ~ help='Delete and overwrite directories, if necessary.')
os.system("rm Out/*")

#
try:
	import pycuda.driver as cuda
	import pycuda.autoinit
	import gpu_fine_contact_detection as Fine_contact
	CUDA=True
	#print('PyCUDA is installed and in use.')
except:
	CUDA=False
	#print("PyCUDA hasn't been found, and it is NOT in use.")


CUDA=True
args = parser.parse_args()
if args.mass is None:
	args.mass = [1, ] * args.num_particles
assert len(args.mass) == args.num_particles, 'Number of particles should be equal with the number of defined masses.'

np.random.seed(args.seed)

if args.force:
	shutil.rmtree(args.dirname, ignore_errors=True)
Path(args.dirname).mkdir(parents=True, exist_ok=True)

############################# INPUT DATA ##############################

######### Simulation data WRITE #######################

# number of timesteps
#num_timesteps = args.num_timesteps
num_timesteps=1000000      #1000000*2

# timestep size
#timestep_size = args.timestep_size
timestep_size=1e-5 #[s]


########## System data WRITE ############################


##num_particles = args.num_particles

# slope angle
slope_angle = 19.1 #[deg]
#slope_angle = 0
slope_angle_rad=slope_angle*np.pi/180


# external acceleration and torque 
g=9800 #[mm/s^2]
acc_ext_def=np.array([g*math.sin(slope_angle_rad),0,-g*math.cos(slope_angle_rad)])  #[mm/s]  )
torq_ext_def=np.array([0,0,0])

## mass
#mass = args.mass


# spring constancts
Kn = 1e4
en = 0.8
et = 0.6
mu=0.1

Kn_wall=1e2 #1e2
en_wall=0.7 #0.6
et_wall=0.6
mu_wall=0.1

# number of types of particles
num_type=23
#num_type=3

#maxmimum particle size
max_particlesize = 7.6

########### System initialize ##############################

# what type of particle to generate
# 0 ... plane
# 1 ... spherical polyhedron
# 2 ... random polyhedron
typee=np.zeros(num_type)
size_plane_x=np.zeros(num_type)
size_plane_y=np.zeros(num_type)
size_plane_z=np.zeros(num_type)
density=np.zeros(num_type)
#how many particles of each type is made, default is 1
num_part=np.full((num_type, 1), 1)      
num_vertices_min=np.zeros(num_type)
num_vertices_max=np.zeros(num_type)
radius=np.zeros(num_type)
variation_radius=np.zeros(num_type)

########### Systme elements WRITE ###############################

wall_thickness=max_particlesize*10

### PLANE TOP ##

typee[0]=0
size_plane_x[0] = 8700   #[mm]
size_plane_y[0] = 1300   #[mm]
size_plane_z[0] = wall_thickness  #[mm]
density[0]=2640e-9  #[kg/mm^3]

## PLANE OUTFLOW TOP ##

typee[1]=0
size_plane_x[1] = 3500   #[mm]
size_plane_y[1] = 3670   #[mm]
size_plane_z[1] = wall_thickness  #[mm]
density[1]=2640e-9  #[kg/mm^3]

## PLANE SIDEWALL2 ##

typee[2]=0
size_plane_x[2] = 8700   #[mm]
size_plane_y[2] = 300   #[mm]
size_plane_z[2] = wall_thickness   #[mm]
density[2]=2640e-9  #[kg/mm^3]

## PLANE SIDEWALL1 ##

typee[3]=0
size_plane_x[3] = 8700   #[mm]
size_plane_y[3] = 300   #[mm]
size_plane_z[3] =wall_thickness   #[mm]
density[3]=2640e-9  #[kg/mm^3]


## PLANE TOP SIDE1 ##


typee[4]=0
size_plane_x[4] = 2000   #[mm]
size_plane_y[4] = 300   #[mm]
size_plane_z[4] = wall_thickness  #[mm]
density[4]=2640e-9  #[kg/mm^3]

## PLANE TOP SIDE2 ##

typee[5]=0
size_plane_x[5] = 2000   #[mm]
size_plane_y[5] = 300   #[mm]
size_plane_z[5] = wall_thickness  #[mm]
density[5]=2640e-9  #[kg/mm^3]

## PLANE END BOX BOTTOM ##

typee[6]=0
size_plane_x[6] = 300   #[mm]
size_plane_y[6] = 1300   #[mm]
size_plane_z[6] = wall_thickness  #[mm]
density[6]=2640e-9  #[kg/mm^3]


## PLANE SIDEWALL OUTFLOW 1 ##


typee[7]=0
size_plane_x[7] = 3500   #[mm]
size_plane_y[7] = 300   #[mm]
size_plane_z[7] = wall_thickness   #[mm]
density[7]=2640e-9  #[kg/mm^3]


## PLANE SIDEWALL OUTFLOW 2 ##

typee[8]=0
size_plane_x[8] = 3500   #[mm]
size_plane_y[8] = 300   #[mm]
size_plane_z[8] = wall_thickness  #[mm]
density[8]=2640e-9  #[kg/mm^3]



## PLANE END BOX OUTFLOW ##


typee[9]=0
size_plane_x[9] = 300   #[mm]
size_plane_y[9] = 3670   #[mm]
size_plane_z[9] = wall_thickness   #[mm]
density[9]=2640e-9  #[kg/mm^3]


## PLANE DEFLECTOR 1 ##

typee[10]=0
size_plane_x[10] = 2000   #[mm]
size_plane_y[10] = 300   #[mm]
size_plane_z[10] = wall_thickness  #[mm]
density[10]=2640e-9  #[kg/mm^3]

## PLANE DEFLECTOR 2 ##


typee[11]=0
size_plane_x[11] = 2000   #[mm]
size_plane_y[11] = 300   #[mm]
size_plane_z[11] = wall_thickness  #[mm]
density[11]=2640e-9  #[kg/mm^3]



## PLANE OUTFLOW EXTENSION 1 ##

typee[12]=0
size_plane_x[12] = 300   #[mm]
size_plane_y[12] = 1185   #[mm]
size_plane_z[12] = wall_thickness  #[mm]
density[12]=2640e-9  #[kg/mm^3]


## PLANE OUTFLOW EXTENSION 2 ##

typee[13]=0
size_plane_x[13] = 300   #[mm]
size_plane_y[13] = 1185   #[mm]
size_plane_z[13] = wall_thickness   #[mm]
density[13]=2640e-9  #[kg/mm^3]


## PLANE OPENING ##

typee[14]=0
size_plane_x[14] = 150   #[mm]
size_plane_y[14] = 500   #[mm]
size_plane_z[14] = wall_thickness  #[mm]
density[14]=2640e-9  #[kg/m^3]

## PLANE OPENING SIDE1 ##

typee[15]=0
size_plane_x[15] = 150    #[mm]
size_plane_y[15] = 400   #[mm]
size_plane_z[15] = wall_thickness  #[mm]
density[15]=2640e-9  #[kg/mm^3]

## PLANE OPENING SIDE2 ##

typee[16]=0
size_plane_x[16] = 150   #[mm]
size_plane_y[16] = 400   #[mm]
size_plane_z[16] = wall_thickness   #[mm]
density[16]=2640e-9  #[kg/mm^3]

## PLANE OPENING TOP ##

typee[17]=0
size_plane_x[17] = 150   #[mm]
size_plane_y[17] = 1300  #[mm]
size_plane_z[17] = wall_thickness   #[mm]
density[17]=2640e-9  #[kg/mm^3]




## PLANE BASE OF OUTFLOW ##

typee[18]=0
size_plane_x[18] = 3500   #[mm]
size_plane_y[18] = 3670   #[mm]
size_plane_z[18] = wall_thickness   #[mm]
density[18]=2640e-9  #[kg/mm^3]



## PLANE BASE ##

typee[19]=0
size_plane_x[19] = 8700   #[mm]
size_plane_y[19] = 1300   #[mm]
size_plane_z[19] = wall_thickness   #[mm]
density[19]=2640e-9  #[kg/mm^3]

### PLANE BASE ##

#typee[1]=0
#size_plane_x[1] = 8700   #[mm]
#size_plane_y[1] = 1300   #[mm]
#size_plane_z[1] = wall_thickness   #[mm]
#density[1]=2640e-9  #[kg/mm^3]

num_particles_temp=20
#num_particles_temp=2

####################### Particle data WRITE ############################

## SPHERICAL POLYHEDRA 1

typee[20]=1
num_part[20]=1000          #454719
num_vertices_min[20]= 8
num_vertices_max[20]= 14
radius[20]=6.25 #[mm]
variation_radius[20]=1.28 #[mm]
density[20]=2630e-9  #[kg/mm^3]


## SPHERICAL POLYHEDRA 2


typee[21]=1
num_part[21]=1000          #454719
num_vertices_min[21]= 8
num_vertices_max[21]= 14
radius[21]=2.54 #[mm]
variation_radius[21]=1.28 #[mm]
density[21]=2630e-9  #[kg/mm^3]

## SPHERICAL POLYHEDRA 3


typee[22]=1
num_part[22]=1000          #454719
num_vertices_min[22]= 8
num_vertices_max[22]= 14
radius[22]=1.44 #[mm]
variation_radius[22]=1.28 #s[mm]
density[22]=2640e-9  #[kg/mm^3]

## SPHERICAL POLYHEDRA 1

####typee[2]=1
####num_part[2]=1          #454719
####num_vertices_min[2]= 12
####num_vertices_max[2]= 14
####radius[2]=6.25 #[mm]
####variation_radius[2]=1.28 #[mm]
####density[2]=2630e-9  #[kg/mm^3]


### CUBE ##

#side_length = 20
#num_part[2]=1
#typee[2]=0
#num_vertices_min[2]= 8
#num_vertices_max[2]= 8
#density[2]=2630e-9  #[kg/mm^3]

## RANDOM POLYHEDRA ##

## minimum number of corners for the polyhedra (has to be greater or eq. to 4)
#num_vertices_min = 4
#num_vertices_max = 5
## minimum and maximum radius for the polyhedra
#max_radius = 2
#min_radius = 1


#################### Create the system ######################


num_particles_wall=num_particles_temp
num_particles=np.sum(num_part)
particle = np.ndarray((num_particles), dtype='object')
for i in range(0, num_particles_temp):
	particle[i] = Particle(i)
	particle[i].Plane_maker_3D(size_plane_x[i],size_plane_y[i],size_plane_z[i],density[i],max_particlesize,typee[i],acc_ext_def,torq_ext_def)

		

#particle[0].Cube_maker_3D(side_length,density,typee,acc_ext_def,torq_ext_def)
#for i in range(0,num_particles):
#	particle[i].Random_convex_polygon_maker_3D(num_vertices_min,num_vertices_max,min_radius,max_radius)

###################### Put the system together WRITE ##########################

## PLANE SIDEWALL1 ##

transl = [0, 1300/2, 300/2]
particle[3].Rotator_quaternion_initial(90, np.array([1, 0, 0]))
particle[3].Translate_wall(transl)

## PLANE SIDEWALL2 ##

transl = [0, -1300/2, 300/2]
particle[2].Rotator_quaternion_initial(90, np.array([1, 0, 0]))
particle[2].Translate_wall(transl)

## PLANE TOP ##

transl = [0, 0, 300]
particle[0].Translate_wall(transl)

## PLANE OPENING ##

transl = [-(8700/2-2000+5), 0, 150/2]
particle[14].Rotator_quaternion_initial(90, np.array([0, 1, 0]))
particle[14].Translate_wall(transl)


## PLANE OPENING SIDE1 ##

transl = [-(8700/2-2000+5), -(500/2+400/2), 150/2]
particle[15].Rotator_quaternion_initial(90, np.array([0, 1, 0]))
particle[15].Translate_wall(transl)

## PLANE OPENING SIDE2 ##

transl = [-(8700/2-2000+5), 500/2+400/2, 150/2]
particle[16].Rotator_quaternion_initial(90, np.array([0, 1, 0]))
particle[16].Translate_wall(transl)

## PLANE OPENING TOP ##

transl = [-(8700/2-2000+5), 0, 150/2+150]
particle[17].Rotator_quaternion_initial(90, np.array([0, 1, 0]))
particle[17].Translate_wall(transl)

## PLANE END BOX BOTTOM ##

transl = [-8700/2, 0, 300/2]
particle[6].Rotator_quaternion_initial(90, np.array([0, 1, 0]))
particle[6].Translate_wall(transl)

## PLANE BASE OF OUTFLOW ##

transl = [8700/2+(math.cos(slope_angle_rad)*3500)/2-200,0,math.sin(slope_angle_rad)*3500/2]
particle[18].Rotator_quaternion_initial(-19.1, np.array([0, 1, 0]))
particle[18].Translate_wall(transl)

## PLANE SIDEWALL OUTFLOW 1 ##

transl = [8700/2+(math.cos(slope_angle_rad)*3500)/2-200,3670/2,math.sin(slope_angle_rad)*3500/2+300/math.cos(slope_angle_rad)/2-10]
particle[7].Rotator_quaternion_initial(90, np.array([1, 0, 0]))
particle[7].Rotator_quaternion_initial(-19.1, np.array([0, 1, 0]))
particle[7].Translate_wall(transl)

## PLANE SIDEWALL OUTFLOW 2 ##

transl = [8700/2+(math.cos(slope_angle_rad)*3500)/2-200,-3670/2,math.sin(slope_angle_rad)*3500/2+300/math.cos(slope_angle_rad)/2-10]
particle[8].Rotator_quaternion_initial(90, np.array([1, 0, 0]))
particle[8].Rotator_quaternion_initial(-19.1, np.array([0, 1, 0]))
particle[8].Translate_wall(transl)

## PLANE OUTFLOW TOP ##

transl = [8700/2+(math.cos(slope_angle_rad)*3500)/2-200,0,math.sin(slope_angle_rad)*3500/2+300/math.cos(slope_angle_rad)-20]
particle[1].Rotator_quaternion_initial(-19.1, np.array([0, 1, 0]))
particle[1].Translate_wall(transl)


## PLANE OUTFLOW EXTENSION 1 ##

transl = [8700/2, 1300/2+(3670-1300)/2/2, 300/2+30]
particle[12].Rotator_quaternion_initial(90, np.array([0, 1, 0]))
particle[12].Translate_wall(transl)

## PLANE OUTFLOW EXTENSION 2 ##

transl = [8700/2, -(1300/2+(3670-1300)/2/2), 300/2+30]
particle[13].Rotator_quaternion_initial(90, np.array([0, 1, 0]))
particle[13].Translate_wall(transl)

## PLANE DEFLECTOR 1 ##


transl = [-(8700/2-2000+5)+math.sqrt(2000**2-400**2)/2,400/2+500/2, 300/2]
particle[10].Rotator_quaternion_initial(90, np.array([1, 0, 0]))
particle[10].Rotator_quaternion_initial(math.asin(400/2000)*180/math.pi, np.array([0, 0, 1]))
particle[10].Translate_wall(transl)

## PLANE DEFLECTOR 2 ##

transl = [-(8700/2-2000+5)+math.sqrt(2000**2-400**2)/2,-(400/2+500/2), 300/2]
particle[11].Rotator_quaternion_initial(90, np.array([1, 0, 0]))
particle[11].Rotator_quaternion_initial(-math.asin(400/2000)*180/math.pi, np.array([0, 0, 1]))
particle[11].Translate_wall(transl)

## PLANE END BOX OUTFLOW ##


transl = [8700/2+math.cos(slope_angle_rad)*3500-400, 0, math.sin(slope_angle_rad)*3500+300/2-30]
particle[9].Rotator_quaternion_initial(90+19.1, np.array([0, 1, 0]))
particle[9].Translate_wall(transl)

## PLANE TOP SIDE1 ##

transl = [-2000/2-2350, 500/2, 300/2]
particle[4].Rotator_quaternion_initial(90, np.array([1, 0, 0]))
particle[4].Translate_wall(transl)

## PLANE TOP SIDE2 ##

transl = [-2000/2-2350, -500/2, 300/2]
particle[5].Rotator_quaternion_initial(90, np.array([1, 0, 0]))
particle[5].Translate_wall(transl)


# Randomly translate particle
#for i in range(0,num_particles):
#	transl=[np.random.random()*(size_box-max_radius*2),np.random.random()*(size_box-max_radius*2),np.random.random()*(size_box-max_radius*2)]
#	particle[i].Translate(transl)





############################# Calculate system size ############################################

num_vertices_maxi=int(max(num_vertices_max))

syssize_x_max=-1e7    
syssize_x_min=1e7 
syssize_y_max=-1e7    
syssize_y_min=1e7   
syssize_z_max=-1e7    
syssize_z_min=1e7   
for i in range(0,num_particles_wall):
	for j in range(0,particle[i].num_vertices):
		if particle[i].vertices[j].vertex_coo[0]<syssize_x_min:
			syssize_x_min=particle[i].vertices[j].vertex_coo[0]
		if particle[i].vertices[j].vertex_coo[0]>syssize_x_max:
			syssize_x_max=particle[i].vertices[j].vertex_coo[0]
		if particle[i].vertices[j].vertex_coo[1]<syssize_y_min:
			syssize_y_min=particle[i].vertices[j].vertex_coo[1]
		if particle[i].vertices[j].vertex_coo[1]>syssize_y_max:
			syssize_y_max=particle[i].vertices[j].vertex_coo[1]
		if particle[i].vertices[j].vertex_coo[2]<syssize_z_min:
			syssize_z_min=particle[i].vertices[j].vertex_coo[2]
		if particle[i].vertices[j].vertex_coo[2]>syssize_z_max:
			syssize_z_max=particle[i].vertices[j].vertex_coo[2]


numcells_x = int((syssize_x_max-syssize_x_min)/max_particlesize)
numcells_y = int((syssize_y_max-syssize_y_min)/max_particlesize)
numcells_z = int((syssize_z_max-syssize_z_min)/max_particlesize)
cellsize_x = (syssize_x_max-syssize_x_min)/numcells_x
cellsize_y = (syssize_y_max-syssize_y_min)/numcells_y
cellsize_z = (syssize_z_max-syssize_z_min)/numcells_z

#for i in range(0,num_particles_wall):
	#particle[i].Wall_to_hash(cellsize_x,cellsize_y,cellsize_z,numcells_x,numcells_y)
#wall_dict=Coarse_contact.Wall_hash(particle,num_particles_wall)	

#Here is to check the place where to put the particles
#i=17			
#for j in range(0,particle[i].num_vertices):
#	print(particle[i].vertices[j].vertex_coo)

############################ Initial conditions WRITE ###########################

for i in range(0,num_particles_wall):
	particle[i].force_external=np.array([0,0,0])
	particle[i].torque_external=np.array([0,0,0])


#particle[0].velocity = np.array([0, 1, -1])
#particle[0].angular_velocity = np.array([0, 0, 10])

################ Initialize arrays for Verlet ###################################################

velocity_arr=np.zeros((num_particles,3), dtype=np.float64)
velocity_12_arr=np.zeros((num_particles,3), dtype=np.float64)
angular_velocity_arr=np.zeros((num_particles,3), dtype=np.float64)
angular_velocity_bf_arr=np.zeros((num_particles,3), dtype=np.float64)
angular_velocity_12_bf_arr=np.zeros((num_particles,3),dtype=np.float64)
dx_all_arr=np.zeros((num_particles,3),dtype=np.float64)

quaternion_arr=[]
quaternion_12_arr=[]
quaternion_all_arr=[]
vertices_arr=np.zeros((num_particles,num_vertices_maxi,3), dtype=np.float64)
mass_arr=np.zeros((num_particles,1), dtype=np.float64)
moment_of_inertia_arr=np.zeros((num_particles,3,3),dtype=np.float64)
moment_of_inertia_inv_arr=np.zeros((num_particles,3,3),dtype=np.float64)
center_of_mass_arr=np.zeros((num_particles,3),dtype=np.float64)
force_external_arr=np.zeros((num_particles,3),dtype=np.float64)
torque_external_arr=np.zeros((num_particles,3),dtype=np.float64)
force_arr=np.zeros((num_particles,3),dtype=np.float64)
torque_arr=np.zeros((num_particles,3),dtype=np.float64)
for i in range(0,num_particles):
	quaternion_arr.append([1,0,0,0])
	quaternion_12_arr.append([1,0,0,0])
	quaternion_all_arr.append([1,0,0,0])

quaternion_arr=np.asarray(quaternion_arr, dtype=np.float64)
quaternion_12_arr=np.asarray(quaternion_12_arr, dtype=np.float64)
quaternion_all_arr=np.asarray(quaternion_all_arr, dtype=np.float64)

hashh_arr=np.zeros((num_particles,1), dtype=np.int64)

for i in range(0,num_particles_temp):	
	mass_arr[i]=particle[i].mass
	for j in range(0, particle[i].num_vertices):
		for k in range(0,3):			
			vertices_arr[i,j,k]=particle[i].vertices[j].vertex_coo[k]
			
	
	for j in range(0,3):
		center_of_mass_arr[i,j]=particle[i].center_of_mass[j]
		force_external_arr[i,j]=particle[i].force_external[j]
		torque_external_arr[i,j]=particle[i].torque_external[j]
		for k in range(0,3):
			moment_of_inertia_arr[i,j,k]=particle[i].moment_of_inertia_bf[j,k]
			moment_of_inertia_inv_arr[i,j,k]=particle[i].moment_of_inertia_bf_inv[j,k]

#itt keszul el a par dictionary	
displacement_tang_hash=pd.DataFrame(columns=["Displx","Disply","Displz","change"])


#itt meg az elso abra a rendszerrol magarol
Vtk.Vtk_creator(particle, str(Path(args.dirname) / "Out_{}.vtk".format(0)),1,num_particles_wall,quaternion_all_arr,dx_all_arr)

################ Print out starting infos #######################################

if CUDA:
	print("CUDA is used")
else:
	print("CPU is used, not CUDA")
print("Number of particles: ",num_particles)
print("Number of timesteps: ",num_timesteps)
print("Timestep size: ",timestep_size)

filedataout = "./Data/Data_"
os.system("rm ./Data/Data_*")


################ ITERATION ######################################################################
#for i in range(0, 3):
for i in range(0, num_timesteps):

	if i % 1000 == 0:
		print("Timestep: "+str(i)) #, center_of_mass_arr[2])
	
	force = np.zeros((num_particles, 3),dtype=np.float32)
	torquee = np.zeros((num_particles, 3),dtype=np.float32)
	
	##################  Create a single particle every Xth step ################################
	
	start = time.time()
	#if i % 5 == 0 and num_particles_temp+3<=num_particles:	
		#for k in range(0,3):
	
	if i % 5 == 0 and num_particles_temp<num_particles:	
		for k in range(0,3):
			particle[num_particles_temp+k] = Particle(num_particles_temp+k)	
		
		####particle[num_particles_temp].Spherical_convex_polygon_maker_3D(num_vertices_min[2],num_vertices_max[2],radius[2],variation_radius[2],density[2],typee[2],acc_ext_def,torq_ext_def)
		#particle[num_particles_temp].Cube_maker_3D(side_length,density[2],typee[2],acc_ext_def,torq_ext_def)
		particle[num_particles_temp].Spherical_convex_polygon_maker_3D(num_vertices_min[20],num_vertices_max[20],radius[20],variation_radius[20],density[20],typee[20],acc_ext_def,torq_ext_def)
		particle[num_particles_temp+1].Spherical_convex_polygon_maker_3D(num_vertices_min[21],num_vertices_max[21],radius[21],variation_radius[21],density[21],typee[21],acc_ext_def,torq_ext_def)
		particle[num_particles_temp+2].Spherical_convex_polygon_maker_3D(num_vertices_min[22],num_vertices_max[22],radius[22],variation_radius[22],density[22],typee[22],acc_ext_def,torq_ext_def)
		
		for k in range(0,3):
			transl = [-np.random.random()*(4312-max_particlesize-(2393+max_particlesize))-(2393+max_particlesize),np.random.random()*(212-max_particlesize+212-max_particlesize)-(212-max_particlesize), np.random.random()*(70-max_particlesize-(38+max_particlesize))+(38+8)]
			#transl = [0,0,50]
			#particle[num_particles_temp+k].Rotator_quaternion_initial(30, np.array([0, 1, 0]))
			
			particle[num_particles_temp+k].Translate(transl)
			mass_arr[num_particles_temp+k]=particle[num_particles_temp+k].mass.copy()
			for l in range(0, particle[num_particles_temp+k].num_vertices):
				for m in range(0,3):			
					vertices_arr[num_particles_temp+k,l,m]=particle[num_particles_temp+k].vertices[l].vertex_coo[m].copy()
			for l in range(0,3):
				center_of_mass_arr[num_particles_temp+k,l]=particle[num_particles_temp+k].center_of_mass[l].copy()
				force_external_arr[num_particles_temp+k,l]=particle[num_particles_temp+k].force_external[l].copy()
				torque_external_arr[num_particles_temp+k,l]=particle[num_particles_temp+k].torque_external[l].copy()
				for m in range(0,3):
					moment_of_inertia_arr[num_particles_temp+k,l,m]=particle[num_particles_temp+k].moment_of_inertia_bf[l,m].copy()
					moment_of_inertia_inv_arr[num_particles_temp+k,l,m]=particle[num_particles_temp+k].moment_of_inertia_bf_inv[l,m].copy()
		
		#num_particles_temp=num_particles_temp+3
		num_particles_temp=num_particles_temp+1

	end = time.time()
	#if i % 100 == 0:
		#print("Number of particles now: ", num_particles_temp-num_particles_wall)
		#print("Time for creating particles: ", end - start)

	#################### Check for flown away particles ########################### DEL, if GPU OK!
	
	#start = time.time()
	
	#for j in range(num_particles_wall, num_particles_temp):		
		#if particle[j].center_of_mass[0]>syssize_x_max or particle[j].center_of_mass[1]>syssize_y_max or particle[j].center_of_mass[2]>syssize_z_max or particle[j].center_of_mass[0]<syssize_x_min or particle[j].center_of_mass[1]<syssize_y_min or particle[j].center_of_mass[2]<syssize_z_min:
			#print("flewn away")
			#particle[j].Translate(-particle[j].center_of_mass)
			#particle[j].Translate([-np.random.random()*(4312-max_particlesize-(2393+max_particlesize))-(2393+max_particlesize),np.random.random()*(212-max_particlesize+212-max_particlesize)-(212-max_particlesize), np.random.random()*(262-max_particlesize-(38+max_particlesize))+(38+8)])
			#particle[j].velocity=np.array([0,0,0])
			#particle[j].velocity_12=np.array([0,0,0])
			#particle[j].force_normal=np.array([0,0,0])
			#particle[j].force_n_spring=np.array([0,0,0])
			#particle[j].force_n_damping=np.array([0,0,0])
			#particle[j].torque=np.array([0,0,0])
			#particle[j].torque_bf=np.array([0,0,0])
			#particle[j].angular_velocity_bf=np.array([0,0,0])
			#particle[j].angular_velocity_12_bf=np.array([0,0,0])
			#particle[j].angular_velocity=np.array([0,0,0])
			#particle[j].angular_velocity_12=np.array([0,0,0])
			#particle[j].angular_acceleration_bf=np.array([0,0,0])
			#particle[j].quaternion=np.array([1,0,0,0])
			#particle[j].quaternion_12=np.array([1,0,0,0])
			#particle[j].quaternion_all=np.array([1,0,0,0])
			##dx=np.array([0,0,0])
	
	#end = time.time()
	#if i % 100 == 0:
		#print("Time for checking for flown away part: ", end - start)
		
    #################### Velocity Verlet 1st phase ######################### Del if GPU OK!
    
	#start = time.time()
	if i==0:
		for j in range(num_particles_wall, num_particles_temp):	
			dx=particle[j].velocity_12*timestep_size
			#print("dx",dx)		
			angular_velocity_34_bf=particle[j].angular_velocity_12_bf+1/4*particle[j].angular_acceleration_bf*timestep_size		
			#print("av34bf",angular_velocity_34_bf)
			angular_velocity_34=Quater.Quat_triple_prod(particle[j].quaternion_12,angular_velocity_34_bf,Quater.Quat_inv(particle[j].quaternion_12)) 
			#print("av34",angular_velocity_34)    		
			
			if np.linalg.norm(angular_velocity_34)!=0:
				temp1=np.array(np.sin(np.linalg.norm(angular_velocity_34)*timestep_size/4)*angular_velocity_34/np.linalg.norm(angular_velocity_34))
			else:
				temp1=np.array([0,0,0])
			#print("temp1",temp1)
			temp2=np.array([np.cos(np.linalg.norm(angular_velocity_34)*timestep_size/4),temp1[0],temp1[1],temp1[2]])
			quaternion_old=particle[j].quaternion.copy()
			particle[j].quaternion=Quater.Quat_multipl(temp2,particle[j].quaternion_12) 
			#print("q",particle[j].quaternion)  		
			particle[j].quaternion_all=Quater.Quat_multipl(particle[j].quaternion,quaternion_old)
			#print("qall",particle[j].quaternion_all)  
			particle[j].angular_velocity_bf=particle[j].angular_velocity_12_bf+1/2*particle[j].angular_acceleration_bf*timestep_size
			#print("avbf",particle[j].angular_velocity_bf)	
			particle[j].angular_velocity=Quater.Quat_triple_prod(particle[j].quaternion,particle[j].angular_velocity_bf,Quater.Quat_inv(particle[j].quaternion))  
			#print("av",particle[j].angular_velocity)
			particle[j].velocity=particle[j].velocity_12+1/particle[j].mass*particle[j].force_normal*timestep_size
			#print("v",particle[j].velocity)
			#itt biztos, hogy van egy kis numerikus bizonytalansag, mert nagyon enyhen csokken a volume of the particle, + a quaternion elso eleme sem egzaktul 0
			particle[j].Rotator_quaternion(particle[j].quaternion)
			
			#if j==2:
				#for k in range(0,particle[j].num_vertices):
					#print("CPU vert",particle[j].vertices[k].vertex_coo)
			#print("CPU q",particle[j].quaternion)
			particle[j].Translate(dx)
			
			quaternion_arr[j]=particle[j].quaternion.copy()
			quaternion_all_arr[j]=particle[j].quaternion_all.copy()
			angular_velocity_bf_arr[j]=particle[j].angular_velocity_bf.copy()
			angular_velocity_arr[j]=particle[j].angular_velocity.copy()
			velocity_arr[j]=particle[j].velocity.copy()
			
			

	
	#end = time.time()
	#if i % 100 == 0:
		#print("Time for Verlet Pahse1: ", end - start)

	#for j in range(num_particles_wall, num_particles_temp):	
	#	particle[j].center_of_mass=center_of_mass_arr[j].copy()
	################### Coarse contact detection ######################################
	start = time.time()
		
	pairs_to_check=[]	
	for k in range(0,num_particles_wall):
		for j in range(num_particles_wall, num_particles_temp):		
			if abs(np.dot(particle[k].normvec,center_of_mass_arr[j]-particle[k].center_of_mass))<6*max_particlesize: # and np.linalg.norm(particle[j].center_of_mass-particle[k].center_of_mass)<particle[k].maxsize+max_particlesize:
				pairs_to_check.append([j,k])	
		#particle[j].Particles_to_cells_hash(cellsize_x,cellsize_y,cellsize_z,numcells_x,numcells_y)
	xc=(center_of_mass_arr[:,0]/cellsize_x).astype(int)
	yc=(center_of_mass_arr[:,1]/cellsize_y).astype(int)
	zc=(center_of_mass_arr[:,2]/cellsize_z).astype(int)
	hashh_arr=xc+yc*numcells_x+zc*numcells_x*numcells_y
	pairs_to_check = Coarse_contact.Linked_cell_hash(particle, numcells_x,numcells_y,num_particles_wall,num_particles_temp,pairs_to_check,hashh_arr)
	end = time.time()
	#if i % 100 == 0:
		#print("Time for Coarse contact detection: ", end - start)
	
	#print(center_of_mass_arr[num_particles_wall:num_particles_temp])	
    ###################### Fine contact detection via GJK an EPA and calculate force #######################

	if pairs_to_check:
		start = time.time()
		
		particle_A_center_of_mass = []
		particle_B_center_of_mass = []
		particle_A_vertices = []
		particle_B_vertices = []
		particle_A_velocity = []
		particle_A_angular_velocity = []
		particle_B_angular_velocity = []
		particle_B_velocity = []
		particle_A_numvertices = []
		particle_B_numvertices = []
		particle_A_id = []
		particle_B_id = []
		displacement_tang=[]
		quaternion_A=[]
		quaternion_B=[]
		mom_inertia_size_A=[]
		mom_inertia_size_B=[]
		

		for j in range(0, len(pairs_to_check)):
			particle_A_center_of_mass.append(center_of_mass_arr[pairs_to_check[j][0]])
			particle_B_center_of_mass.append(center_of_mass_arr[pairs_to_check[j][1]])
			vertA = []
			vertB = []
			for k in range(0, particle[pairs_to_check[j][0]].num_vertices):
				vertA.append(vertices_arr[pairs_to_check[j][0]][k])
			for k in range(particle[pairs_to_check[j][0]].num_vertices, num_vertices_maxi):
				vertA.append([0,0,0])
			for k in range(0, particle[pairs_to_check[j][1]].num_vertices):
				vertB.append(vertices_arr[pairs_to_check[j][1]][k])
			for k in range(particle[pairs_to_check[j][1]].num_vertices, num_vertices_maxi):
				vertB.append([0,0,0])
			particle_A_vertices.append(vertA)
			particle_B_vertices.append(vertB)
			
			particle_A_velocity.append(velocity_arr[pairs_to_check[j][0]])
			particle_B_velocity.append(velocity_arr[pairs_to_check[j][1]])
			
			particle_A_angular_velocity.append(angular_velocity_arr[pairs_to_check[j][0]])
			particle_B_angular_velocity.append(angular_velocity_arr[pairs_to_check[j][1]])
			particle_A_numvertices.append(particle[pairs_to_check[j][0]].num_vertices)
			particle_B_numvertices.append(particle[pairs_to_check[j][1]].num_vertices)
			particle_A_id.append(pairs_to_check[j][0])
			particle_B_id.append(pairs_to_check[j][1])
			if pairs_to_check[j][0]<pairs_to_check[j][1]:
				contact_hash=pairs_to_check[j][0]+pairs_to_check[j][1]**2
			else:
				contact_hash=pairs_to_check[j][1]+pairs_to_check[j][0]**2
			if (contact_hash in displacement_tang_hash.index.values)==False:
				##displacement_tang_hash['Displx'].loc[contact_hash]+=0
			##except:
				displacement_tang_hash.loc[contact_hash]=[0,0,0,0]
			displacement_tang.append(displacement_tang_hash.loc[contact_hash])
			quaternion_A.append(quaternion_arr[pairs_to_check[j][0]])
			quaternion_B.append(quaternion_arr[pairs_to_check[j][1]])
			mom_inertia_size_A.append(particle[pairs_to_check[j][0]].moment_of_inertia_size)
			mom_inertia_size_B.append(particle[pairs_to_check[j][1]].moment_of_inertia_size)
	   
		
		particle_A_center_of_mass = np.asarray(particle_A_center_of_mass, dtype=dtype)
		particle_B_center_of_mass = np.asarray(particle_B_center_of_mass, dtype=dtype)
		particle_A_vertices = np.asarray(particle_A_vertices, dtype=dtype)
		particle_B_vertices = np.asarray(particle_B_vertices, dtype=dtype)
		particle_A_velocity = np.asarray(particle_A_velocity, dtype=dtype)
		particle_B_velocity = np.asarray(particle_B_velocity, dtype=dtype)
		particle_A_angular_velocity = np.asarray(particle_A_angular_velocity, dtype=dtype)
		particle_B_angular_velocity = np.asarray(particle_B_angular_velocity, dtype=dtype)
		particle_A_numvertices= np.asarray(particle_A_numvertices, dtype=np.int32)
		particle_B_numvertices= np.asarray(particle_B_numvertices, dtype=np.int32)
		particle_A_id = np.asarray(particle_A_id, dtype=np.int32)
		particle_B_id = np.asarray(particle_B_id, dtype=np.int32)
		displacement_tang=np.asarray(displacement_tang, dtype=dtype)
		quaternion_A = np.asarray(quaternion_A, dtype=dtype)
		quaternion_B = np.asarray(quaternion_B, dtype=dtype)
		mom_inertia_size_A=np.asarray(mom_inertia_size_A, dtype=dtype)
		mom_inertia_size_B=np.asarray(mom_inertia_size_B, dtype=dtype)

		#print(pairs_to_check)
		
		#print(displacement_tang_hash)
		end = time.time()
		#if i % 100 == 0:
		#	print("Time for Fill for Fine contact detection: ", end - start)

		start = time.time()
		
		#print("q",quaternion_arr)
		
		if CUDA:
			(force,torquee,displacement_tang,collision)=\
					Fine_contact.Contact_detection_GPU(\
								particle_A_center_of_mass,   particle_B_center_of_mass,\
								particle_A_vertices,	 	 particle_B_vertices,\
								particle_A_velocity,		 particle_B_velocity,\
								particle_A_angular_velocity, particle_B_angular_velocity,\
								particle_A_numvertices,      particle_B_numvertices,\
								particle_A_id, 			     particle_B_id,\
								displacement_tang,\
								Kn, en, et, mu, \
								Kn_wall, en_wall, et_wall, mu_wall, num_particles_wall,\
								force,                       torquee,\
								num_vertices_maxi,\
								quaternion_A,    			 quaternion_B,\
								timestep_size,\
								mom_inertia_size_A,   		 mom_inertia_size_B)	

		else:
			(force,torquee,displacement_tang,collision)=\
					Fine_contact.Contact_detection_GPU(\
								particle_A_center_of_mass,   particle_B_center_of_mass,\
								particle_A_vertices,	 	 particle_B_vertices,\
								particle_A_velocity,		 particle_B_velocity,\
								particle_A_angular_velocity, particle_B_angular_velocity,\
								particle_A_numvertices,      particle_B_numvertices,\
								particle_A_id, 			     particle_B_id,\
								displacement_tang,\
								Kn, en, et, mu, \
								Kn_wall, en_wall, et_wall, mu_wall, num_particles_wall,\
								force,                       torquee,\
								num_vertices_maxi,\
								quaternion_A,    			 quaternion_B,\
								timestep_size,\
								mom_inertia_size_A,   		 mom_inertia_size_B)	
								
									
		for j in range(0,len(pairs_to_check)):
			if particle_A_id[j]<particle_B_id[j]:
				contact_hash=particle_A_id[j]+particle_B_id[j]**2
			else:
				contact_hash=particle_B_id[j]+particle_A_id[j]**2

			displacement_tang_hash.loc[contact_hash,'Displx']=displacement_tang[j][0]
			displacement_tang_hash.loc[contact_hash,'Disply']=displacement_tang[j][1]
			displacement_tang_hash.loc[contact_hash,'Displz']=displacement_tang[j][2]
			displacement_tang_hash.loc[contact_hash,'change']=displacement_tang[j][3]

		
		#itt kitorlom a 0 contact time-osokat
		displacement_tang_hash = displacement_tang_hash.drop(displacement_tang_hash['change'].isnull().index)
		#displacement_tang_hash['change']=0
		
		end = time.time()
		#if len(collision)!=0:    #i % 100 == 0:
			#print("Time for GPU Fine contact detection: ", end - start)
			#print("Collision info: ", len(collision),sum(collision))
    
    ########################## Velocity Verlet GPU ############################################################
	start = time.time()
	
	force[0:num_particles_wall,:]=0.
	torquee[0:num_particles_wall,:]=0.

	force_arr=force+force_external_arr
	torque_arr=torquee+torque_external_arr	
	
	
	#Valszeg az baj, h a force es a torque csak float32!
	if CUDA:
		(velocity_12_arr,           quaternion_arr,       angular_velocity_bf_arr,\
		angular_velocity_12_bf_arr, angular_velocity_arr, quaternion_12_arr,\
		velocity_arr,               center_of_mass_arr,   vertices_arr,\
		quaternion_all_arr,         dx_all_arr)\
		\
		=gpu_Verlet.Verlet_GPU(\
		\
		velocity_arr,               velocity_12_arr,      angular_velocity_12_bf_arr,\
		quaternion_12_arr,   		quaternion_arr,       quaternion_all_arr,\
		angular_velocity_bf_arr,	angular_velocity_arr, force_arr,\
		torque_arr,         		vertices_arr,         center_of_mass_arr,\
		timestep_size, 		        mass_arr,             num_particles,\
		moment_of_inertia_inv_arr,	moment_of_inertia_arr,dx_all_arr,\
		num_vertices_maxi,      	syssize_x_max,   	  syssize_x_min,\
		syssize_y_max,      		syssize_y_min,        syssize_z_max,\
		syssize_z_min,              num_particles_wall,   num_particles_temp)
	
	#print(velocity_arr)
		
	end = time.time()
	#if i % 100 == 0:
		#print("Time for GPU Verlet: ", end - start)
	
	
		#print(quaternion_arr)
	########################### Velocity Verlet 2nd Phase ############################# Del if GPU OK!
	#start = time.time()
	
	#for j in range(num_particles_wall, num_particles_temp):
		#particle[j].force_normal = force[j]+particle[j].force_external
		##print("force %.10f",particle[j].force_normal)
		#particle[j].torque= torquee[j]+particle[j].torque_external
		##print("torque %.10f",particle[j].torque)	
		#velocity_32=particle[j].velocity_12+1/particle[j].mass*particle[j].force_normal*timestep_size
		##print("v32 %.10f",velocity_32)
		#particle[j].torque_bf=Quater.Quat_triple_prod(Quater.Quat_inv(particle[j].quaternion),particle[j].torque,particle[j].quaternion)
		##print("t_bf",particle[j].torque_bf)

		#particle[j].angular_acceleration_bf=np.matmul(particle[j].moment_of_inertia_bf_inv,particle[j].torque_bf-np.cross(particle[j].angular_velocity_bf,np.matmul(particle[j].moment_of_inertia_bf,particle[j].angular_velocity_bf)))
		##print("angacc_bf",particle[j].angular_acceleration_bf)
		#angular_acceleration=Quater.Quat_triple_prod(particle[j].quaternion,particle[j].angular_acceleration_bf,Quater.Quat_inv(particle[j].quaternion))
		##print("angacc",angular_acceleration)
		#angular_velocity_32_bf=particle[j].angular_velocity_12_bf+particle[j].angular_acceleration_bf*timestep_size
		##print("angv32bf",angular_velocity_32_bf)
		#if np.linalg.norm(particle[j].angular_velocity)!=0:
			#temp11=np.sin(np.linalg.norm(particle[j].angular_velocity)*timestep_size/2)*particle[j].angular_velocity/np.linalg.norm(particle[j].angular_velocity)
		#else:
			#temp11=np.array([0,0,0])
		#temp22=np.array([np.cos(np.linalg.norm(particle[j].angular_velocity)*timestep_size/2),temp11[0],temp11[1],temp11[2]])
		#quaternion_32=Quater.Quat_multipl(temp22,particle[j].quaternion_12) 
		##print("q32",quaternion_32)     
		#angular_velocity_32=Quater.Quat_triple_prod(quaternion_32,angular_velocity_32_bf,Quater.Quat_inv(quaternion_32))
		##print("av32",angular_velocity_32) 
		#particle[j].velocity_12=velocity_32.copy()
		##print("v12",particle[j].velocity_12) 	
		#particle[j].quaternion_12=quaternion_32.copy()
		##print("q12",particle[j].quaternion_12) 
		#particle[j].angular_velocity_12=angular_velocity_32.copy()
		##print("av12",particle[j].angular_velocity_12)
		#particle[j].angular_velocity_12_bf=angular_velocity_32_bf.copy()
		##print("av12bf",particle[j].angular_velocity_12_bf)
	
	#end = time.time()
	#if i % 100 == 0:
		#print("Time for Verlet2: ", end - start)

	########################## Write out particle data #################################
	
	if i % 1000 == 0:	
		for j in range(num_particles_wall, num_particles_temp):			
			Writeout.Particle_data_writer(particle[j], filedataout, i)

	
	if i % 1000 == 1:
		quaternion_all_arr,dx_all_arr=Vtk.Vtk_creator(particle, str(Path(args.dirname) / "Out_{}.vtk".format(i)),0,num_particles_temp,quaternion_all_arr,dx_all_arr)

	################### Check volume conservation TEST ##############################
	#for j in range(num_particles_wall, num_particles_temp):
		#polyhedra=[]
		#for k in range(0,particle[j].num_vertices):
			#polyhedra.append([vertices_arr[j,k,0],vertices_arr[j,k,1],vertices_arr[j,k,2]])
			##print([vertices_arr[j,k,0],vertices_arr[j,k,1],vertices_arr[j,k,1]])
		#polyhedra=np.array(polyhedra)	
	
		#hull= ConvexHull(polyhedra)
		#volume = ConvexHull(polyhedra).volume
		#print("vGPU",volume)
		
	#for j in range(num_particles_wall, num_particles_temp):
		#polyhedra=[]
		#for k in range(0,particle[j].num_vertices):
			#polyhedra.append([particle[j].vertices[k].vertex_coo[0],particle[j].vertices[k].vertex_coo[1],particle[j].vertices[k].vertex_coo[2]])
		#polyhedra=np.array(polyhedra)	
	
		#hull= ConvexHull(polyhedra)
		#volume = ConvexHull(polyhedra).volume
		#print("vCPU",volume)

############################# Particle_plotting ##############################################
# particle plotting
# partall=[]
# for i in range(0,len(particle)):
#   partall.append(particle[i])
# if coll==1:
    # partall=[]
    # partall.append(particle[0])
    # Plot.Polygons_plot_3D_multi(partall,contact_point_A)

    # partall=[]
    # partall.append(particle[1])
# if coll==1:
    # particle[0].Translate(-penetration_depth*penetration_normal)
    # Plot.Polygons_plot_3D_multi(partall,contact_point_B,contact_point_A)

# Plot.Minkowski_difference_plot_3D(particle[0],particle[1],contact_point)

# Plot.Polygon_plot_3D(mink,fout)
# Plot.Polygon_plot_3D(A,fout)
# Plot.Polygon_plot_3D(B,fout)
# Plot.Polygons_plot_3D_multi(particle)

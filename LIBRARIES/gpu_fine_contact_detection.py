import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import sys
from jinja2 import Template
import numpy as np
from pathlib import Path
import time

def gpu_contact_detection_init(**args):
	with open(Path(__file__).parent / "gpu_fine_contact_detection.cu") as f:
		template = Template(f.read())
	gpu_code = template.render(**args)
	mod = SourceModule(gpu_code)
	return mod.get_function("contact_detection")


def gpu_contact_detection(calc_function, particle_A_center_of_mass,particle_B_center_of_mass,particle_A_vertices,particle_B_vertices,particle_A_velocity,particle_B_velocity,particle_A_angular_velocity, particle_B_angular_velocity, particle_A_numvertices,particle_B_numvertices,particle_A_id,particle_B_id,particle_contacttime,force,torque):

	num_pairs_P = np.asarray(particle_A_center_of_mass.shape[0], dtype=np.int32)  
	tempi = np.zeros(shape = (3,3,3), dtype=np.float32 )
	block = (64, 1, 1)
	grid = (int(num_pairs_P / 1024 / 64 + 1),1024) #ez a jobb valszeg
	collision=np.zeros(shape=particle_A_center_of_mass.shape[0], dtype=np.int32)
	pendepth=np.zeros(shape=particle_A_center_of_mass.shape[0], dtype=np.float32)
	tochecki=np.zeros(shape=particle_A_center_of_mass.shape[0], dtype=np.float32)
	pair_IDs=np.zeros(shape=(particle_A_center_of_mass.shape[0],2), dtype=np.int32)
	calc_function(cuda.In(particle_A_center_of_mass),cuda.In(particle_B_center_of_mass),cuda.In(particle_A_vertices),cuda.In(particle_B_vertices),cuda.In(particle_A_velocity),cuda.In(particle_B_velocity),cuda.In(particle_A_angular_velocity), cuda.In(particle_B_angular_velocity),cuda.In(particle_A_numvertices),cuda.In(particle_B_numvertices),cuda.In(particle_A_id),cuda.In(particle_B_id),cuda.InOut(particle_contacttime),cuda.InOut(force),cuda.InOut(torque),cuda.In(num_pairs_P),cuda.InOut(pair_IDs),cuda.InOut(collision),cuda.InOut(pendepth),cuda.InOut(tochecki),block=block,grid=grid)
	#if collision[0]==1:
	#	print("Collision info: ",collision)
	#	print(particle_contacttime)
	#printi=0
	#for i in range(0,len(collision)):
		#if collision[i]>0:
			#printi=1
			##print(collision)
	#if printi==1:		
		#print(particle_contacttime)
	return force,torque,tempi,collision, particle_contacttime


def Contact_detection_GPU(particle_A_center_of_mass,particle_B_center_of_mass,particle_A_vertices,particle_B_vertices,particle_A_velocity,particle_B_velocity,particle_A_angular_velocity, particle_B_angular_velocity,particle_A_numvertices,particle_B_numvertices,particle_A_id,particle_B_id,particle_contacttime,Ks,Kd,Kt,mu,force,torque,num_vertices_max):

	start = time.time()
	calc_func = gpu_contact_detection_init(Ks=Ks,Kd=Kd,Kt=Kt,mu=mu,num_pairs=particle_A_center_of_mass.shape[0],num_vertices_max=num_vertices_max)
	force,torque,tempi,collision,particle_contacttime = gpu_contact_detection(calc_func, particle_A_center_of_mass,particle_B_center_of_mass,particle_A_vertices,particle_B_vertices,particle_A_velocity,particle_B_velocity,particle_A_angular_velocity, particle_B_angular_velocity,particle_A_numvertices,particle_B_numvertices,particle_A_id,particle_B_id,particle_contacttime,force,torque)
	end = time.time()
	#print("Time for run ",end - start)
	

	return (force,torque,particle_contacttime)



    

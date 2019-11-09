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


def gpu_contact_detection(calc_function, particle_A_center_of_mass,particle_B_center_of_mass,particle_A_vertices,particle_B_vertices,particle_A_velocity,particle_B_velocity,particle_A_numvertices,particle_B_numvertices,particle_A_id,particle_B_id,force):

	num_pairs_P = np.asarray(particle_A_center_of_mass.shape[0], dtype=np.int32)  
	tempi = np.zeros(shape = (3,3,3), dtype=np.float32 )
	block = (64, 1, 1)
	#grid = (1024, int(num_pairs_P / 1024 / 64 + 1))
	grid = (int(num_pairs_P / 1024 / 64 + 1),1024) #ez a jobb valszeg

	collision=np.zeros(shape=particle_A_center_of_mass.shape[0], dtype=np.int32)
	pendepth=np.zeros(shape=particle_A_center_of_mass.shape[0], dtype=np.double)
	tochecki=np.zeros(shape=particle_A_center_of_mass.shape[0], dtype=np.float32)

	#print(particle_A_vertices)
	pair_IDs=np.zeros(shape=(particle_A_center_of_mass.shape[0],2), dtype=np.int32)
	calc_function(cuda.In(particle_A_center_of_mass),cuda.In(particle_B_center_of_mass),cuda.In(particle_A_vertices),cuda.In(particle_B_vertices),cuda.In(particle_A_velocity),cuda.In(particle_B_velocity),cuda.In(particle_A_numvertices),cuda.In(particle_B_numvertices),cuda.In(particle_A_id),cuda.In(particle_B_id),cuda.InOut(force),cuda.In(num_pairs_P),cuda.InOut(pair_IDs),cuda.InOut(collision),cuda.InOut(pendepth),cuda.InOut(tochecki),block=block,grid=grid)
	#print(pair_IDs)
	#print(colltype)
	print(pair_IDs)
	print("Coll",collision)
	print(pendepth)
	return force,tempi,collision


def Contact_detection_GPU(particle_A_center_of_mass,particle_B_center_of_mass,particle_A_vertices,particle_B_vertices,particle_A_velocity,particle_B_velocity,particle_A_numvertices,particle_B_numvertices,particle_A_id,particle_B_id,Ks,Kd,mu,force,num_vertices_max):

	start = time.time()
	calc_func = gpu_contact_detection_init(Ks=Ks,Kd=Kd,mu=mu,num_pairs=particle_A_center_of_mass.shape[0],num_vertices_max=num_vertices_max)
	force,tempi,collision = gpu_contact_detection(calc_func, particle_A_center_of_mass,particle_B_center_of_mass,particle_A_vertices,particle_B_vertices,particle_A_velocity,particle_B_velocity,particle_A_numvertices,particle_B_numvertices,particle_A_id,particle_B_id,force)
	end = time.time()
	#print("Time for run ",end - start)
	

	#return collision
	sys.exit(0)


    

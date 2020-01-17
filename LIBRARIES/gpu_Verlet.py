import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import sys
from jinja2 import Template
import numpy as np
from pathlib import Path
import time

def gpu_Verlet_init(**args):
	with open(Path(__file__).parent / "gpu_Verlet.cu") as f:
		template = Template(f.read())
	gpu_code = template.render(**args)
	mod = SourceModule(gpu_code)
	return mod.get_function("Verlet")


def gpu_Verlett(calc_function,velocity_arr,velocity_12_arr,angular_velocity_12_bf_arr,quaternion_12_arr,quaternion_arr,quaternion_all_arr,angular_velocity_bf_arr,angular_velocity_arr,force_arr,torque_arr,vertices_arr,center_of_mass_arr,mass_arr,moment_of_inertia_inv_arr,moment_of_inertia_arr,dx_all_arr):

	num_part_P = np.asarray(center_of_mass_arr.shape[0], dtype=np.int32)  
	block = (64, 1, 1)
	grid = (int(num_part_P / 1024 / 64 + 1),1024) 
	calc_function(cuda.InOut(velocity_arr),cuda.InOut(velocity_12_arr),cuda.InOut(angular_velocity_12_bf_arr),cuda.InOut(quaternion_12_arr),cuda.InOut(quaternion_arr),cuda.InOut(quaternion_all_arr),cuda.InOut(angular_velocity_bf_arr),cuda.InOut(angular_velocity_arr),cuda.In(force_arr),cuda.In(torque_arr),cuda.InOut(vertices_arr),cuda.InOut(center_of_mass_arr),cuda.In(mass_arr),cuda.In(moment_of_inertia_inv_arr),cuda.In(moment_of_inertia_arr),cuda.InOut(dx_all_arr),cuda.In(num_part_P),block=block,grid=grid)
	return (velocity_12_arr,quaternion_arr,angular_velocity_bf_arr,angular_velocity_12_bf_arr,angular_velocity_arr,quaternion_12_arr,velocity_arr,center_of_mass_arr,vertices_arr,quaternion_all_arr,dx_all_arr)


def Verlet_GPU(velocity_arr,velocity_12_arr,angular_velocity_12_bf_arr,quaternion_12_arr,quaternion_arr,quaternion_all_arr,angular_velocity_bf_arr,angular_velocity_arr,force_arr,torque_arr,vertices_arr,center_of_mass_arr,timestep_size,mass_arr,num_particles,moment_of_inertia_inv_arr,moment_of_inertia_arr,dx_all_arr,num_vertices_max,syssize_x_max,syssize_x_min,syssize_y_max,syssize_y_min,syssize_z_max,syssize_z_min):

	calc_func = gpu_Verlet_init(timestep_size=timestep_size,num_particles=num_particles,num_vertices_max=num_vertices_max,syssize_x_max=syssize_x_max,syssize_x_min=syssize_x_min,syssize_y_max=syssize_y_max,syssize_y_min=syssize_y_min,syssize_z_max=syssize_z_max,syssize_z_min=syssize_z_min)
	velocity_12_arr,quaternion_arr,angular_velocity_bf_arr,angular_velocity_12_bf_arr,angular_velocity_arr,quaternion_12_arr,velocity_arr,center_of_mass_arr,vertices_arr,quaternion_all_arr,dx_all_arr = gpu_Verlett(calc_func,velocity_arr,velocity_12_arr,angular_velocity_12_bf_arr,quaternion_12_arr,quaternion_arr,quaternion_all_arr,angular_velocity_bf_arr,angular_velocity_arr,force_arr,torque_arr,vertices_arr,center_of_mass_arr,mass_arr,moment_of_inertia_inv_arr,moment_of_inertia_arr,dx_all_arr)
	return (velocity_12_arr,quaternion_arr,angular_velocity_bf_arr,angular_velocity_12_bf_arr,angular_velocity_arr,quaternion_12_arr,velocity_arr,center_of_mass_arr,vertices_arr,quaternion_all_arr,dx_all_arr)

    

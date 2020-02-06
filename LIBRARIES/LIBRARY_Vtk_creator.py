import numpy as np

def Vtk_creator(particle,filename,num_part_min,num_part_max,quaternion_all_arr,dx_all_arr):
	vtk_out=open(filename,"w")
	vtk_out.write("# vtk DataFile Version 2.0\n")
	vtk_out.write("# kata\n")
	vtk_out.write("ASCII\n")
	vtk_out.write("DATASET POLYDATA\n")
	num_points=0
	
	for i in range(num_part_min,num_part_max):
		particle[i].quaternion_all=quaternion_all_arr[i].copy()
		particle[i].dx_all=dx_all_arr[i].copy()
		particle[i].Rotator_quaternion_all()
		particle[i].Translate_all()
		quaternion_all_arr[i]=np.array([1,0,0,0])
		dx_all_arr[i]=np.array([0,0,0])
		
	
	
	for i in range(num_part_min,num_part_max):
		num_points=num_points+particle[i].num_vertices
	vtk_out.write("POINTS\t"+str(num_points)+"\tfloat\n")
	for i in range(num_part_min,num_part_max):
		for j in range(particle[i].num_vertices):
			vtk_out.write(str(particle[i].vertices[j].vertex_coo[0])+"\t"+str(particle[i].vertices[j].vertex_coo[1])+"\t"+str(particle[i].vertices[j].vertex_coo[2])+"\n")
	num_faces=0
	for i in range(num_part_min,num_part_max):
		num_faces=num_faces+particle[i].num_faces
	vtk_out.write("POLYGONS\t"+str(num_faces)+"\t"+str(num_faces*4)+"\n")
	

	
	addon=0
	for i in range(num_part_min,num_part_max):
		for j in range(particle[i].num_faces):
			vtk_out.write(str(3)+"\t"+str(particle[i].faces[j].face_vertices_id[0]+addon)+"\t"+str(particle[i].faces[j].face_vertices_id[1]+addon)+"\t"+str(particle[i].faces[j].face_vertices_id[2]+addon)+"\n")
		addon=addon+particle[i].num_vertices
	vtk_out.close()
	
	return(quaternion_all_arr,dx_all_arr)

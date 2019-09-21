def Vtk_creator(particle,filename):
	vtk_out=open(filename,"w")
	vtk_out.write("# vtk DataFile Version 2.0\n")
	vtk_out.write("# kata\n")
	vtk_out.write("ASCII\n")
	vtk_out.write("DATASET POLYDATA\n")
	num_points=0
	for i in range(len(particle)):
		num_points=num_points+particle[i].num_vertices
	vtk_out.write("POINTS\t"+str(num_points)+"\tfloat\n")
	for i in range(len(particle)):
		for j in range(particle[i].num_vertices):
			vtk_out.write(str(particle[i].vertices[j].vertex_coo[0])+"\t"+str(particle[i].vertices[j].vertex_coo[1])+"\t"+str(particle[i].vertices[j].vertex_coo[2])+"\n")
	num_faces=0
	for i in range(len(particle)):
		num_faces=num_faces+particle[i].num_faces
	vtk_out.write("POLYGONS\t"+str(num_faces)+"\t"+str(num_faces*4)+"\n")
	addon=0
	for i in range(len(particle)):
		for j in range(particle[i].num_faces):
			vtk_out.write(str(3)+"\t"+str(particle[i].faces[j].face_vertices_id[0]+addon)+"\t"+str(particle[i].faces[j].face_vertices_id[1]+addon)+"\t"+str(particle[i].faces[j].face_vertices_id[2]+addon)+"\n")
		addon=addon+particle[i].num_vertices
	vtk_out.close()

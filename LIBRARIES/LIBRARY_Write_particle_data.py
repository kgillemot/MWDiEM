import os

def Particle_data_writer(particle,filename,timestep):
	if timestep==0:
		#os.system("rm "+str(filename)+str(particle.particle_id)+".dat")
		f_out=open(filename+str(particle.particle_id)+".dat","w")
	else:
		f_out=open(filename+str(particle.particle_id)+".dat","a")
	if timestep==0:
		f_out.write("Timestep\t Cm_x\t Cm_y\t Cm_z\t vx\t vy\t vz\t fx\t fy\t fz\t fsx\t fsy\t fsz\t fdx\t fdy\t fdz\n")
	f_out.write(str(timestep)+"\t"+str(particle.center_of_mass[0])+"\t"+str(particle.center_of_mass[1])+"\t"+str(particle.center_of_mass[2])+"\t"+str(particle.velocity[0])+"\t"+str(particle.velocity[1])+"\t"+str(abs(particle.velocity[2]))+"\t"+str(particle.force_normal[0])+"\t"+str(particle.force_normal[1])+"\t"+str(particle.force_normal[2])+"\t"+str(particle.force_n_spring[0])+"\t"+str(particle.force_n_spring[1])+"\t"+str(particle.force_n_spring[2])+"\t"+str(particle.force_n_damping[0])+"\t"+str(particle.force_n_damping[1])+"\t"+str(particle.force_n_damping[2])+"\n")
	f_out.close()


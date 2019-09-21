import numpy as np

def Force(particle_A,particle_B,penetration_depth,penetration_normal,contact_point_A,contact_point_B,Ks,Kd,mu):
	
	######### NORMAL FORCE ##############
	# Ez valoszinuleg jo, de azert meg tesztelni kell
	
	if np.dot(particle_B.center_of_mass-particle_A.center_of_mass,penetration_normal)<0:
		penetration_normal=-penetration_normal
		
	#spring force

	Fn_As=(Ks*penetration_depth**(3/2))*penetration_normal
	if np.dot(particle_B.center_of_mass-particle_A.center_of_mass,Fn_As)>0:
		Fn_As=-Fn_As

	#damping force
	
	v_rel=particle_A.velocity-particle_B.velocity
	v_rel_n=np.dot(v_rel,penetration_normal)*penetration_normal
	Fn_Ad=-(Kd*penetration_depth**(1/2))*v_rel_n
	

	#full normal force
	
	Fn_A=Fn_As+Fn_Ad
	
	########## TANGENTIAL FORCE ################
	#print(particle_A.velocity)
	v_t_rel=v_rel-v_rel_n
	if np.linalg.norm(v_t_rel)!=0:
		Ft_A=-min(mu*np.linalg.norm(Fn_A),Ks*np.linalg.norm(v_t_rel))*v_t_rel/np.linalg.norm(v_t_rel)
	else:
		Ft_A=np.array([0,0,0])
	
	Ft_B=-Ft_A
	
	
	########### TORQUE #########################
	
	TorqueA=np.cross(contact_point_A-particle_A.center_of_mass,Ft_A)
	TorqueB=np.cross(contact_point_B-particle_B.center_of_mass,Ft_B)

	
	
	
	return(Fn_A,Fn_As,Fn_Ad,TorqueA,TorqueB)

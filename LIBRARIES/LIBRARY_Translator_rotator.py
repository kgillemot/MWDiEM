def Velocity_verlet(penetration_depth,penetration_normal,contact_point_A,contact_point_B):
	#starting time
	t=0
	#timestep
	delta_t=0.000001

	#starting position
	xt=[0 0 10]

	#starting velocity
	vt=[0 0 0]
	
	#external force
	fe=[0 0 10]

	#constant force
	


	#calculate the force vector at t+dt

	vt_dt=vt_12dt+1/2*a

	fn=K*penetration_depth^(3/2)*penetration_normal-Kd*penetration_depth*(1/2)*np.vec((v1-v2),penetration_normal)*penetration_normal
	an=fn/m
	
	vt_12dt=vt+1/2*an*dt
	xt=xt+vt_12dt*dt
	
	



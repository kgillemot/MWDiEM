inline __device__ double3 operator+(const double3& a, const double3& b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ double3 operator-(const double3& a, const double3& b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ double3 operator*(const double3& a, const double& b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}

inline __device__ double3 operator/(const double3& a, const double& b)
{
    return make_double3(a.x / b, a.y / b, a.z / b);
}

inline __device__ double dot(const double3& a, const double3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z ;
}

inline __device__ double3 cross_product(const double3& a,const double3& b)
{
    return make_double3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x );
}


inline __device__ void Quat_inv(const double* quaternion, double* quaternion_inv)
{
	quaternion_inv[0]=quaternion[0];
	quaternion_inv[1]=quaternion[1]*(-1);
	quaternion_inv[2]=quaternion[2]*(-1);
	quaternion_inv[3]=quaternion[3]*(-1);
}




inline __device__ void Quat_mult(const double* quaternion1, const double* quaternion2, double* quaternion_mult)
{
	double3 vec1=make_double3(quaternion1[1],quaternion1[2],quaternion1[3]);
	double3 vec2=make_double3(quaternion2[1],quaternion2[2],quaternion2[3]);
	double3 quatsec=vec2*quaternion1[0]+vec1*quaternion2[0]+cross_product(vec1,vec2);
	double quatfirst=quaternion1[0]*quaternion2[0]-dot(vec1,vec2);
	
	quaternion_mult[0]=quatfirst;
	quaternion_mult[1]=quatsec.x;
	quaternion_mult[2]=quatsec.y;
	quaternion_mult[3]=quatsec.z;
}


inline __device__ void Quat_tripl(const double* quaternion1, const double3 vector, const double* quaternion2, double3* vec_tripl)
{
	double vec_quat[4];
	vec_quat[0]=0;
	vec_quat[1]=vector.x;
	vec_quat[2]=vector.y;
	vec_quat[3]=vector.z;
	
	double quat_mult1[4];
	Quat_mult(vec_quat,quaternion2,quat_mult1);
	double quat_trip[4];
	Quat_mult(quaternion1,quat_mult1,quat_trip);
	double3 k=make_double3(0,0,0);
	k.x=quat_trip[1];
	k.y=quat_trip[2];
	k.z=quat_trip[3];
	*vec_tripl=k;
}

inline __device__ void Matmul1(const double (*matrix)[3], const double3 vector, double3* vec_tripl)
{
	double3 k=make_double3(0,0,0);
	k.x=matrix[0][0]*vector.x+matrix[0][1]*vector.y+matrix[0][2]*vector.z;
	k.y=matrix[1][0]*vector.x+matrix[1][1]*vector.y+matrix[1][2]*vector.z;
	k.z=matrix[2][0]*vector.x+matrix[2][1]*vector.y+matrix[2][2]*vector.z;
	*vec_tripl=k;
}

__global__ void Verlet(double* velocity_arr,
						double* velocity_12_arr, 
						double* angular_velocity_12_bf_arr,
						double* quaternion_12_arr,
						double* quaternion_arr,
						double* quaternion_all_arr,
						double* angular_velocity_bf_arr,
						double* angular_velocity_arr,
						double force_normal_arr[{{num_particles}}][3],
						double torque_arr[{{num_particles}}][3],
						double* vertices_arr,
						double* center_of_mass_arr,
						double mass_arr[{{num_particles}}],
						double moment_of_inertia_inv_arr[{{num_particles}}][3][3],
						double moment_of_inertia_arr[{{num_particles}}][3][3],
						double* dx_all_arr,
						int* num_part_P)
                                
{
	const int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
	const int P = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	if (P >= *num_part_P) 
		return;
	
	////////////////////////////////////////////////////////// Initialize #########################
	if (P>={{num_part_min}} && P<{{num_part_max}})
	{
	
		double3 velocity = make_double3(velocity_arr[P*3+0],velocity_arr[P*3+1],velocity_arr[P*3+2]);
		double3 velocity_12 = make_double3(velocity_12_arr[P*3+0],velocity_12_arr[P*3+1],velocity_12_arr[P*3+2]);
		double3 angular_velocity_12_bf = make_double3(angular_velocity_12_bf_arr[P*3+0],angular_velocity_12_bf_arr[P*3+1],angular_velocity_12_bf_arr[P*3+2]);
		double3 angular_velocity_bf = make_double3(angular_velocity_bf_arr[P*3+0],angular_velocity_bf_arr[P*3+1],angular_velocity_bf_arr[P*3+2]);
		double3 angular_velocity = make_double3(angular_velocity_arr[P*3+0],angular_velocity_arr[P*3+1],angular_velocity_arr[P*3+2]);
		double3 center_of_mass=make_double3(center_of_mass_arr[P*3+0],center_of_mass_arr[P*3+1],center_of_mass_arr[P*3+2]);

		//if (P>19 && P<29)
		//{
			//printf("veli GPU %lf %lf %lf\n",velocity_arr[P*3+0],velocity_arr[P*3+1],velocity_arr[P*3+2]);
		//}	
		
		const double3 force=make_double3(force_normal_arr[P][0],force_normal_arr[P][1],force_normal_arr[P][2]);
		const double3 torque=make_double3(torque_arr[P][0],torque_arr[P][1],torque_arr[P][2]);
		
		const double mass=mass_arr[P];

		double mom_in_inv[3][3];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				mom_in_inv[i][j]=moment_of_inertia_inv_arr[P][i][j];
			}
		}	
		
		double mom_in[3][3];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				mom_in[i][j]=moment_of_inertia_arr[P][i][j];
			}
		}	
		
		double3 vertices[{{num_vertices_max}}];
		for (int i = 0; i < {{num_vertices_max}}; i++)
			vertices[i]=make_double3(vertices_arr[P*{{num_vertices_max}}*3+i*3+0],vertices_arr[P*{{num_vertices_max}}*3+i*3+1],vertices_arr[P*{{num_vertices_max}}*3+i*3+2]);
		
		double quaternion[4];
		double quaternion_all[4];
		double quaternion_12[4];
		for (int i = 0; i < 4; i++)
		{
			quaternion[i]=quaternion_arr[P*4+i];
			quaternion_all[i]=quaternion_all_arr[P*4+i];
			quaternion_12[i]=quaternion_12_arr[P*4+i];
		}
		
		double3 dx_all = make_double3(dx_all_arr[P*3+0],dx_all_arr[P*3+1],dx_all_arr[P*3+2]);
		
		///////////////////////////////////// Calculate /////////////////////////////////
		
		double3 velocity_32=velocity_12+force*1/mass*{{timestep_size}};

		//printf("force GPU %0.10lf %0.10lf %0.10lf\n",force.x,force.y,force.z);
		//printf("torque GPU %0.10lf %0.10lf %0.10lf\n",torque.x,torque.y,torque.z);
		//printf("v32 GPU %0.10lf %0.10lf %0.10lf\n",velocity_32.x,velocity_32.y,velocity_32.z);
		double3 torque_bf=make_double3(0,0,0);
		double quaternion_inv[4];
		Quat_inv(quaternion,quaternion_inv);
		Quat_tripl(quaternion_inv, torque, quaternion, &torque_bf);
		//printf("GPU torqubf %0.10lf %0.10lf %0.10lf\n",torque_bf.x,torque_bf.y,torque_bf.z);
		
		double3 angular_acceleration_bf=make_double3(0,0,0);
		double3 angular_acceleration_bf_p1=make_double3(0,0,0);
		double3 angular_acceleration_bf_p2=make_double3(0,0,0);
		Matmul1(mom_in,angular_velocity_bf,&angular_acceleration_bf_p1);
		angular_acceleration_bf_p2=torque_bf-cross_product(angular_velocity_bf,angular_acceleration_bf_p1);
		Matmul1(mom_in_inv,angular_acceleration_bf_p2,&angular_acceleration_bf);
		//printf("GPU angacc %0.10lf %0.10lf %0.10lf\n",angular_acceleration_bf.x,angular_acceleration_bf.y,angular_acceleration_bf.z);
		
		double3 angular_acceleration=make_double3(0,0,0);
		Quat_tripl(quaternion, angular_acceleration_bf, quaternion_inv, &angular_acceleration);
		//printf("GPU angacc %0.10lf %0.10lf %0.10lf\n",angular_acceleration.x,angular_acceleration.y,angular_acceleration.z);
		
		double3 angular_velocity_32_bf=angular_velocity_12_bf+angular_acceleration_bf*{{timestep_size}};
		//printf("av32bf GPU %0.10lf %0.10lf %0.10lf\n",angular_velocity_32_bf.x,angular_velocity_32_bf.y,angular_velocity_32_bf.z);
		
		double quaternion_32[4];
		double angular_velocity_norm=sqrt(angular_velocity.x*angular_velocity.x+angular_velocity.y*angular_velocity.y+angular_velocity.z*angular_velocity.z);
		double3 temp11=make_double3(0,0,0);
		if (angular_velocity_norm!=0)
		{
			temp11=angular_velocity/angular_velocity_norm*sin(angular_velocity_norm*{{timestep_size}}/2);
		}
		double temp112=cos(angular_velocity_norm*{{timestep_size}}/2);
		double temp22[4];
		temp22[0]=temp112;
		temp22[1]=temp11.x;
		temp22[2]=temp11.y;
		temp22[3]=temp11.z;     
		Quat_mult(temp22,quaternion_12,quaternion_32);	
		//printf("q32 GPU %0.10lf %0.10lf %0.10lf %0.10lf\n",quaternion_32[0],quaternion_32[1],quaternion_32[2],quaternion_32[3]);   
		
		double3 angular_velocity_32=make_double3(0,0,0);
		double quaternion_32_inv[4];
		Quat_inv(quaternion_32,quaternion_32_inv);
		Quat_tripl(quaternion_32, angular_velocity_32_bf, quaternion_32_inv, &angular_velocity_32);
		//printf("av32 GPU %0.10lf %0.10lf %0.10lf\n",angular_velocity_32.x,angular_velocity_32.y,angular_velocity_32.z);
		
		velocity_12=velocity_32;	
		//printf("v12 GPU %0.10lf %0.10lf %0.10lf\n",velocity_12.x,velocity_12.y,velocity_12.z);
		
		quaternion_12[0]=quaternion_32[0];
		quaternion_12[1]=quaternion_32[1];
		quaternion_12[2]=quaternion_32[2];
		quaternion_12[3]=quaternion_32[3];
		//printf("q12 GPU %0.10lf %0.10lf %0.10lf %0.10lf\n",quaternion_12[0],quaternion_12[1],quaternion_12[2],quaternion_12[3]); 
		
		angular_velocity_12_bf=angular_velocity_32_bf;
		//printf("av12bf GPU %0.10lf %0.10lf %0.10lf\n",angular_velocity_12_bf.x,angular_velocity_12_bf.y,angular_velocity_12_bf.z);
			
		double3 dx=velocity_12*{{timestep_size}};
		double3 dx_old=dx_all;
		dx_all=dx_old+dx;
		//printf("dx GPU %0.10lf %0.10lf %0.10lf\n",dx.x,dx.y,dx.z);

		double3 angular_velocity_34_bf=angular_velocity_12_bf+angular_acceleration_bf*{{timestep_size}}*1/4;		
		//printf("av34bf GPU %0.10lf %0.10lf %0.10lf\n",angular_velocity_34_bf.x,angular_velocity_34_bf.y,angular_velocity_34_bf.z);
		
		double3 angular_velocity_34=make_double3(0,0,0);	
		double quaternion_12_inv[4];
		Quat_inv(quaternion_12,quaternion_12_inv);
		Quat_tripl(quaternion_12, angular_velocity_34_bf, quaternion_12_inv, &angular_velocity_34);
		//printf("av34 GPU %0.10lf %0.10lf %0.10lf\n",angular_velocity_34.x,angular_velocity_34.y,angular_velocity_34.z);
		
		double angular_velocity_34_norm=sqrt(angular_velocity_34.x*angular_velocity_34.x+angular_velocity_34.y*angular_velocity_34.y+angular_velocity_34.z*angular_velocity_34.z);
		//printf("av34norm GPU %lf \n",angular_velocity_34_norm);
		double3 temp1=make_double3(0,0,0);
		if (angular_velocity_34_norm!=0)
		{
			//printf("k");
			temp1=angular_velocity_34/angular_velocity_34_norm*sin(angular_velocity_34_norm*{{timestep_size}}/4);
		}
		//printf("temp1 GPU %0.9lf %0.9lf %0.9lf\n",temp1.x,temp1.y,temp1.z);
		double temp12=cos(angular_velocity_34_norm*{{timestep_size}}/4);
		double temp2[4];
		temp2[0]=temp12;
		temp2[1]=temp1.x;
		temp2[2]=temp1.y;
		temp2[3]=temp1.z;     
		double quaternion_old[4];
		quaternion_old[0]=quaternion[0];
		quaternion_old[1]=quaternion[1];
		quaternion_old[2]=quaternion[2];
		quaternion_old[3]=quaternion[3];
		Quat_mult(temp2,quaternion_12,quaternion);
		//printf("q GPU %0.10lf %0.10lf %0.10lf %0.10lf\n",quaternion[0],quaternion[1],quaternion[2],quaternion[3]); 	 

		Quat_mult(quaternion,quaternion_old,quaternion_all);
		//printf("q_allGPU %0.10lf %0.10lf %0.10lf %0.10lf\n",quaternion_all[0],quaternion_all[1],quaternion_all[2],quaternion_all[3]);    

		angular_velocity_bf=angular_velocity_12_bf+angular_acceleration_bf*{{timestep_size}}*1/2;
		//printf("avbf GPU %0.10lf %0.10lf %0.10lf\n",angular_velocity_bf.x,angular_velocity_bf.y,angular_velocity_bf.z);
		
		double qq_inv[4];
		Quat_inv(quaternion,qq_inv);
		Quat_tripl(quaternion, angular_velocity_bf, qq_inv, &angular_velocity);
		//printf("av GPU %0.10lf %0.10lf %0.10lf\n",angular_velocity.x,angular_velocity.y,angular_velocity.z);

		velocity=velocity_12+force*{{timestep_size}}/mass;
		//#itt biztos, hogy van egy kis numerikus bizonytalansag, mert nagyon enyhen csokken a volume of the particle, + a quaternion elso eleme sem egzaktul 0
		//printf("v GPU %0.10lf %0.10lf %0.10lf\n",velocity.x,velocity.y,velocity.z);
		
		//printf("q GPU %0.10lf %0.10lf %0.10lf %0.10lf\n",quaternion[0],quaternion[1],quaternion[2],quaternion[3]);
		double q_inv[4];
		Quat_inv(quaternion,q_inv);
		for (int i = 0; i < {{num_vertices_max}}; i++)
		{
			//if (P==2)
			//{
				//printf("GPU  %0.10lf  %0.10lf  %0.10lf\n",vertices[i].x,vertices[i].y,vertices[i].z);
			//}
			double3 temp1=vertices[i]-center_of_mass;
			//if (P==2)
			//{
				//printf("v-c GPU %0.10lf %0.10lf %0.10lf\n",temp1.x,temp1.y,temp1.z);
			//}
			double3 rot_vert;
			Quat_tripl(quaternion, temp1, q_inv, &rot_vert);
			//if (P==2)
			//{
				//printf("rto GPU %0.10lf %0.10lf %0.10lf\n",rot_vert.x,rot_vert.y,rot_vert.z);
			//}
			vertices[i]=rot_vert+center_of_mass;
			//if (P==2)
			//{
				//printf("vert GPU %0.10lf %0.10lf %0.10lf\n",vertices[i].x,vertices[i].y,vertices[i].z);
			//}
			

		}	
		
		//if (P>19 && P<29)
		//{
			//printf("%lf %lf %lf\n",dx.x,dx.y,dx.z);
		//}
		



		center_of_mass=center_of_mass+dx;
		//if (P==2)
		//{
			//printf("%0.10lf %0.10lf %0.10lf\n",center_of_mass.x,center_of_mass.y,center_of_mass.z);
		//}
		for (int i = 0; i < {{num_vertices_max}}; i++)
		{	
			vertices[i]=vertices[i]+dx;
		}	
		
		//if (P>19 && P<29)
		//{
			//printf("%lf %lf %lf\n",center_of_mass.x,center_of_mass.y,center_of_mass.z);
		//}
		//for (int i = 0; i < {{num_vertices_max}}; i++)
		//{
			//if (P==2)
			//{
				//printf("vert GPU %0.10lf %0.10lf %0.10lf\n",vertices[i].x,vertices[i].y,vertices[i].z);
			//}
		//}
		
		
		if (center_of_mass.x>{{syssize_x_max}} || center_of_mass.x<{{syssize_x_min}} || center_of_mass.y>{{syssize_y_max}} || center_of_mass.y<{{syssize_y_min}} || center_of_mass.z>{{syssize_z_max}} || center_of_mass.z<{{syssize_z_min}})
		{
			printf("flewn_away_GPU %d %f %f %f\n",P,dx.x,dx.y,dx.z);
			velocity=make_double3(0,0,0);
			velocity_12=make_double3(0,0,0);
			angular_velocity_12_bf=make_double3(0,0,0);
			angular_velocity_bf=make_double3(0,0,0);
			angular_velocity=make_double3(0,0,0);
			double3 dx_uj=dx*(-1);
			center_of_mass=center_of_mass+dx_uj;
			for (int i = 0; i < {{num_vertices_max}}; i++)
			{	
				vertices[i]=vertices[i]+dx_uj;
			}
		}
		
		//printf("%0.10lf %0.10lf %0.10lf\n",velocity.x,velocity.y,velocity.z);
		velocity_arr[P*3+0]=velocity.x;
		velocity_arr[P*3+1]=velocity.y;
		velocity_arr[P*3+2]=velocity.z;
		
		velocity_12_arr[P*3+0]=velocity_12.x;
		velocity_12_arr[P*3+1]=velocity_12.y;
		velocity_12_arr[P*3+2]=velocity_12.z;
	 
		angular_velocity_12_bf_arr[P*3+0]=angular_velocity_12_bf.x;
		angular_velocity_12_bf_arr[P*3+1]=angular_velocity_12_bf.y;
		angular_velocity_12_bf_arr[P*3+2]=angular_velocity_12_bf.z;
		
		center_of_mass_arr[P*3+0]=center_of_mass.x;
		center_of_mass_arr[P*3+1]=center_of_mass.y;
		center_of_mass_arr[P*3+2]=center_of_mass.z;
		
		dx_all_arr[P*3+0]=dx_all.x;
		dx_all_arr[P*3+1]=dx_all.y;
		dx_all_arr[P*3+2]=dx_all.z;

		angular_velocity_bf_arr[P*3+0]=angular_velocity_bf.x;
		angular_velocity_bf_arr[P*3+1]=angular_velocity_bf.y;
		angular_velocity_bf_arr[P*3+2]=angular_velocity_bf.z;
		
		angular_velocity_arr[P*3+0]=angular_velocity.x;
		angular_velocity_arr[P*3+1]=angular_velocity.y;
		angular_velocity_arr[P*3+2]=angular_velocity.z;
		
		quaternion_12_arr[P*4+0]=quaternion_12[0];
		quaternion_12_arr[P*4+1]=quaternion_12[1];
		quaternion_12_arr[P*4+2]=quaternion_12[2];
		quaternion_12_arr[P*4+3]=quaternion_12[3];
		
		quaternion_arr[P*4+0]=quaternion[0];
		quaternion_arr[P*4+1]=quaternion[1];
		quaternion_arr[P*4+2]=quaternion[2];
		quaternion_arr[P*4+3]=quaternion[3];
		
		quaternion_all_arr[P*4+0]=quaternion_all[0];
		quaternion_all_arr[P*4+1]=quaternion_all[1];
		quaternion_all_arr[P*4+2]=quaternion_all[2];
		quaternion_all_arr[P*4+3]=quaternion_all[3];
		
		for (int i = 0; i < {{num_vertices_max}}; i++)
		{
			vertices_arr[P*{{num_vertices_max}}*3+i*3+0]=vertices[i].x;
			vertices_arr[P*{{num_vertices_max}}*3+i*3+1]=vertices[i].y;
			vertices_arr[P*{{num_vertices_max}}*3+i*3+2]=vertices[i].z;
		}
		//printf("GPU %f %f %f %f\n",quaternion_all[0],quaternion_all[1],quaternion_all[2],quaternion_all[3]);    
		
		double a[4];
		a[0]=1;
		a[1]=2;
		a[2]=-3;
		a[3]=-1;
		
		double b[4];
		b[0]=-1;
		b[1]=3;
		b[2]=43;
		b[3]=-1;
		
		double a_inv[4];
		Quat_inv(a,a_inv);
		
		double a_mult[4];
		
		Quat_mult(a, b, a_mult);
		double3 res=make_double3(0,0,0);
		double3 veci=make_double3(1,4,-2);
		Quat_tripl(a, veci, a_inv, &res);
		//printf("ezi %lf %lf %lf\n",res.x,res.y,res.z);
	}	
	return;
	
}

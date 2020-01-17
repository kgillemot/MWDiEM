inline __device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator*(const float3& a, const float& b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float3 operator/(const float3& a, const float& b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z ;
}

inline __device__ float3 cross_product(const float3& a,const float3& b)
{
    return make_float3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x );
}

inline __device__ float triple_product(const float3& a,const float3& b,const float3& c)
{
    return dot(a, cross_product(b, c) );
}

inline __device__ int  clamp(const int& m,const int& v)
{
    return min(m, max(0, v) );
}

inline __device__ void array_min_finder(const float* x, int* indmin, const int num_triangles)
{
	*indmin=0;
	float xmin=x[0];
	for (int kk=0;kk<num_triangles;kk++)
		{
			if (x[kk]<xmin)
			{
				xmin=x[kk];
				*indmin=kk;
			}	
		}	
}

inline __device__ void  Support_funtion(const float3* vertices_A,const float3* vertices_B,const int num_vertices_A,const int num_vertices_B,const float3 d, float3* diff, int* a, int* b)
{
    float dist1=-1e30;
    int idx1=0;
    float dist2=-1e30;
    int idx2=0;
    
	//printf("%f %f %f \n",d.x,d.y,d.z);
	//printf("PartA\n");
    for (int i=0;i<num_vertices_A;i++)
	{
		const float ll=dot(d,vertices_A[i]);
		//printf("%4.16lf \n",ll);
		//printf("%f %f %f \n",vertices_A[i].x,vertices_A[i].y,vertices_A[i].z);
		const float temp_distance = dot(d,vertices_A[i]);
		if (temp_distance>dist1)
		{
			dist1=temp_distance;
			idx1=i;
		}
	}
	
	//printf("\n");
	//printf("PartB\n");

    for (int i=0;i<num_vertices_B;i++)
    {
		const float ll=dot(d*(-1),vertices_B[i]);
		//printf("%4.17lf \n",ll);
		//printf("%f %f %f \n",vertices_B[i].x,vertices_B[i].y,vertices_B[i].z);
		const float temp_distance = dot(d*(-1),vertices_B[i]);
		if (temp_distance>dist2)
		{
			dist2=temp_distance;
			idx2=i;
		}
	}
	//printf("\n");

	*a=idx1;
	*b=idx2;
	
	*diff=make_float3(vertices_A[*a].x-vertices_B[*b].x,vertices_A[*a].y-vertices_B[*b].y,vertices_A[*a].z-vertices_B[*b].z); 
	//float3 temppp=*diff;
	//printf("%d %d %f %f %f\n",idx1,idx2, temppp.x, temppp.y, temppp.z);
	
}

inline __device__ void Check_linesegment(const float3 P1,const float3 P2, int* coll,float3* d)
{
    *coll=-1;
    *d=make_float3(0.0,0.0,0.0);
	const float3 V21=P2-P1;
    if (dot(V21,P2)<0.0)
    {
		*coll=0;
		return; 

    }
	const float3 test=cross_product(V21,P2);
    if (test.x==0.0 && test.y==0.0 && test.z==0.0)
    {
		*coll=1;
		return; 

    }
    const float3 Vtemp=cross_product(V21,P2*(-1));
    *d=cross_product(Vtemp,V21);
    return; 
}


inline __device__ void Check_triangle(const float3 P1,const float3 P2,const float3 P3, int* coll, float3* d)
{
    *coll=-1;
	*d=make_float3(0.0,0.0,0.0);
	
	int coll2;
	Check_linesegment(P2,P3,&coll2,d);
    if (coll2==1)
	{
        *coll=1;
        return;
	}
	
	int coll3;
	Check_linesegment(P1,P3,&coll3,d);
    if (coll3==1)
    {
        *coll=1;
        return;
    }
    
    if (coll2+coll3>-1.0)
    {
        *coll=0;
        return;
    }
    
   	const float3 V21=P2-P1;
	const float3 V32=P3-P2;
	const float3 V13=P1-P3;

    const float3 n=cross_product(V32,V13);
    const float3 n2=cross_product(V32,n);
    const float3 n3=cross_product(V13,n);

    if (dot(n2,P2)<0.0)
	{
        *d=n2;
        return;
    }
    if (dot(n3,P3)<0.0)
    {
        *d=n3;
		return;
    }
    if (dot(P3,n)==0.0)
    {
        *coll=1;
        return;
    }
    if (dot(n,P3)>0.0)
    {
        *d=n*(-1.0);
    }
    else
    {
        *d=n;
    }
    return;
}

inline __device__ void Check_Tetrahedron(const float3 P1,const float3 P2,const float3 P3,const float3 P4,int* coll, float3* d, int* todel)
{
    *coll=-1;
	*d=make_float3(0.0,0.0,0.0);
    *todel=0;
    
    int coll1;
    Check_triangle(P1,P2,P4,&coll1,d);
    if (coll1==1)
    {
        *coll=1;
        return;
    }
    
    int coll2;
    Check_triangle(P2,P3,P4,&coll2,d);
    if (coll2==1)
    {
        *coll=1;
        return;
    }
    
    int coll3;
    Check_triangle(P3,P1,P4,&coll3,d);
    if (coll3==1)
    {
        *coll=1;
        return;
    }
    
    if (coll1+coll2+coll3>-2)
    {  
        *coll=0;
        return;
    }
    
	const float3 V13=P1-P3;
	const float3 V21=P2-P1;
	const float3 V32=P3-P2;
  	const float3 V41=P4-P1;
    const float3 n2=cross_product(V21,V41);
  	const float3 V42=P4-P2;
  	const float3 n3=cross_product(V32,V42);
  	const float3 V43=P4-P3;
  	const float3 n4=cross_product(V13,V43);

    if (dot(n2,P4)<0.0)
    {
        *d=n2;
        *todel=3;
    }
	else if (dot(n3,P4)<0.0)
	{
        *d=n3;
        *todel=1;
    }    
    else if (dot(n4,P4)<0.0)
    {
        *d=n4;
        *todel=2;
    }
    else
    {
        *coll=1;
    }
    return;
}

inline __device__ void CorrectFlat(const float3 S1, const float3 S2,const float3 S3,const float3 S4,int* change)
{    
    const float det1=S1.x*(S2.y*S3.z-S3.y*S2.z)-S2.x*(S1.y*S3.z-S3.y*S1.z)+S3.x*(S1.y*S2.z-S2.y*S1.z);
    *change=0;
    if (det1<0.000001 && det1>-0.000001)
    {
		const float det2=S1.x*(S2.y*S4.z-S4.y*S2.z)-S2.x*(S1.y*S4.z-S4.y*S1.z)+S4.x*(S1.y*S2.z-S2.y*S1.z);
		if (det2<0.000001 && det2>-0.000001)
		{
            *change=1;
        }
    }
}

inline __device__ void AddTriangle(const float3* V, int (*T)[3],float3* n,float* dist ,int* valid, const int* edges, const int triangle_id)
{
	int old;
	
	T[triangle_id][0]=edges[0];
	T[triangle_id][1]=edges[1];
	T[triangle_id][2]=edges[2];
	
	n[triangle_id]=cross_product(V[T[triangle_id][1]]-V[T[triangle_id][0]],V[T[triangle_id][2]]-V[T[triangle_id][1]]);
	const float len_n=sqrt(n[triangle_id].x*n[triangle_id].x+n[triangle_id].y*n[triangle_id].y+n[triangle_id].z*n[triangle_id].z);
	n[triangle_id]=n[triangle_id]/len_n;
	dist[triangle_id]=(dot(n[triangle_id],V[T[triangle_id][0]]));
	valid[triangle_id]=1;

	if (dist[triangle_id]<0.0)
	{
		old=T[triangle_id][2];
		T[triangle_id][2]=T[triangle_id][1];
		T[triangle_id][1]=old;
		dist[triangle_id]=dist[triangle_id]*(-1.0);
		n[triangle_id]=n[triangle_id]*(-1.0);
	}
	
	//check if the projection of the normal vector is inside the triangle or not
	const float3 normpoint=(n[triangle_id])*abs(dist[triangle_id]);

	const float3 Vn0=normpoint-V[T[triangle_id][0]];
	const float3 Vn1=normpoint-V[T[triangle_id][1]];
	const float3 Vn2=normpoint-V[T[triangle_id][2]];

	if (dot(cross_product(V[T[triangle_id][1]]-V[T[triangle_id][0]],n[triangle_id]),V[T[triangle_id][0]]*(-1.0))<0.0)
	{
		if (dot(cross_product(V[T[triangle_id][2]]-V[T[triangle_id][1]],n[triangle_id]),V[T[triangle_id][1]]*(-1.0))<0.0)
		{
			if (dot(cross_product(V[T[triangle_id][0]]-V[T[triangle_id][2]],n[triangle_id]),V[T[triangle_id][2]]*(-1.0))<0.0)
			{
				valid[triangle_id]=1;	
			}
			else
			{
				valid[triangle_id]=-2;
				dist[triangle_id]=1e30;
			}
		}
		else
		{
			valid[triangle_id]=-2;	
			dist[triangle_id]=1e30;
		}
	}
	else
	{
		valid[triangle_id]=-2;
		dist[triangle_id]=1e30;
	}
}


inline __device__ void  EPA(const float3 S1,const int aa1,const int bb1,const float3 S2,const int aa2,const int bb2,const float3 S3,const int aa3,const int bb3,const float3 S4,const int aa4,const int bb4,const float3* vertices_A,const float3* vertices_B,float* pen_depth, const int num_vertices_A,const int num_vertices_B,float3* cont_point_A,float3* cont_point_B, float3* pen_normal)
{
	//#include <cudart.h>
	
	// itt van valami fele numerikus instabilitas. Ha ket ugyanolyan tavolsag jon ki a Support functionvab, akkor a pythonban sokszor kicsit kulonbozo a 
	//szam es ezert ott masik oldalt valaszt. Ez nem mindig, de neha vezet kulonbozo eredmenyhez. A print parancsok, amik benne vannak a kodba, ezt szolgaljak megtlalni. 
	
	float xmin_old=1e29;
	float Suj_proj_len=2*xmin_old+1;
	int counter=0;
	int num_vertices=4;
	int num_triangles=0;		
	int T[50][3];
	float3 n[50];
	float dist[50];
	int valid[50];
	int triangle_id;
	float xmin_new;
	int indmin=0;
	int delpair[50][50];
	float3 Suj;
	int aauj;
	int bbuj;
	int edges[3];
	int onepointsA[50];
	int onepointsB[50];
	float3 V[50];
	
	// initail vertices

	V[0]=S1;
	V[1]=S2;
	V[2]=S3;
	V[3]=S4;
	
	onepointsA[0]=aa1;
	onepointsB[0]=bb1;
	onepointsA[1]=aa2;
	onepointsB[1]=bb2;
	onepointsA[2]=aa3;
	onepointsB[2]=bb3;
	onepointsA[3]=aa4;
	onepointsB[3]=bb4;

	// initial triangles

	edges[0]=0;
	edges[1]=1;
	edges[2]=2;
	triangle_id=0;
	AddTriangle(V,T,n,dist,valid,edges,triangle_id);
	num_triangles=num_triangles+1;

	edges[0]=1;
	edges[1]=2;
	edges[2]=3;
	triangle_id=1;
	AddTriangle(V,T,n,dist,valid,edges,triangle_id);
	num_triangles=num_triangles+1;
		
	edges[0]=2;
	edges[1]=0;
	edges[2]=3;
	triangle_id=2;
	AddTriangle(V,T,n,dist,valid,edges,triangle_id);
	num_triangles=num_triangles+1;
		
	edges[0]=0;
	edges[1]=1;
	edges[2]=3;
	triangle_id=3;
	AddTriangle(V,T,n,dist,valid,edges,triangle_id);
	num_triangles=num_triangles+1;
	


	while(xmin_old<Suj_proj_len) 
	{
		//printf("counter: %d\n", counter);
		
		array_min_finder(dist,&indmin,num_triangles);
		xmin_new=dist[indmin];
		xmin_old=xmin_new;
		
		Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,n[indmin],&Suj,&aauj,&bbuj);
		//printf("%f %f %f\n",Suj.x,Suj.y,Suj.z);
		Suj_proj_len=dot(n[indmin],Suj);
		
		onepointsA[num_vertices]=aauj;
		onepointsB[num_vertices]=bbuj;

		//check the final criteria for EPA
		if (xmin_old>Suj_proj_len || (xmin_old*1.00001>Suj_proj_len && xmin_old*0.99999<Suj_proj_len))
		{
			break;
		}
		if (counter>49)
		{
			printf("No min depth found!");
			break;
		}
		else
		{
			counter=counter+1;
		}
		
		//add new point
		V[num_vertices].x=Suj.x;
		V[num_vertices].y=Suj.y;
		V[num_vertices].z=Suj.z;
	
		num_vertices=num_vertices+1;
		
		//find out which triangles to delete (the ones with -1 are to be deleted)
		for (int i=0;i<num_triangles;i++)
		{
			if ((valid[i]==1 || valid[i]==-2) && (dot(V[num_vertices-1]-V[T[i][0]],n[i]))>=float(0))
			{
				valid[i]=-1;
			}
		}
	
		//collect how many times an edge pair is used
		for (int i=0;i<50*50;i++)
		{
			((int*)delpair)[i]=0;
		}
		//cudaMemset(delpair,0,50*50*sizeof(int));
		//for (int i=0;i<50;i++)
		//{
			//for (int j=0;j<50;j++)
			//{
				//delpair[i][j]=0;
			//}
		//}
		for (int i=0;i<num_triangles;i++)
		{
			if (valid[i]==-1)
			{
				delpair[T[i][1]][T[i][0]]=delpair[T[i][1]][T[i][0]]+1;
				delpair[T[i][0]][T[i][1]]=delpair[T[i][0]][T[i][1]]+1;		
				delpair[T[i][2]][T[i][1]]=delpair[T[i][2]][T[i][1]]+1;		
				delpair[T[i][1]][T[i][2]]=delpair[T[i][1]][T[i][2]]+1;	
				delpair[T[i][0]][T[i][2]]=delpair[T[i][0]][T[i][2]]+1;
				delpair[T[i][2]][T[i][0]]=delpair[T[i][2]][T[i][0]]+1;
			}
		}
		
		//if (counter==2)
		//for (int i = 0; i <num_triangles; i++) 
		//{
			//printf("Triangle (valid): %d %d\n",i,valid[i]);
			//printf("%f %d %d %d\n", dist[i],delpair[T[i][0]][T[i][1]],delpair[T[i][1]][T[i][2]],delpair[T[i][2]][T[i][0]]);	

			//printf("%f %f %f %d\n", V[T[i][0]].x,V[T[i][0]].y,V[T[i][0]].z,T[i][0]);	
			//printf("%f %f %f %d\n", V[T[i][1]].x,V[T[i][1]].y,V[T[i][1]].z,T[i][1]);	
			//printf("%f %f %f %d\n", V[T[i][2]].x,V[T[i][2]].y,V[T[i][2]].z,T[i][2]);	
			//printf("\n");

		//}
		
		//create the new triangles
		for (int i=0;i<num_triangles;i++)
		{
			if (valid[i]==-1)
			{
				if (delpair[T[i][0]][T[i][1]]<2)
				{
					edges[0]=T[i][0];
					edges[1]=T[i][1];
					edges[2]=num_vertices-1;
					triangle_id=num_triangles;
					AddTriangle(V,T,n,dist,valid,edges,triangle_id);
					num_triangles=num_triangles+1;
				}
				if (delpair[T[i][1]][T[i][2]]<2)
				{
					edges[0]=T[i][1];
					edges[1]=T[i][2];
					edges[2]=num_vertices-1;
					triangle_id=num_triangles;
					AddTriangle(V,T,n,dist,valid,edges,triangle_id);
					num_triangles=num_triangles+1;
				}	
				if (delpair[T[i][2]][T[i][0]]<2)
				{

					edges[0]=T[i][2];
					edges[1]=T[i][0];
					edges[2]=num_vertices-1;
					triangle_id=num_triangles;
					AddTriangle(V,T,n,dist,valid,edges,triangle_id);
					num_triangles=num_triangles+1;
				}
				valid[i]=0;
				dist[i]=3e20;
			}
		}
		counter=counter+1;
	}

	*pen_depth=xmin_new;
	int a=T[indmin][0];
	int b=T[indmin][1];
	int c=T[indmin][2];
	//printf("the serial number of the vertices for the triangle %d %d %d\n",a,b,c);
	//printf("normalvector: %f %f %f :\n", n[indmin].x,n[indmin].y,n[indmin].z);
	
	//#Now get the contact information, project the normalvector onto the triangle
	float3 Proj=n[indmin]*(*pen_depth);
	//printf("projection of the normalvect onto the triangle: %f %f %f\n",Proj.x,Proj.y,Proj.z);
	
	float3 A=V[T[indmin][0]];
	float3 B=V[T[indmin][1]];
	float3 C=V[T[indmin][2]];
	
	//printf("vertex coordinates of A: %f %f %f\n",A.x,A.y,A.z);
	//printf("vertex coordiantes of B: %f %f %f\n",B.x,B.y,B.z);
	//printf("vertex coordiantes of C: %f %f %f\n",C.x,C.y,C.z);
	
	float3 v0=B-A;
	float3 v1=C-A;
	float3 v2=Proj-A;
	
	if (dot(cross_product(v0,v1),n[indmin])<0)
	{
		v0=C-A;
		v1=B-A;
		//printf("change");
	}
	
	//printf("V0: %f %f %f\n",v0.x,v0.y,v0.z);
	//printf("V1: %f %f %f\n",v1.x,v1.y,v1.z);
	//printf("V2: %f %f %f\n",v2.x,v2.y,v2.z);
	
	float v00=dot(v0,v0);
	float v01=dot(v0,v1);
	float v11=dot(v1,v1);
	float v20=dot(v2,v0);
	float v21=dot(v2,v1);
	
	//printf("%f %f %f %f %f\n",v00,v01,v11,v20,v21);
	
	float denom=v00*v11-v01*v01;
	float v=(v11*v20-v01*v21)/denom;
	float w=(v00*v21-v01*v20)/denom;
	float u=1.0-w-v;

	//printf("v,w,u: %f %f %f\n",v,w,u);
	//printf("vertices of partA: %d %d %d\n",onepointsA[a],onepointsA[b],onepointsA[c]);
	//printf("vertices of partB: %d %d %d\n",onepointsB[a],onepointsB[b],onepointsB[c]);

	*cont_point_A=vertices_A[onepointsA[a]]*u+vertices_A[onepointsA[b]]*v+vertices_A[onepointsA[c]]*w;
	*cont_point_B=vertices_B[onepointsB[a]]*u+vertices_B[onepointsB[b]]*v+vertices_B[onepointsB[c]]*w;
	*pen_normal=n[indmin];

}



inline __device__ void Force(const float3 center_of_mass_A, const float3 center_of_mass_B, const float penetration_depth,float3 penetration_normal,const float3 contact_point_A,const float3 contact_point_B,const float Ks,const float Kd,const float Kt, const float mu,const float3 partA_vel,const float3 partB_vel,const float3 partA_angvel,const float3 partB_angvel, float* contacttime, float3* Fn_A,float3* Fn_As,float3* Fn_Ad,float3* TorqueA,float3* TorqueB)
{	
	//######### NORMAL FORCE ##############
	// Ez valoszinuleg jo, de azert meg tesztelni kell
	
	*contacttime=2;	
	if (dot(center_of_mass_B-center_of_mass_A,penetration_normal)<0.0)
	{
		penetration_normal=penetration_normal*(-1.0);
	}	
	
	//printf("%f %f %f\n",penetration_normal.x,penetration_normal.y,penetration_normal.z);
	//printf("%f\n",penetration_depth);
	
	float3 uij=partA_vel+cross_product(partA_angvel,contact_point_A-center_of_mass_A)-partB_vel-cross_product(partB_angvel,contact_point_B-center_of_mass_B);
	
	//spring force

	//float3 temp=penetration_normal*(Ks*pow(penetration_depth,3.0/2.0));
	//printf("%f %f %f\n",temp.x,temp.y,temp.z);
	*Fn_As=penetration_normal*(Ks*pow(penetration_depth,3.0/2.0));
	//*Fn_As=temp;
	if (dot(center_of_mass_B-center_of_mass_A,*Fn_As)>0.0)
	{
		*Fn_As=*Fn_As*(-1.0);
	}
	
	//printf("%f\n",Ks);
	//printf("kata %f %f %f\n",Fn_As[0],Fn_As[1],Fn_As[2]);
	//printf("%f\n",penetration_normal.z*(Ks*pow(penetration_depth,3.0/2.0)));
	
	//damping force
	float3 v_rel=partA_vel-partB_vel;
	float3 v_rel_n=penetration_normal*dot(v_rel,penetration_normal);
	*Fn_Ad=v_rel_n*(-1.0)*(Kd*pow(penetration_depth,1.0/2.0));
	
	//full normal force
	
	*Fn_A=*Fn_As+*Fn_Ad;
	
	
	//########## TANGENTIAL FORCE ################

	float3 v_t_rel=v_rel-v_rel_n;
	//printf("%f\n",v_t_rel.x);
	float Fn_A_norm=sqrt(dot(*Fn_A,*Fn_A));
	float v_t_rel_norm=sqrt(dot(v_t_rel,v_t_rel));
	float3 Ft_A;
	
	if (v_t_rel.x==0.0 && v_t_rel.y==0.0 && v_t_rel.z==0.0) 
	{
		Ft_A=make_float3(0.0,0.0,0.0);
	}
	else
	{
		Ft_A=(v_t_rel/v_t_rel_norm*min(mu*Fn_A_norm,Kt*v_t_rel_norm))*(-1.0);
	}

	float3 Ft_B=Ft_A*(-1.0);
	
	//########### TORQUE #########################
	
	*TorqueA=cross_product(contact_point_A-center_of_mass_A,Ft_A);  //   +*Fn_A);
	*TorqueB=cross_product(contact_point_B-center_of_mass_B,Ft_B); //-*Fn_A);

}


__global__ void contact_detection(float particle_A_center_of_mass[{{num_pairs}}][3],
                                float particle_B_center_of_mass[{{num_pairs}}][3],
                                float particle_A_vertices[{{num_pairs}}][{{num_vertices_max}}][3],
                                float particle_B_vertices[{{num_pairs}}][{{num_vertices_max}}][3],
                                float particle_A_velocity[{{num_pairs}}][3],
                                float particle_B_velocity[{{num_pairs}}][3],
                                float particle_A_angular_velocity[{{num_pairs}}][3],
                                float particle_B_angular_velocity[{{num_pairs}}][3],
                                int* particle_A_numvertices,
                                int* particle_B_numvertices,
                                int* particle_A_id,
                                int* particle_B_id,
                                float* particle_contacttime,
                                float* force,
                                float* torque,
                                int* num_pairs_P,
                                int* pair_IDs,
                                int* collision,
								float* pendepth,
                                float* tochecki)
                                
                                //#float tempi[3][3][3],
                                
{

	////////////////////////// Define pairs

	const int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
	const int P = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	if (P >= *num_pairs_P) return;
	collision[P] = -1;
	
	///////////////////////  Define single variables

	// Center of mass
	const float3 center_of_mass_A = make_float3(particle_A_center_of_mass[P][0],particle_A_center_of_mass[P][1],particle_A_center_of_mass[P][2]);
	const float3 center_of_mass_B = make_float3(particle_B_center_of_mass[P][0],particle_B_center_of_mass[P][1],particle_B_center_of_mass[P][2]);

	//// Velocity
	const float3 velocity_A = make_float3(particle_A_velocity[P][0],particle_A_velocity[P][1],particle_A_velocity[P][2]);
	const float3 velocity_B = make_float3(particle_B_velocity[P][0],particle_B_velocity[P][1],particle_B_velocity[P][2]);
	
	const float3 angular_velocity_A = make_float3(particle_A_angular_velocity[P][0],particle_A_angular_velocity[P][1],particle_A_angular_velocity[P][2]);
	const float3 angular_velocity_B = make_float3(particle_B_angular_velocity[P][0],particle_B_angular_velocity[P][1],particle_B_angular_velocity[P][2]);

	// Particle id
	const int part_id_A = particle_A_id[P];
	const int part_id_B = particle_B_id[P];
	


	// Number of real vertices
	const int num_vertices_A = particle_A_numvertices[P];
	const int num_vertices_B = particle_B_numvertices[P];

	// Vertices array
	const float3* vertices_A = (float3*)((void*)particle_A_vertices[P]);
	const float3* vertices_B = (float3*)((void*)particle_B_vertices[P]);
	//float3 vertices_A[{{num_vertices_max}}];
	//for (int i = 0; i < num_vertices_A; i++)
		//vertices_A[i]=make_float3(particle_A_vertices[P][i][0],particle_A_vertices[P][i][1],particle_A_vertices[P][i][2]);
	
	// Force
	//float3 local_force_normal = make_float3(force[P*3+0],force[P*3+1],force[P*3+2]);
	//float3 local_torque = make_float3(torque[P*3+0],torque[P*3+1],torque[P*3+2]);
	float3 local_force_normal=make_float3(0.0,0.0,0.0);
	float3 local_torque_A=make_float3(0.0,0.0,0.0);
	float3 local_torque_B=make_float3(0.0,0.0,0.0);
	
	float contime=particle_contacttime[P];
	
	// Pair IDs
	pair_IDs[2*P]=part_id_A;
	pair_IDs[2*P+1]=part_id_B;

	
	////////////////////////////////////// GJK

	int coll=-1;
	float3 S1=make_float3(0.0,0.0,0.0);
	float3 S2=make_float3(0.0,0.0,0.0);
	float3 S3=make_float3(0.0,0.0,0.0);
	float3 S4=make_float3(0.0,0.0,0.0);
	int aa1=0, bb1=0;
	int aa2=0, bb2=0;
	int aa3=0, bb3=0;
	int aa4=0, bb4=0;
	float pen_depth=0.0;
	
	//Single point
	const float3 d1=center_of_mass_B-center_of_mass_A;	
	Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d1,&S1,&aa1,&bb1);
	tochecki[P]=bb1;
	if (dot(d1,S1)<0.0)
	{
		//no collision
		coll=0;
		collision[P]=coll;
		particle_contacttime[P]=0;
		return;
	}

	// Line segment
	const float3 d2=d1*(-1);    
	Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d2,&S2,&aa2,&bb2);
	if (dot(d2,S2)<0.0)
	{
		//no collision LINE SAT
		coll=0;
		collision[P]=coll;
		particle_contacttime[P]=0;
		return;
	}

	float3 d3;
	Check_linesegment(S1,S2,&coll,&d3);
	if (coll==0)
	{
		// NO COLLISION Line Voronoi
		coll=0;
		collision[P]=coll;
		particle_contacttime[P]=0;
		return;
	}

	//Triangle
	Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d3,&S3,&aa3,&bb3);
	if (dot(d3,S3)<0.0)
	{
		//no collision Triangle SAT
		coll=0;
		collision[P]=coll;
		particle_contacttime[P]=0;
		return;
	}

	float3 d4;
	Check_triangle(S1,S2,S3,&coll,&d4);
	if (coll==0)
	{
		// NO COLLISION Triangle Voronoi
		coll=0;
		collision[P]=coll;
		particle_contacttime[P]=0;
		return;
	}

	float3 contpointA;
	float3 contpointB;
	float3 Fn_A;
	float3 Fn_As;
	float3 Fn_Ad;
	float3 TorqueA;
	float3 TorqueB;
	float3 pen_normal;
	int todel;
	int change;
	int zz=0;
	float3 S1uj;
	float3 S2uj;
	float3 S3uj;
	int aa1uj;
	int bb1uj;
	int aa2uj;
	int bb2uj;
	int aa3uj;
	int bb3uj;
	
	//Tetrahedron
	while (zz<50)
	{
		Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d4,&S4,&aa4,&bb4);
		
		if (dot(d4,S4)<0.0)
		{
			coll=0;
			collision[P]=coll;
			particle_contacttime[P]=0;
			return;
		}
		
		if (S4.x==0.0 && S4.y==0.0 && S4.z==0.0)
		{
			coll=1;	
			int change;
			CorrectFlat(S1,S2,S3,S4,&change);
			if (change==1)
			{
				d4=cross_product(S1,S2);
				Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d4,&S4,&aa4,&bb4);
			}
			collision[P]=coll;
			EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,vertices_A,vertices_B,&pen_depth,num_vertices_A,num_vertices_B,&contpointA,&contpointB,&pen_normal);
			Force(center_of_mass_A,center_of_mass_B,pen_depth,pen_normal,contpointA,contpointB,{{Ks}},{{Kd}},{{Kt}},{{mu}},velocity_A,velocity_B,angular_velocity_A,angular_velocity_B, &contime,&Fn_A, &Fn_As, &Fn_Ad, &TorqueA, &TorqueB);			
			local_force_normal=Fn_A;
			local_torque_A=TorqueA;
			local_torque_B=TorqueB;
			pendepth[P]=pen_depth;
			particle_contacttime[P]=contime;
			break;
		}
		
		Check_Tetrahedron(S1,S2,S3,S4,&coll,&d4,&todel);
		if (coll==0)
		{
			coll=0;
			collision[P]=coll;
			particle_contacttime[P]=0;
			return;
		}
		
		if (coll==1)
		{
			coll=1;
			CorrectFlat(S1,S2,S3,S4,&change);
			if (change==1)
			{
				d4=cross_product(S1,S2);
				Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d4,&S4,&aa4,&bb4);
			}
			collision[P]=coll;
			//if (P==17)
			//{
			EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,vertices_A,vertices_B,&pen_depth,num_vertices_A,num_vertices_B,&contpointA,&contpointB,&pen_normal);	
			Force(center_of_mass_A,center_of_mass_B,pen_depth,pen_normal,contpointA,contpointB,{{Ks}},{{Kd}},{{Kt}},{{mu}},velocity_A,velocity_B,angular_velocity_A,angular_velocity_B,&contime, &Fn_A, &Fn_As, &Fn_Ad, &TorqueA, &TorqueB);
			local_force_normal=Fn_A;
			local_torque_A=TorqueA;
			local_torque_B=TorqueB;
			//printf("Norm_force: %d %f %f %f\n",P, local_force_normal.x,local_force_normal.y,local_force_normal.z);
			//printf("Norm_torque: %d %f %f %f\n",P, local_torque_A.x,local_torque_A.y,local_torque_A.z);
			//printf("Cont pointt A: %d %f %f %f\n",P, contpointA.x,contpointA.y,contpointA.z);
			//printf("Cont point B: %d %f %f %f\n",P, contpointB.x,contpointB.y,contpointB.z);
			//}
			pendepth[P]=pen_depth;
			particle_contacttime[P]=contime;
			break;
		}

		if (todel==1)
		{
			S1uj=S2;
			aa1uj=aa2;
			bb1uj=bb2;
			S2uj=S3;
			aa2uj=aa3;
			bb2uj=bb3;
			S3uj=S4;
			aa3uj=aa4;
			bb3uj=bb4;
		}
		if (todel==2)
		{
			S1uj=S3;
			aa1uj=aa3;
			bb1uj=bb3;
			S2uj=S1;
			aa2uj=aa1;
			bb2uj=bb1;
			S3uj=S4;
			aa3uj=aa4;
			bb3uj=bb4;
		}
		if (todel==3)
		{
			S1uj=S1;
			aa1uj=aa1;
			bb1uj=bb1;
			S2uj=S2;
			aa2uj=aa2;
			bb2uj=bb2;
			S3uj=S4;
			aa3uj=aa4;
			bb3uj=bb4;
		}
		if (todel==4)
		{
			S1uj=S1;
			aa1uj=aa1;
			bb1uj=bb1;
			S2uj=S2;
			aa2uj=aa2;
			bb2uj=bb2;
			S3uj=S3;
			aa3uj=aa3;
			bb3uj=bb3;
		}
		S1=S1uj;
		aa1=aa1uj;
		bb1=bb1uj;
		S2=S2uj;
		aa2=aa2uj;
		bb2=bb2uj;
		S3=S3uj;
		aa3=aa3uj;
		bb3=bb3uj;
		zz=zz+1;
	}
	
	
	
	//return(force)

	/////////////////////////// Testi
	//tempi[1][2][1] = {{Ks}};
	//tempi[1][2][1] = d1.z;
	//tempi[1][2][1] = vertices_A[3].z;
	//tempi[0][0][0]=S1.x;
	//tempi[0][0][0]=num_vertices_A;
    //pendepth[P] = Tuj[2][0];
	
	// force update

	atomicAdd(force+part_id_A*3+0, local_force_normal.x);
	atomicAdd(force+part_id_A*3+1, local_force_normal.y);
	atomicAdd(force+part_id_A*3+2, local_force_normal.z);


	atomicAdd(force+part_id_B*3+0, -1*local_force_normal.x);
	atomicAdd(force+part_id_B*3+1, -1*local_force_normal.y);
	atomicAdd(force+part_id_B*3+2, -1*local_force_normal.z);
	
	atomicAdd(torque+part_id_A*3+0, local_torque_A.x);
	atomicAdd(torque+part_id_A*3+1, local_torque_A.y);
	atomicAdd(torque+part_id_A*3+2, local_torque_A.z);

	atomicAdd(torque+part_id_B*3+0, local_torque_B.x);
	atomicAdd(torque+part_id_B*3+1, local_torque_B.y);
	atomicAdd(torque+part_id_B*3+2, local_torque_B.z);
	
	return;
}

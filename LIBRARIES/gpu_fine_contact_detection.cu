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

inline __device__ double triple_product(const double3& a,const double3& b,const double3& c)
{
    return dot(a, cross_product(b, c) );
}

inline __device__ int  clamp(const int& m,const int& v)
{
    return min(m, max(0, v) );
}

inline __device__ void array_min_finder(const double* x, int* indmin, const int num_triangles)
{
	//valszeg ez is jol mukodik, a bemenettel van a baj
	*indmin=0;
	double xmin=x[0];
	//int len_x=sizeof(x);
	for (int kk=0;kk<num_triangles;kk++)
		{
			//printf("%lf\n",x[kk]);
			if (x[kk]<xmin)
			{
				xmin=x[kk];
				*indmin=kk;
			}	
		}	
	//printf("\n %d\n",*indmin);
	//printf("\n %lf\n\n",xmin);
}

inline __device__ void  Support_funtion(const double3* vertices_A,const double3* vertices_B,const int num_vertices_A,const int num_vertices_B,const double3 d, double3* diff, int* a, int* b)
{
	// ez a fuggveny jol mukodik!
    double	 dist1=-1000000;
    int idx1=0;
    //printf("%lf %lf %lf \n",d.x,d.y,d.z);
    //printf("PartA\n");
    for (int i=0;i<num_vertices_A;i++)
	{
		const double ll=dot(d,vertices_A[i]);
		//printf("%lf \n",ll);
		//printf("%lf %lf %lf \n",vertices_A[i].x,vertices_A[i].y,vertices_A[i].z);
		const double temp_distance = dot(d,vertices_A[i]);
		if (temp_distance>dist1)
		{
			dist1=temp_distance;
			idx1=i;
		}
	}
	
	//printf("\n");
	//printf("PartB\n");
	double dist2=-1000000;
    int idx2=0;
    for (int i=0;i<num_vertices_B;i++)
    {
		const double ll=dot(d*(-1),vertices_B[i]);
		//printf("%lf \n",ll);
		//printf("%lf %lf %lf \n",vertices_B[i].x,vertices_B[i].y,vertices_B[i].z);
		const double temp_distance = dot(d*(-1),vertices_B[i]);
		if (temp_distance>dist2)
		{
			dist2=temp_distance;
			idx2=i;
		}
	}
	//printf("\n");

	*a=idx1;
	*b=idx2;
	*diff=make_double3(vertices_A[*a].x-vertices_B[*b].x,vertices_A[*a].y-vertices_B[*b].y,vertices_A[*a].z-vertices_B[*b].z); 
}






inline __device__ void Check_linesegment(const double3 P1,const double3 P2, int* coll,double3* d)
{
    *coll=-1;
    *d=make_double3(0,0,0);
	const double3 V21=P2-P1;
    if (dot(V21,P2)<0)
    {
		*coll=0;
		return; 

    }
	const double3 test=cross_product(V21,P2);
    if (test.x==0 && test.y==0 && test.z==0)
    {
		*coll=1;
		return; 

    }
    const double3 Vtemp=cross_product(V21,P2*(-1));
    *d=cross_product(Vtemp,V21);
    return; 
}


inline __device__ void Check_triangle(const double3 P1,const double3 P2,const double3 P3, int* coll, double3* d)
{
    *coll=-1;
	*d=make_double3(0,0,0);
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
    if (coll2+coll3>-1)
    {
        *coll=0;
        return;
    }
   	const double3 V21=P2-P1;
	const double3 V32=P3-P2;
	const double3 V13=P1-P3;

    const double3 n=cross_product(V32,V13);
    const double3 n2=cross_product(V32,n);
    const double3 n3=cross_product(V13,n);

    if (dot(n2,P2)<0)
	{
        *d=n2;
        return;
    }
    if (dot(n3,P3)<0)
    {
        *d=n3;
		return;
    }
    if (dot(P3,n)==0)
    {
        *coll=1;
        return;
    }
    if (dot(n,P3)>0)
    {
        *d=n*(-1);
    }
    else
    {
        *d=n;
    }
    return;
}

inline __device__ void Check_Tetrahedron(const double3 P1,const double3 P2,const double3 P3,const double3 P4,int* coll, double3* d, int* todel)
{
    *coll=-1;
	*d=make_double3(0,0,0);
    *todel=0;
    int coll1;
    int coll2;
    int coll3;
    
    Check_triangle(P1,P2,P4,&coll1,d);
    if (coll1==1)
    {
        *coll=1;
        return;
    }
    Check_triangle(P2,P3,P4,&coll2,d);
    if (coll2==1)
    {
        *coll=1;
        return;
    }
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
    
	const double3 V13=P1-P3;
	const double3 V21=P2-P1;
	const double3 V32=P3-P2;
  	const double3 V41=P4-P1;
    const double3 n2=cross_product(V21,V41);
  	const double3 V42=P4-P2;
  	const double3 n3=cross_product(V32,V42);
  	const double3 V43=P4-P3;
  	const double3 n4=cross_product(V13,V43);

    if (dot(n2,P4)<0)
    {
        *d=n2;
        *todel=3;
    }
	else if (dot(n3,P4)<0)
	{
        *d=n3;
        *todel=1;
    }    
    else if (dot(n4,P4)<0)
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

inline __device__ void CorrectFlat(const double3 S1, const double3 S2,const double3 S3,const double3 S4,int* change)
{    
    const double det1=S1.x*(S2.y*S3.z-S3.y*S2.z)-S2.x*(S1.y*S3.z-S3.y*S1.z)+S3.x*(S1.y*S2.z-S2.y*S1.z);
    *change=0;
    if (det1<0.00000000001 && det1>-0.00000000001)
    {
		const double det2=S1.x*(S2.y*S4.z-S4.y*S2.z)-S2.x*(S1.y*S4.z-S4.y*S1.z)+S4.x*(S1.y*S2.z-S2.y*S1.z);
		if (det2<0.00000000001 && det2>-0.00000000001)
		{
            *change=1;
        }
    }
}

inline __device__ void AddTriangle(const double3* V, int (*T)[3],double3* n,double* dist ,int* valid, const int* edges, const int triangle_id)
{
	
	T[triangle_id][0]=edges[0];
	T[triangle_id][1]=edges[1];
	T[triangle_id][2]=edges[2];
	
	n[triangle_id]=cross_product(V[T[triangle_id][1]]-V[T[triangle_id][0]],V[T[triangle_id][2]]-V[T[triangle_id][1]]);
	const double len_n=sqrt(n[triangle_id].x*n[triangle_id].x+n[triangle_id].y*n[triangle_id].y+n[triangle_id].z*n[triangle_id].z);
	n[triangle_id]=n[triangle_id]/len_n;
	dist[triangle_id]=(dot(n[triangle_id],V[T[triangle_id][0]]));
	
	
	
	valid[triangle_id]=1;
	int old;
	if (dist[triangle_id]<0)
	{
		//printf("kata3");
		old=T[triangle_id][2];
		T[triangle_id][2]=T[triangle_id][1];
		T[triangle_id][1]=old;
		dist[triangle_id]=dist[triangle_id]*(-1);
		n[triangle_id]=n[triangle_id]*(-1);
		//printf("%lf \n",dist[*num_triangles+1]);
	}
	
	//printf("%lf \n",dist[triangle_id]);
	
	//check if the projection of the normal vector is inside the triangle or not
	const double3 normpoint=(n[triangle_id])*abs(dist[triangle_id]);
	

	
	const double3 Vn0=normpoint-V[T[triangle_id][0]];
	const double3 Vn1=normpoint-V[T[triangle_id][1]];
	const double3 Vn2=normpoint-V[T[triangle_id][2]];
	
	//printf("%d %d %d triagID\n", T[triangle_id][0],T[triangle_id][1],T[triangle_id][2]);

	//printf("%lf %lf %lf V0\n", V[T[triangle_id][0]].x,V[T[triangle_id][0]].y,V[T[triangle_id][0]].z);
	//printf("%lf %lf %lf V1\n", V[T[triangle_id][1]].x,V[T[triangle_id][1]].y,V[T[triangle_id][1]].z);
	//printf("%lf %lf %lf V2\n", V[T[triangle_id][2]].x,V[T[triangle_id][2]].y,V[T[triangle_id][2]].z);
	//printf("%lf %lf %lf normal\n", n[triangle_id].x,n[triangle_id].y,n[triangle_id].z);


	

	//if (dot(cross_product(V[T[triangle_id][0]],Vn0),cross_product(V[T[triangle_id][0]],V[T[triangle_id][2]]*(-1)))<0)
	//{
	if (dot(cross_product(V[T[triangle_id][1]]-V[T[triangle_id][0]],n[triangle_id]),V[T[triangle_id][0]]*(-1))<0)
	{
		if (dot(cross_product(V[T[triangle_id][2]]-V[T[triangle_id][1]],n[triangle_id]),V[T[triangle_id][1]]*(-1))<0)
		{
			if (dot(cross_product(V[T[triangle_id][0]]-V[T[triangle_id][2]],n[triangle_id]),V[T[triangle_id][2]]*(-1))<0)
			{
		//if (dot(cross_product(V[T[triangle_id][1]],Vn1),cross_product(V[T[triangle_id][1]],V[T[triangle_id][0]]*(-1)))<0)
		//{
			//if (dot(cross_product(V[T[triangle_id][2]],Vn2),cross_product(V[T[triangle_id][2]],V[T[triangle_id][1]]*(-1)))<0)
			//{
				valid[triangle_id]=1;	
			//}
			}
			else
			{
				////printf("kata1");
				valid[triangle_id]=-2;
				dist[triangle_id]=1e30;
			}
		}
		else
		{
			////printf("kata2");
			valid[triangle_id]=-2;	
			dist[triangle_id]=1e30;
		}
	}
	else
	{
		//printf("kata3");
		valid[triangle_id]=-2;
		dist[triangle_id]=1e30;
	}
	
	//printf("%lf\n",dist[triangle_id]);
}





//inline __device__ void  Rand_EPA(const double3 S1,const int aa1,const int bb1,const double3 S2,const int aa2,const int bb2,const double3 S3,const int aa3,const int bb3,const double3 S4,const int aa4,const int bb4,particle_A,particle_B):
//{
	//d=[[0,0,1],[0,1,0],[1,0,0],[0,0,-1],[0,-1,0],[-1,0,0],[1,1,0],[1,0,1],[0,1,1],[-1,-1,0],[-1,0,-1],[0,-1,-1],[1,1,1],[-1,-1,-1]]	
	//for i in range(0,15):
		//duj=d[i];
		//Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,duj,&Suj,&aauj,&bbuj);
//}


inline __device__ void  EPA(const double3 S1,const int aa1,const int bb1,const double3 S2,const int aa2,const int bb2,const double3 S3,const int aa3,const int bb3,const double3 S4,const int aa4,const int bb4,const double3* vertices_A,const double3* vertices_B,double* pen_depth, const int num_vertices_A,const int num_vertices_B)
{
	
	
	double xmin_old=20000000;
	double Suj_proj_len=xmin_old+1;
	
	int counter=0;

	int num_vertices=4;
	// initail vertices
	double3 V[100];
	V[0]=S1;
	V[1]=S2;
	V[2]=S3;
	V[3]=S4;

	// initial triangles
	int num_triangles=0;		
	int T[100][3];
	double3 n[100];
	double dist[100];
	int valid[100];
	//int todelete[100];
	int triangle_id;

	//for (int i = 0; i <num_vertices; i++) 
	//{
		//printf("%e %e %e V\n", V[i].x, V[i].y, V[i].z);
	//}

	
	//int (*array)[cols]
	
	int edges[3];
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
	
	
	

	//for (int i = 0; i <num_triangles; i++) 
	//{
		//printf("%d \n", valid[i]);
	//}
	
	//for (int i = 0; i <num_triangles; i++) 
	//{
		//printf("%e \n", V[i].x);
		//printf("%e \n", V[i].y);
		//printf("%e \n", V[i].z);
	//}
	
	//for (int i = 0; i <num_triangles; i++) 
	//{
        //for (int j = 0; j < 3; j++) 
        //{
            //printf("%d ", T[i][j]);
        //}
        //printf("\n");
    //}
	
	double xmin_new;
	
	//int kkkk=1;
	//while(kkkk<2)
	//{
	//	kkkk=2;
	while(xmin_old<Suj_proj_len) // && counter<4)
	{
		printf("counter: %d\n", counter);
		//printf("Vertices\n");
		//for (int i = 0; i <num_vertices; i++) 
		//{
			//printf("%lf %lf %lf %d\n", V[i].x,V[i].y,V[i].z,i);
		//}
		
		
		

		
		//find new point
		int indmin=0;
		
		double3 Suj;
		int aauj;
		int bbuj;
		
		array_min_finder(dist,&indmin,num_triangles);
		xmin_new=dist[indmin];
		xmin_old=xmin_new;
		//printf("%d\n",indmin);
		
		//printf("%lf %lf %lf\n",n[indmin].x, n[indmin].y, n[indmin].z);
		//printf("%d\n",num_vertices_B);
		//printf("%d\n",num_vertices_A);
		Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,n[indmin],&Suj,&aauj,&bbuj);
		//printf("%lf %lf %lf\n",Suj.x, Suj.y, Suj.z);
		//printf("%d %d\n",aauj,bbuj);

		Suj_proj_len=dot(n[indmin],Suj);
		//printf("%lf %lf %lf\n", Suj.x,Suj.y,Suj.z);
		
		//check the final criteria for EPA
		if (xmin_old>Suj_proj_len || (xmin_old*1.000000000001>Suj_proj_len && xmin_old*0.99999999999<Suj_proj_len))
		{
			break;
		}
		else
		{
			counter=counter+1;
		}
		
		//printf("%lf \n", Suj.z);
		
		//add new point
		V[num_vertices].x=Suj.x;
		V[num_vertices].y=Suj.y;
		V[num_vertices].z=Suj.z;
		
		num_vertices=num_vertices+1;
		
		
		
		
		//find out which triangles to delete (the ones with -1 are to be deleted)
		for (int i=0;i<num_triangles;i++)
		{
			if ((valid[i]==1 || valid[i]==-2) && (dot(V[num_vertices-1]-V[T[i][0]],n[i]))>=double(0))
			{
				valid[i]=-1;
			}
		}
		
		//collect how many times an edge pair is used
		int delpair[100][100];
		for (int i=0;i<100;i++)
		{
			for (int j=0;j<100;j++)
			{
				delpair[i][j]=0;
			}
		}
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
		
		////collect how many times an edge is used
		//int nr_vertices_del[100];
		
		//for (int i=0;i<100;i++)
		//{
			//nr_vertices_del[i]=0;
		//}
		//for (int i=0;i<num_triangles;i++)
		//{
			//if (valid[i]==-1)
			//{
				//for (int j=0;j<3;j++)
				//{
					//nr_vertices_del[T[i][j]]=nr_vertices_del[T[i][j]]+1;
				//}
			//}
		//}
		
		
		if (counter==2)
			for (int i = 0; i <num_triangles; i++) 
			{
				printf("Triangle (valid): %d %d\n",i,valid[i]);
				//printf("%lf %d %d %d\n", dist[i],nr_vertices_del[T[i][0]],nr_vertices_del[T[i][1]],nr_vertices_del[T[i][2]]);	
				printf("%lf %d %d %d\n", dist[i],delpair[T[i][0]][T[i][1]],delpair[T[i][1]][T[i][2]],delpair[T[i][2]][T[i][0]]);	

				printf("%lf %lf %lf %d\n", V[T[i][0]].x,V[T[i][0]].y,V[T[i][0]].z,T[i][0]);	
				printf("%lf %lf %lf %d\n", V[T[i][1]].x,V[T[i][1]].y,V[T[i][1]].z,T[i][1]);	
				printf("%lf %lf %lf %d\n", V[T[i][2]].x,V[T[i][2]].y,V[T[i][2]].z,T[i][2]);	
				printf("\n");

			}
		
		//printf("\n");
		//for (int i = 0; i <20; i++) 
		//{
			//printf("%d %d\n", nr_vertices_del[i],i);
		//}
		
		//create the new triangles
		for (int i=0;i<num_triangles;i++)
		{
			if (valid[i]==-1)
			{
				if (delpair[T[i][0]][T[i][1]]<2)
				//if (nr_vertices_del[T[i][0]]<2 && nr_vertices_del[T[i][1]]<2)		
				{
					edges[0]=T[i][0];
					edges[1]=T[i][1];
					edges[2]=num_vertices-1;
					triangle_id=num_triangles;
					AddTriangle(V,T,n,dist,valid,edges,triangle_id);
					num_triangles=num_triangles+1;
				}
				if (delpair[T[i][1]][T[i][2]]<2)
				//if (nr_vertices_del[T[i][1]]-int(float(nr_vertices_del[T[i][1]])/2)!=0 && nr_vertices_del[T[i][2]]-int(float(nr_vertices_del[T[i][2]])/2)!=0)
//				if (nr_vertices_del[T[i][1]]<2 && nr_vertices_del[T[i][2]]<2)
				{
					//edges[0]=T[todelete[i]][1];
					edges[0]=T[i][1];
					edges[1]=T[i][2];
					edges[2]=num_vertices-1;
					triangle_id=num_triangles;
					AddTriangle(V,T,n,dist,valid,edges,triangle_id);
					num_triangles=num_triangles+1;
				}	
				if (delpair[T[i][2]][T[i][0]]<2)
				//if (nr_vertices_del[T[i][2]]-int(float(nr_vertices_del[T[i][2]])/2)!=0 && nr_vertices_del[T[i][0]]-int(float(nr_vertices_del[T[i][0]])/2)!=0)
//				if (nr_vertices_del[T[i][2]]<2 && nr_vertices_del[T[i][0]]<2)
				{

					edges[0]=T[i][2];
					edges[1]=T[i][0];
					edges[2]=num_vertices-1;
					triangle_id=num_triangles;
					AddTriangle(V,T,n,dist,valid,edges,triangle_id);
					num_triangles=num_triangles+1;
				}
				valid[i]=0;
				dist[i]=300000000;
			}
		}
		
		

		//create the new triangles
		//for (int i=0;i<num_triangles;i++)
		//{
			//if (valid[i]==-1)
			//{
				//if (nr_vertices_del[T[i][0]]-int(float(nr_vertices_del[T[i][0]])/2)!=0 && nr_vertices_del[T[i][1]]-int(float(nr_vertices_del[T[i][1]])/2)!=0)
				////if (nr_vertices_del[T[i][0]]<2 && nr_vertices_del[T[i][1]]<2)		
				//{
					//edges[0]=T[i][0];
					//edges[1]=T[i][1];
					//edges[2]=num_vertices-1;
					//triangle_id=num_triangles;
					//AddTriangle(V,T,n,dist,valid,edges,triangle_id);
					//num_triangles=num_triangles+1;
				//}
				//if (nr_vertices_del[T[i][1]]-int(float(nr_vertices_del[T[i][1]])/2)!=0 && nr_vertices_del[T[i][2]]-int(float(nr_vertices_del[T[i][2]])/2)!=0)
////				if (nr_vertices_del[T[i][1]]<2 && nr_vertices_del[T[i][2]]<2)
				//{
					////edges[0]=T[todelete[i]][1];
					//edges[0]=T[i][1];
					//edges[1]=T[i][2];
					//edges[2]=num_vertices-1;
					//triangle_id=num_triangles;
					//AddTriangle(V,T,n,dist,valid,edges,triangle_id);
					//num_triangles=num_triangles+1;
				//}	
				//if (nr_vertices_del[T[i][2]]-int(float(nr_vertices_del[T[i][2]])/2)!=0 && nr_vertices_del[T[i][0]]-int(float(nr_vertices_del[T[i][0]])/2)!=0)
////				if (nr_vertices_del[T[i][2]]<2 && nr_vertices_del[T[i][0]]<2)
				//{

					//edges[0]=T[i][2];
					//edges[1]=T[i][0];
					//edges[2]=num_vertices-1;
					//triangle_id=num_triangles;
					//AddTriangle(V,T,n,dist,valid,edges,triangle_id);
					//num_triangles=num_triangles+1;
				//}
				//valid[i]=0;
				//dist[i]=300000000;
			//}
		//}
		
	}
	//printf("%lf",xmin_new);
	*pen_depth=xmin_new;


	//vertices_id_A[0]=aa1;
	//vertices_id_A[1]=aa2;
	//vertices_id_A[2]=aa3;
	//vertices_id_A[3]=aa4;
	//vertices_id_B[0]=bb1;
	//vertices_id_B[1]=bb2;
	//vertices_id_B[2]=bb3;
	//vertices_id_B[3]=bb4;

}



__global__ void contact_detection(double particle_A_center_of_mass[{{num_pairs}}][3],
                                double particle_B_center_of_mass[{{num_pairs}}][3],
                                double particle_A_vertices[{{num_pairs}}][{{num_vertices_max}}][3],
                                double particle_B_vertices[{{num_pairs}}][{{num_vertices_max}}][3],
                                double particle_A_velocity[{{num_pairs}}][3],
                                double particle_B_velocity[{{num_pairs}}][3],
                                int* particle_A_numvertices,
                                int* particle_B_numvertices,
                                int* particle_A_id,
                                int* particle_B_id,
                                double* force,
                                int* num_pairs_P,
                                int* pair_IDs,
                                int* collision,
								double* pendepth,
                                double* tochecki)
                                
                                //#double tempi[3][3][3],
                                
{




	////////////////////////// Define pairs

	const int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
	const int P = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	if (P >= *num_pairs_P) return;
	collision[P] = -1;
	////collision[P]=make_double3(0,0,0);

	//int Tuj[100][3];
	
	///////////////////////  Define single variables

	// Center of mass
	const double3 center_of_mass_A = make_double3(particle_A_center_of_mass[P][0],particle_A_center_of_mass[P][1],particle_A_center_of_mass[P][2]);
	const double3 center_of_mass_B = make_double3(particle_B_center_of_mass[P][0],particle_B_center_of_mass[P][1],particle_B_center_of_mass[P][2]);

	//// Velocity
	const double3 velocity_A = make_double3(particle_A_velocity[P][0],particle_A_velocity[P][1],particle_A_velocity[P][2]);
	const double3 velocity_B = make_double3(particle_B_velocity[P][0],particle_B_velocity[P][1],particle_B_velocity[P][2]);

	// Particle id
	const int part_id_A = particle_A_id[P];
	const int part_id_B = particle_B_id[P];

	// Number of real vertices
	const int num_vertices_A = particle_A_numvertices[P];
	const int num_vertices_B = particle_B_numvertices[P];

	// Vertices array

	const double3* vertices_A = (double3*)((void*)particle_A_vertices[P]);
	const double3* vertices_B = (double3*)((void*)particle_B_vertices[P]);


	//double3 vertices_A[{{num_vertices_max}}];
	//for (int i = 0; i < num_vertices_A; i++)
		//vertices_A[i]=make_double3(particle_A_vertices[P][i][0],particle_A_vertices[P][i][1],particle_A_vertices[P][i][2]);
		
	//double3 vertices_B[{{num_vertices_max}}];
	//for (int i = 0; i < num_vertices_B; i++)
		//vertices_B[i]=make_double3(particle_B_vertices[P][i][0],particle_B_vertices[P][i][1],particle_B_vertices[P][i][2]);
	
	
		
	// Force
	double3 local_force = make_double3(force[P*3+0],force[P*3+1],force[P*3+2]);

	////////////////////////////////////// GJK
	pair_IDs[2*P]=part_id_A;
	pair_IDs[2*P+1]=part_id_B;

	int coll=-1;
	double3 S1=make_double3(0,0,0);
	double3 S2=make_double3(0,0,0);
	double3 S3=make_double3(0,0,0);
	double3 S4=make_double3(0,0,0);
	int aa1=0, bb1=0;
	int aa2=0, bb2=0;
	int aa3=0, bb3=0;
	int aa4=0, bb4=0;
	double pen_depth=0;
	//Single point

	const double3 d1=center_of_mass_B-center_of_mass_A;	
	
	Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d1,&S1,&aa1,&bb1);
	tochecki[P]=bb1;

	if (dot(d1,S1)<0)
	{
		//no collision
		coll=0;
		collision[P]=coll;
		//colltype[P]=0;
		return;
	}

	// Line segment
	const double3 d2=d1*(-1);    
	Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d2,&S2,&aa2,&bb2);
	if (dot(d2,S2)<0)
	{
		//no collision LINE SAT
		coll=0;
		collision[P]=coll;
		//colltype[P]=1;
		return;
	}

	double3 d3;
	Check_linesegment(S1,S2,&coll,&d3);
	if (coll==0)
	{
		// NO COLLISION Line Voronoi
		coll=0;
		collision[P]=coll;
		//colltype[P]=2;
		return;
	}


	//Triangle
	Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d3,&S3,&aa3,&bb3);
	if (dot(d3,S3)<0)
	{
		//no collision Triangle SAT
		coll=0;
		collision[P]=coll;
		//colltype[P]=3;
		return;
	}

	double3 d4;
	Check_triangle(S1,S2,S3,&coll,&d4);
	if (coll==0)
	{
		// NO COLLISION Triangle Voronoi
		coll=0;
		collision[P]=coll;
		//colltype[P]=4;

		return;
	}

	//Tetrahedron
	int zz=0;
	while (zz<10)
	{
		Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d4,&S4,&aa4,&bb4);
		if (dot(d4,S4)<0)
		{
			coll=0;
			collision[P]=coll;
			//colltype[P]=5;

			return;
		}
		if (S4.x==0 && S4.y==0 && S4.z==0)
		{
			coll=1;	
			int change;
			CorrectFlat(S1,S2,S3,S4,&change);
			if (change==1)
			{
				d4=cross_product(S1,S2);
				Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d4,&S4,&aa4,&bb4);
			}
			//(penetration_depth,penetration_normal,contact_point_A,contact_point_B)=Fine_contact.EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,A,B)

			//(Fn_A,Fn_As,Fn_Ad,TorqueA,TorqueB)=Force(particle_A_center_of_mass[i],particle_B_center_of_mass[i],particle_A_velocity[i],particle_B_velocity[i],penetration_depth,penetration_normal,contact_point_A,contact_point_B,Ks,Kd,mu)
			//force[i][0]=Fn_A
			//force[i][1]=Fn_As
			//force[i][2]=Fn_Ad
			//force[i][3]=np.array(Torque_A)
			//force[i][4]=Torque_B
			collision[P]=coll;
			//colltype[P]=6;
			EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,vertices_A,vertices_B,&pen_depth,num_vertices_A,num_vertices_B);


			return;
		}
		int todel;
		Check_Tetrahedron(S1,S2,S3,S4,&coll,&d4,&todel);
		if (coll==0)
		{
			coll=0;
			collision[P]=coll;
			//colltype[P]=7;

			return;
		}
		if (coll==1)
		{
			coll=1;
			int change;
			CorrectFlat(S1,S2,S3,S4,&change);
			if (change==1)
			{
				d4=cross_product(S1,S2);
				Support_funtion(vertices_A,vertices_B,num_vertices_A,num_vertices_B,d4,&S4,&aa4,&bb4);
			}
			//(penetration_depth,penetration_normal,contact_point_A,contact_point_B)=Fine_contact.EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,A,B)
			//(Fn_A,Fn_As,Fn_Ad,TorqueA,TorqueB)=Force(particle_A_center_of_mass[i],particle_B_center_of_mass[i],particle_A_velocity[i],particle_B_velocity[i],penetration_depth,penetration_normal,contact_point_A,contact_point_B,Ks,Kd,mu)
			//force[i]=[Fn_A,Fn_As,Fn_Ad,Torque_A,Torque_B]
			collision[P]=coll;
			//colltype[P]=8;

			//if (P==7)
			//{
			EPA(S1,aa1,bb1,S2,aa2,bb2,S3,aa3,bb3,S4,aa4,bb4,vertices_A,vertices_B,&pen_depth,num_vertices_A,num_vertices_B);
			
			pendepth[P]=pen_depth;
			//}
			return;
		}
		double3 S1uj;
		double3 S2uj;
		double3 S3uj;
		double3 S4uj;
		int aa1uj;
		int bb1uj;
		int aa2uj;
		int bb2uj;
		int aa3uj;
		int bb3uj;
		
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

	atomicAdd(force+part_id_A*3+0, local_force.x);
	atomicAdd(force+part_id_A*3+1, local_force.y);
	atomicAdd(force+part_id_A*3+2, local_force.z);

	atomicAdd(force+part_id_B*3+0, -local_force.x);
	atomicAdd(force+part_id_B*3+1, -local_force.y);
	atomicAdd(force+part_id_B*3+2, -local_force.z);


	return;
}

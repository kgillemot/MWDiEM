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

__global__ void contact_detection(float particle_A_center_of_mass[{{num_pairs}}][3],
                                float particle_B_center_of_mass[{{num_pairs}}][3],
                                float particle_A_vertices[{{num_pairs}}][{{num_vertices_max}}][3],
                                float particle_B_vertices[{{num_pairs}}][{{num_vertices_max}}][3],
                                float particle_A_velocity[{{num_pairs}}][3],
                                float particle_B_velocity[{{num_pairs}}][3],
                                int* particle_A_numvertices,
                                int* particle_B_numvertices,
                                int* particle_A_id,
                                int* particle_B_id,
                                float* force,
                                int* num_pairs_P,
                                float tempi[3][3][3])
{




////////////////////////// Define pairs

const int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
const int P = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
if (P >= *num_pairs_P) return;


/////////////////////////  Define single variables

// Center of mass
const float3 center_of_mass_A = make_float3(particle_A_center_of_mass[P][0],particle_A_center_of_mass[P][1],particle_A_center_of_mass[P][2]);
const float3 center_of_mass_B = make_float3(particle_B_center_of_mass[P][0],particle_B_center_of_mass[P][1],particle_B_center_of_mass[P][2]);

//// Velocity
const float3 velocity_A = make_float3(particle_A_velocity[P][0],particle_A_velocity[P][1],particle_A_velocity[P][2]);
const float3 velocity_B = make_float3(particle_B_velocity[P][0],particle_B_velocity[P][1],particle_B_velocity[P][2]);

// Particle id
const int part_id_A = particle_A_id[P];
const int part_id_B = particle_B_id[P];

// Number of real vertices
const int num_vertices_A = particle_A_numvertices[P];
const int num_vertices_B = particle_B_numvertices[P];

// Force
float3 local_force = make_float3(force[P*3+0],force[P*3+1],force[P*3+2]);

// Vertices

float3* vertices_A[{{num_vertices_max}}];
float3* vertices_B[{{num_vertices_max}}];

for (int i = 0; i < num_vertices_A; i++)
	vertices_A[i]=make_float3(particle_A_vertices[P][i][0],particle_A_vertices[P][i][1],particle_A_vertices[P][i][2]);
//for (int i = 0; i < particle_B_numvertices[P]; i++)
//	vertices_B[i]=make_float3(particle_B_vertices[(P*10*3+i*3],particle_B_vertices[(P*10*3+i*3+1],particle_B_vertices[(P*10*3+i*3+2])


/////////////////////////// Testi
//tempi[1][2][1] = {{Ks}};
//tempi[1][2][1] = vertices_A[2].z;
//tempi[1][2][1]=num_vertices_B;








///////////////////////// Pairwise calculation




// ... force calc


// force update

atomicAdd(force+part_id_A*3+0, local_force.x);
atomicAdd(force+part_id_A*3+1, local_force.y);
atomicAdd(force+part_id_A*3+2, local_force.z);

atomicAdd(force+part_id_B*3+0, -local_force.x);
atomicAdd(force+part_id_B*3+1, -local_force.y);
atomicAdd(force+part_id_B*3+2, -local_force.z);


return;
}

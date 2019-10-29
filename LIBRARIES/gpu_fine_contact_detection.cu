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

__global__ void contact_detection(float* particle_A_center_of_mass,
                                float* particle_B_center_of_mass,
                                float* particle_A_vertices,
                                float* particle_B_vertices,
                                float* particle_A_velocity,
                                float* particle_B_velocity,
                                int* aa,
                                int* bb,
                                float* force,
                                int* Pmax)
{

// Ks = {{Ks}}

// Pairs
const int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
const int P = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
float3 local_force = make_float3(0,0,0);
if (P >= *Pmax) return;

const float3 center_of_mass_A = make_float3(particle_A_center_of_mass[P*3],particle_A_center_of_mass[P*3+1],particle_A_center_of_mass[P*3+2]);
const float3 center_of_mass_B = make_float3(particle_B_center_of_mass[P*3],particle_B_center_of_mass[P*3+1],particle_B_center_of_mass[P*3+2]);
//... variable definition

//... pairwise calc


// Particles
const int N1 = aa[P];
const int N2 = bb[P];

// ... force calc


// force update

atomicAdd(force+N1*3+0, local_force.x);
atomicAdd(force+N1*3+1, local_force.y);
atomicAdd(force+N1*3+2, local_force.z);

atomicAdd(force+N2*3+0, -local_force.x);
atomicAdd(force+N2*3+1, -local_force.y);
atomicAdd(force+N2*3+2, -local_force.z);


return;
}

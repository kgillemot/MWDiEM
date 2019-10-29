import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import sys
from jinja2 import Template
import numpy as np
from pathlib import Path

code = """
__global__ void test_function(float test[{{N}}][{{M}}][{{K}}])
{

const int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
const int P = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

if ({{N}}<=P) return;

for(int m=0;m<{{M}};m++)
for(int k=0;k<{{K}};k++)
  test[P][m][k] = m+k;
}

"""

template = Template(code)
N=23
M=5
K=7
args = {'N':N,'M':M,'K':K}

gpu_code = template.render(**args)
mod = SourceModule(gpu_code)
test_function = mod.get_function("test_function")

P = N
block = (64, 1, 1)
grid = (1024, int(P / 1024 / 64 + 1))

A = np.zeros(shape = (N,M,K), dtype=np.float32 )
test_function(cuda.InOut(A), block=block,grid=grid)
print("Correct:")
print(A[0,3,:])

A = np.zeros(shape = (N,M,K), dtype=np.float64 )
test_function(cuda.InOut(A), block=block,grid=grid)
print("But be careful with the types:")
print(A[0,3,:])

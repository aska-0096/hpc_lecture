#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __managed__ int bucket[5];

__global__ void initialization(int * bucket){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  bucket[i] = 0;
}
__global__ void reduction(int *bucket,int *key){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&bucket[key[i]],1);
}

__global__ void  bucketsort(int *bucket, int *key){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j=0, k=0;k<=i;j++){
  key[i]=j;
  k+=bucket[j];  
}
} 


int main() {
  int n = 50;
  int range = 5;
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  initialization<<<1,range>>>(bucket);
  cudaDeviceSynchronize();
  reduction<<<1,n>>>(bucket, key);
  cudaDeviceSynchronize();
  bucketsort<<<1,n>>>(bucket, key);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}

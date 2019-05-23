//3 different kernels and a Thrust version


#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>


#define N 2000000000 //7.45GiB

using namespace std;

void warmUpGPU();
__global__ void biggest_number(unsigned int * A, unsigned int * biggest);
__global__ void biggest_number_SM(unsigned int * A, unsigned int * biggest);
__global__ void biggest_number_fewer_threads(unsigned int * A, unsigned int * biggest);
int main(int argc, char *argv[])
{
	
	warmUpGPU();
	
	//change OpenMP settings:
	omp_set_num_threads(1);


	unsigned int * A;
	A=(unsigned int *)malloc(sizeof(unsigned int)*N);
	unsigned int * biggest;
	biggest=(unsigned int *)malloc(sizeof(unsigned int));
	*biggest=0;
	


	printf("\nSize of Array (GiB): %f",(sizeof(unsigned int)*N)/(1024.0*1024.0*1024.0));




  //seed random number generator
  srand(time(NULL));  
  //init values between 0-UINT_MAX
  for (int i=0; i<N; i++){
    A[i]=(((float)rand()/(float)(RAND_MAX))*UINT_MAX); 
  }


  	//CPU
  	unsigned int biggestCPU=0;
  	double tstartCPU=omp_get_wtime();
  	for (int i=0; i<N; i++)
  	{
  		if (A[i]>biggestCPU)
  		{
  			biggestCPU=A[i];
  		}
  	}
  	double tendCPU=omp_get_wtime();
  	printf("\nBiggest (CPU): %u",biggestCPU);
	printf("\nTime (CPU): %f",tendCPU - tstartCPU);



	double tstart=omp_get_wtime();

	//CPU version:
	/*
	for (int i=0; i<N; i++){
		C[i]=A[i]+B[i];
	}
	*/

	
	//CUDA error code:
	
	cudaError_t errCode=cudaSuccess;
	
	if(errCode != cudaSuccess)
	{
		cout << "\nLast error: " << errCode << endl; 	
	}

	unsigned int * dev_A;
	

	//allocate on the device: A
	errCode=cudaMalloc((unsigned int**)&dev_A, sizeof(unsigned int)*N);	
	if(errCode != cudaSuccess) {
	cout << "\nError: A error with code " << errCode << endl; 
	}

	//copy A to device
	errCode=cudaMemcpy( dev_A, A, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: A memcpy error with code " << errCode << endl; 
	}	
	
	unsigned int * dev_biggest;
	//allocate on the device: biggest
	errCode=cudaMalloc((unsigned int**)&dev_biggest, sizeof(unsigned int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_biggest alloc error with code " << errCode << endl; 
	}

	//copy biggest to device
	errCode=cudaMemcpy( dev_biggest, biggest, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_biggest memcpy error with code " << errCode << endl; 
	}
	

	//execute kernel
	const unsigned int totalBlocks=ceil(N*1.0/1024.0);
	printf("\ntotal blocks: %d",totalBlocks);
	

	double tstartkernelonly=omp_get_wtime();

	//select a kernel below:
	// biggest_number<<<totalBlocks,1024>>>(dev_A, dev_biggest);
	// biggest_number_SM<<<totalBlocks,1024>>>(dev_A, dev_biggest);
	biggest_number_fewer_threads<<<(totalBlocks/10),1024>>>(dev_A, dev_biggest);
	cudaDeviceSynchronize();
	double tendkernelonly=omp_get_wtime();
	printf("\nTime kernel only: %f",tendkernelonly - tstartkernelonly);

	if(errCode != cudaSuccess){
		cout<<"Error afrer kernel launch "<<errCode<<endl;
	}

	//copy data from device to host 
	errCode=cudaMemcpy( biggest, dev_biggest, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: getting C result form GPU error with code " << errCode << endl; 
	}
	else
	{
		printf("\nBiggest number: %u",*biggest);
	}

	// for (int i=N-10; i<N; i++)
	// {
	// 	printf("\n%d",C[i]);
	// }
	
	
	double tend=omp_get_wtime();
	
	printf("\nTotal time (s): %f",tend-tstart);


	cudaFree(dev_A);
	cudaFree(dev_biggest);


	///////////////////////////
	//Thrust version
	////////////////////////////

	double tstartthrust=omp_get_wtime();

	thrust::device_vector<unsigned int> d_vec(A, A+N);
	thrust::device_vector<unsigned int>::iterator iter =
	thrust::max_element(d_vec.begin(), d_vec.end());

	unsigned int max_val = *iter;
	printf("\nBiggest val (Thrust): %u",max_val);
	double tendthrust=omp_get_wtime();
	printf("\nTime Thrust (s): %f",tendthrust - tstartthrust);



	printf("\n");

	return 0;
}


//biggest number -- "inefficient reduction"
//surprising: actually takes very little time when subtracting no-op kernel time
__global__ void biggest_number(unsigned int * A, unsigned int * biggest) {

unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 
if (tid>=N){
	return;
}
atomicMax(biggest, A[tid]);
return;
}

//biggest number 
//with blocks computing the maximum number -- slower than global memory
__global__ void biggest_number_SM(unsigned int * A, unsigned int * biggest) {

unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 

if (tid>=N){
	return;
}


__shared__ unsigned int biggestInBlock;
if (threadIdx.x==0)
biggestInBlock=0;
__syncthreads();

atomicMax(&biggestInBlock, A[tid]);
__syncthreads();
if (threadIdx.x==0)
atomicMax(biggest, biggestInBlock);


return;
}

//each thread does 10 comparisons
//reduces atomic operations
__global__ void biggest_number_fewer_threads(unsigned int * A, unsigned int * biggest) {

unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 

if (tid>=(N/10)){
	return;
}

unsigned int myBiggest=0;
#pragma unroll
for (int i=tid*10; i<tid*10+10; i++)
{
	myBiggest=max(myBiggest,A[i]);
}


atomicMax(biggest, myBiggest);


return;
}


__global__ void warmup(unsigned int * tmp) {
if (threadIdx.x==0)
*tmp=555;

return;
}



void warmUpGPU(){
printf("\nWarming up GPU for time trialing...\n");	
unsigned int * dev_tmp;
unsigned int * tmp;
tmp=(unsigned int*)malloc(sizeof(unsigned int));
*tmp=0;
cudaError_t errCode=cudaSuccess;
errCode=cudaMalloc((unsigned int**)&dev_tmp, sizeof(unsigned int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_tmp error with code " << errCode << endl; 
	}

warmup<<<1,256>>>(dev_tmp);

//copy data from device to host 
	errCode=cudaMemcpy( tmp, dev_tmp, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: getting tmp result form GPU error with code " << errCode << endl; 
	}

	printf("\ntmp (changed to 555 on GPU): %d",*tmp);

cudaFree(dev_tmp);

return;
}
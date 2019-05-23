//calculating the area of the circle
//goals:
//compare performance as a function of input size
//show differences in the number within the circle results
//remove aromic adds to wsee performance difference 
//measure time to copy data to gpu
//figure out the time spent doing distance calculations
//fix the atmic add overhead with partial sums

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>

#include <math.h>

#define N 1400000000 //Max: 1400000000

using namespace std;

struct point{
    float x;
    float y;
  };  

void warmUpGPU();
__global__ void areaCircleWPsum(struct point * PointList, unsigned long long int * countPsumNumInCircle);


int main(int argc, char *argv[])
{
	
	warmUpGPU();
	
	//change OpenMP settings:
	omp_set_num_threads(1);

	int i;

  	

  	//seed random number generator
  srand(time(NULL));  


  struct point * PointList;
  PointList=(struct point *)malloc(N * sizeof(struct point));
  //init point values between [0-1] for x and y. 
  //positive/positive quadrant
  for (i=0; i<N; i++){
    PointList[i].x=(((float)rand()/(float)(RAND_MAX))*1.0); 
    PointList[i].y=(((float)rand()/(float)(RAND_MAX))*1.0); 
  }


  printf("\nMemory requested for points to test (GiB): %f", (N*1.0 * sizeof(struct point)/(1024.0*1024.0*1024.0)));

/////////////////////////////
//CPU
////////////////////////////	


	unsigned long long int countInsideCPU=0;
	double tstartCPU=omp_get_wtime();
	// #pragma omp parallel for reduction(+:countInsideCPU)
	  for (int i=0; i<N; i++){
	    float distance=sqrt((PointList[i].x*PointList[i].x)+(PointList[i].y*PointList[i].y));
	    if (distance<=1.0)
	      countInsideCPU++;
	  }
	double tendCPU=omp_get_wtime();

	printf("\nTotal time CPU %f",tendCPU - tstartCPU);
	printf("\nTotal number in circle (CPU): %llu",countInsideCPU);
	printf("\nEstimate of Pi (CPU): %f",(countInsideCPU*1.0/N*1.0)*4.0);



/////////////////////////////
//GPU
////////////////////////////	

	double tstart=omp_get_wtime();


	
	
	
	cudaError_t errCode=cudaSuccess;
	
	if(errCode != cudaSuccess)
	{
		cout << "\nLast error: " << errCode << endl; 	
	}

	struct point * dev_PointList;
	unsigned long long int * countNumInCircle;
	unsigned long long int * dev_countNumInCircle; 
	countNumInCircle=(unsigned long long int *)malloc(sizeof(unsigned long long int)); 
	dev_countNumInCircle=(unsigned long long int *)malloc(sizeof(unsigned long long int));
	*countNumInCircle=0;

	//allocate on the device: PointList
	errCode=cudaMalloc((struct point**)&dev_PointList, sizeof(struct point)*N);	
	if(errCode != cudaSuccess) {
	cout << "\nError: point list error with code " << errCode << endl; 
	}

	//allocate the number in the circle on device
	errCode=cudaMalloc((unsigned long long int**)&dev_countNumInCircle, sizeof(unsigned long long int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: count in circle error with code " << errCode << endl; 
	}

	//copy points to device
	// double tstartdatacpy=omp_get_wtime();

	errCode=cudaMemcpy( dev_PointList, PointList, sizeof(struct point)*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_PointList memcpy error with code " << errCode << endl; 
	}	
	// double tenddatacpy=omp_get_wtime();
	// printf("\nTime to copy data: %f",tenddatacpy - tstartdatacpy);
	
	//copy counts to device
	errCode=cudaMemcpy( dev_countNumInCircle, countNumInCircle, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: count in circle memcpy error with code " << errCode << endl; 
	}

	//calculate blocks
	const unsigned int totalBlocks=ceil(N*1.0/1024.0);
	printf("\ntotal blocks (GPU): %d",totalBlocks);


	
	//partial sum -- array of partial sums
	unsigned long long int * countPSumInCircle;
	countPSumInCircle=(unsigned long long int *)calloc(totalBlocks, sizeof(unsigned long long int)); 
	unsigned long long int * dev_countPSumInCircle;
	dev_countPSumInCircle=(unsigned long long int *)calloc(totalBlocks, sizeof(unsigned long long int)); 

	errCode=cudaMalloc((unsigned long long int**)&dev_countPSumInCircle, sizeof(unsigned long long int)*totalBlocks);	
	if(errCode != cudaSuccess) {
	cout << "\nError: count in circle error with code " << errCode << endl; 
	}

	//copy counts to device
	errCode=cudaMemcpy( dev_countPSumInCircle, countPSumInCircle, sizeof(unsigned long long int)*totalBlocks, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: count in circle memcpy error with code " << errCode << endl; 
	}
	


	// execute partial sum:
	areaCircleWPsum<<<totalBlocks,1024>>>(dev_PointList, dev_countPSumInCircle);	

	if(errCode != cudaSuccess){
		cout<<"Error afrer kernel launch "<<errCode<<endl;
	}

	
	//partial sums
	//copy data from device to host -- partial sums
	errCode=cudaMemcpy( countPSumInCircle, dev_countPSumInCircle, sizeof(unsigned long long int)*totalBlocks, cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: getting C result form GPU error with code " << errCode << endl; 
	}

	//partial sums
	for (int i=0; i<totalBlocks; i++)
	{
		*countNumInCircle+=countPSumInCircle[i];
	}

	printf("\nTotal number in circle (GPU): %llu",*countNumInCircle);
	printf("\nEstimate of Pi (GPU): %f",(*countNumInCircle*1.0/N*1.0)*4.0);
	
	double tend=omp_get_wtime();
	
	printf("\nTotal time (s): %f",tend-tstart);



	printf("\n");

	return 0;
}

__global__ void areaCircle(struct point * PointList, unsigned long long int * countNumInCircle) {

unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 
	
if (tid>N){
	return;
}


if (sqrt((PointList[tid].x*PointList[tid].x)+(PointList[tid].y*PointList[tid].y))<=1.0)
{
	atomicAdd(countNumInCircle, int(1));
}

return;
}

//partial sums on the atomic add to remove bottleneck
__global__ void areaCircleWPsum(struct point * PointList, unsigned long long int * countPsumNumInCircle) {

unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 
	
if (tid>=N){
	return;
}


if (sqrt((PointList[tid].x*PointList[tid].x)+(PointList[tid].y*PointList[tid].y))<=1.0)
{
	atomicAdd(countPsumNumInCircle+blockIdx.x, int(1));
}

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


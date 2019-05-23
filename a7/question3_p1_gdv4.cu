#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "omp.h"

//See values of N in assignment instructions.
#define N 100000
//Do not change the seed, or your answer will not be correct
#define SEED 72

//For GPU implementation
#define BLOCKSIZE 1024


struct pointData{
double x;
double y;
};


void warmUpGPU();
void generateDataset(struct pointData * data);


int main(int argc, char *argv[])
{
   warmUpGPU();
   cudaError_t errCode=cudaSuccess;
	//Read epsilon distance from command line
	if (argc!=2)
	{
	printf("\nIncorrect number of input parameters. Please input an epsilon distance.\n");
	return 0;
	}


	char inputEpsilon[20];
	strcpy(inputEpsilon,argv[1]);
	double epsilon=atof(inputEpsilon);



	//generate dataset:
	struct pointData * data;
	data=(struct pointData*)malloc(sizeof(struct pointData)*N);
	printf("\nSize of dataset (MiB): %f",(2.0*sizeof(double)*N*1.0)/(1024.0*1024.0));
	generateDataset(data);


	omp_set_num_threads(1);





	//Write your code here:
	//The data you need to use is stored in the variable "data",
	//which is of type pointData
   struct pointData[2*N] temp_data;
   for(int i = 0; i < N; i++) {
      for(int j = 0; j < N; j++) {
         temp[i+j] = [data[i],data[j]];
      }
   }
   struct pointData * data_ptr = &temp_data;
   struct pointData * dev_data;
   int total = 0;
   unsigned int * dev_e;
   unsigned int * dev_total;

   double mem_to_start=omp_get_wtime();
   errCode=cudaMalloc((struct pointData**)&dev_data, sizeof(struct pointData)*2*N)
   if(errCode != cudaSuccess) {
      cout << "\nError: A error with code " << errCode << endl;
   }

   errCode=cudaMalloc((unsigned int**)&dev_e, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: A error with code " << errCode << endl;
	}

   errCode=cudaMalloc((unsigned int**)&dev_total, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: A error with code " << errCode << endl;
	}

   errCode=cudaMemcpy( dev_data, data_ptr, sizeof(struct pointData)*2*N, cudaMemcpyHostToDevice);
   if(errCode != cudaSuccess) {
	cout << "\nError: A memcpy error with code " << errCode << endl;
	}

   errCode=cudaMemcpy( dev_e, epsilon, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: A memcpy error with code " << errCode << endl;
	}

   errCode=cudaMemcpy( dev_total, total, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: A memcpy error with code " << errCode << endl;
	}
   double mem_to_end=omp_get_wtime();
   const unsigned int totalBlocks=ceil(N*2.0/1024.0);
   calc_epsilon<<<totalBlocks,1024>>>(dev_data, dev_e, dev_total);

   double mem_from_start=omp_get_wtime();
   errCode=cudaMemcpy( total, dev_total, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: getting C result form GPU error with code " << errCode << endl;
	}

	double mem_from_end=omp_get_wtime();

	printf("\nTotal time mem-copy: %f\n",mem_to_end-mem_to_start+mem_from_end-mem_from_start);
   printf("\nTotal time in kernal: %f\n",mem_from_start-mem_to_end);
   printf("total in epsilon: %f\n", total);
	free(data);
	printf("\n");
	return 0;
}


//Do not modify the dataset generator or you will get the wrong answer
void generateDataset(struct pointData * data)
{

	//seed RNG
	srand(SEED);


	for (unsigned int i=0; i<N; i++){
		data[i].x=1000.0*((double)(rand()) / RAND_MAX);
		data[i].y=1000.0*((double)(rand()) / RAND_MAX);
	}


}

__global__ void warmup(unsigned int * tmp) {
if (threadIdx.x==0)
*tmp=555;

return;
}

__global__ void calc_epsilon(struct pointData *data, unsigned int *epsilon, unsigned int *total) {

unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x);

if (tid>=N*2){
	return;
}
double x1 = data[tid][0].x;
double x2 = data[tid][1].x;
double y1 = data[tid][0].y;
double y2 = data[tid][1].y;

double distance = sqrt(poq(x1-x2,2) + pow(y1-y2,2));
if(distance <= *epsilon) {
   atomicAdd(total, 1);
}

return;
}

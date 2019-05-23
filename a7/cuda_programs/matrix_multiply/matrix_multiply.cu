//naive matrix multiply
//Global memory only
//use 1-D A, B, C on both GPU and CPU


#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>

#include <math.h>

#define N 100 //max 2000 (for time)

using namespace std;



void warmUpGPU();
void compareMatrices(double * C_GPU, double * C_CPU);
__global__ void matrixmulti(double *A, double *B, double *C, unsigned int * debug);

int main(int argc, char *argv[])
{
	
	warmUpGPU();
	
	//change OpenMP settings:
	omp_set_num_threads(1);

	int i,j;

  	

  	//seed random number generator
  srand(time(NULL));  


  double * A;
  double * B;
  double * C;
  double * C_CPU;

  A=(double *)malloc(sizeof(double)*N*N);
  B=(double *)malloc(sizeof(double)*N*N);
  C=(double *)calloc(N*N,sizeof(double));
  C_CPU=(double *)calloc(N*N,sizeof(double));

  //init matrices
  for (i=0; i<N*N; i++){
  	A[i]=i;
  	B[i]=i;
  }


  printf("\nMemory requested for 3x NxN matrices (GiB) %f", (3.0*N*N*sizeof(double)/(1024.0*1024.0*1024.0)));



///////////////////////////
//CPU version:
///////////////////////////

double tstartcpu=omp_get_wtime();

int ROW=0;
int COL=0;

for (ROW=0; ROW<N; ROW++)
	for (COL=0; COL<N; COL++)
		for (int k=0; k<N; k++)
		{
			C_CPU[(ROW*N)+COL]+=A[ROW*N+k]*B[COL+(k*N)];
		}

double tendcpu=omp_get_wtime();
printf("\nTime CPU: %f",tendcpu - tstartcpu);


//print matrix if N is less than 10x10

	int cnt=0;
	if (N<=10)
	{
		printf("\n CPU matrix is: \n");
		for (i=0; i<N; i++){
			for (j=0; j<N; j++){
				printf("%.2f, ",C_CPU[cnt]);
				cnt++;
			}
			printf("\n");
		}
	}


/////////////////////////////
//GPU
////////////////////////////	

	double tstart=omp_get_wtime();

	
	cudaError_t errCode=cudaSuccess;
	
	if(errCode != cudaSuccess)
	{
		cout << "\nLast error: " << errCode << endl; 	
	}

	double * dev_A;
	double * dev_B;
	double * dev_C;
	unsigned int * dev_debug;
	unsigned int * debug;
	debug=(unsigned int *)malloc(sizeof(unsigned int));
	*debug=0;

	//allocate on the device: A, B, C
	errCode=cudaMalloc((double**)&dev_A, sizeof(double)*N*N);	
	if(errCode != cudaSuccess) {
	cout << "\nError: A error with code " << errCode << endl; 
	}

	errCode=cudaMalloc((double**)&dev_B, sizeof(double)*N*N);	
	if(errCode != cudaSuccess) {
	cout << "\nError: B error with code " << errCode << endl; 
	}

	errCode=cudaMalloc((double**)&dev_C, sizeof(double)*N*N);	
	if(errCode != cudaSuccess) {
	cout << "\nError: C error with code " << errCode << endl; 
	}

	//debug
	errCode=cudaMalloc((unsigned int**)&dev_debug, sizeof(unsigned int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: C error with code " << errCode << endl; 
	}

	//copy A to device
	errCode=cudaMemcpy( dev_A, A, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: A memcpy error with code " << errCode << endl; 
	}	
	
	//copy B to device
	errCode=cudaMemcpy( dev_B, B, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: B memcpy error with code " << errCode << endl; 
	}

	//copy C to device (initialized to 0)
	errCode=cudaMemcpy( dev_C, C, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: C memcpy error with code " << errCode << endl; 
	}

	//copy debug to device
	errCode=cudaMemcpy( dev_debug, debug, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: debug memcpy error with code " << errCode << endl; 
	}

	//setup blocks
	const unsigned int totalBlocks=ceil(N*N*1.0/1024.0);
	printf("\ntotal blocks: %d",totalBlocks);



	//execute kernel
	matrixmulti<<<totalBlocks,1024>>>(dev_A, dev_B, dev_C, dev_debug);


	if(errCode != cudaSuccess){
		cout<<"Error afrer kernel launch "<<errCode<<endl;
	}

	// copy C from the GPU
	errCode=cudaMemcpy( C, dev_C, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: getting result form GPU error with code " << errCode << endl; 
	}

	// copy debug from the GPU
	errCode=cudaMemcpy( debug, dev_debug, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: getting result form GPU error with code " << errCode << endl; 
	}
	else{
		printf("\ndebug val: %d",*debug);
	}


	
	
	double tend=omp_get_wtime();

	//print matrix if N is less than 10
	// int cnt=0;
	cnt=0;
	if (N<=10)
	{
		printf("\nGPU Matrix: \n");
		for (i=0; i<N; i++){
			for (j=0; j<N; j++)
			{
				
				printf("%.2f, ",C[cnt]);
				cnt++;
			}
			printf("\n");
		}
	}
	
	printf("\nTotal time GPU (s): %f",tend-tstart);


	compareMatrices(C, C_CPU);

	printf("\n");

	return 0;
}

void compareMatrices(double * C_GPU, double * C_CPU)
{
	double delta=0;
	for (int i=0; i<N*N; i++)
	{
		delta+=fabs(C_CPU[i]-C_GPU[i]);
	}

	printf("\nDelta between matrices: %f",delta);
}

//matrix multiply
//each thread computes a single element of C using a row of A and column of B
__global__ void matrixmulti(double *A, double *B, double *C, unsigned int * debug) {

unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 

int ROW = tid/N; //row
int COL = tid%N; //col

if ((ROW < N) && (COL < N)){
double tmp_sum = 0;
	for (unsigned int k = 0; k < N; k++) {
	double a = A[ROW * N + k];
	double b = B[k * N + COL];
	tmp_sum += a * b;
	}
	C[ROW * N + COL] = tmp_sum;
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


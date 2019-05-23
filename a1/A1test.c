#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

//to compile
//gcc -fopenmp A1test.c -o A1test
//-lomp, -lopenmp (one of these)

int main(){

	omp_set_num_threads(10);

	#pragma omp parallel
	{
		int tid=omp_get_thread_num();
		printf("Hello world from thread %d\n",tid);
	}

	return 0;
}
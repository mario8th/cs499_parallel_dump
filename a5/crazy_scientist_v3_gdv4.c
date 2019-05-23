#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define SIZE 50

double do_crazy_computation(int i,int j);

int main(int argc, char **argv) {
  double mat[SIZE][SIZE];
  int i,j;
  double time[2];

  omp_set_num_threads(2);
  double tstart=omp_get_wtime();
  #pragma omp parallel private(i, j) shared(mat, time)
  {
     #pragma omp for schedule(dynamic) nowait
     for (i=0;i<SIZE;i++)
     { /* loop over the rows */
       for (j=0;j<SIZE;j++)
       {  /* loop over the columns */
         mat[i][j] = do_crazy_computation(i,j);
         fprintf(stderr,".");
       }
     }
     time[omp_get_thread_num()] = omp_get_wtime();

  }
  double tend=omp_get_wtime();
  double elapsed=tend - tstart;
  printf("\nTotal Time (sanity check): %f\n",elapsed);
  printf("Time Thread1: %f\n",time[0] - tstart);
  printf("Time Thread2: %f\n",time[1] - tstart);
  double load_imbalance = fabs(time[0] - time[1]);
  printf("Load imbalance: %f\n", load_imbalance);

  exit(0);
}

//Crazy computation
double do_crazy_computation(int x,int y) {
   int iter;
   double value=0.0;

   for (iter = 0; iter < 5*x*x*x+1 + y*y*y+1; iter++) {
     value +=  (cos(x*value) + sin(y*value));
   }
  return value;
}

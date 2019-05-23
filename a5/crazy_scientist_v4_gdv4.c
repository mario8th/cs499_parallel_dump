#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>

#define SIZE 50

double do_crazy_computation(int i,int j);
void *do_work(void *);

double mat[SIZE][SIZE];
double times[2];
int iterations;
int thread_its[2];
pthread_mutex_t mutex;
pthread_attr_t attr;

int main(int argc, char **argv) {

   pthread_mutex_init(&mutex, NULL);
   pthread_attr_init(&attr);
   pthread_t threads[2];
   iterations = 0;

   double tstart=omp_get_wtime();

   for (int i = 0; i < 2; i++)
   {
      if (pthread_create(&(threads[i]), NULL, do_work, (void *)(long)i))
      {
         fprintf(stderr,"Error while creating thread #%d\n",i);
         exit(1);
      }
   }
   for (int i = 0; i<2; i++)
   {
      pthread_join(threads[i], NULL);
   }
   double tend=omp_get_wtime();
   double elapsed=tend - tstart;
   printf("\nTotal Time (sanity check): %f\n",elapsed);
   printf("Number of iteratsion for thread1: %d\n", thread_its[0]);
   printf("Number of iteratsion for thread2: %d\n", thread_its[1]);
   printf("Time Thread1: %f\n",times[0] - tstart);
   printf("Time Thread2: %f\n",times[1] - tstart);
   double load_imbalance = fabs(times[0] - times[1]);
   printf("Load imbalance: %f\n", load_imbalance);

  exit(0);
}

void* do_work(void *args)
{
   int thread_id = (long) args;
   int i;
   int j;
   while(1)
   {
      pthread_mutex_lock(&mutex);
      if (iterations == SIZE)
      {
         pthread_mutex_unlock(&mutex);
         break;
      }
      i = iterations;
      iterations++;
      pthread_mutex_unlock(&mutex);
      thread_its[thread_id]++;
      for (j=0;j<SIZE;j++)
      {  /* loop over the columns */
        mat[i][j] = do_crazy_computation(i,j);
        fprintf(stderr,".");
      }
   }
   times[thread_id] = omp_get_wtime();
   return NULL;
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

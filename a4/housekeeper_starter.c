#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

void random_sleep(int a, int b);

#define NUM_ITERATIONS 10
#define NUM_WASHERS 3
#define NUM_DRYERS 3

/* Helper function to sleep a random number of microseconds
 * picked between two bounds (provided in seconds)
 */
void random_sleep(int lbound_sec, int ubound_sec) {
   int num_usec;
   num_usec = lbound_sec*100000 + 
              (int)((ubound_sec - lbound_sec)*100000 * ((double)(rand()) / RAND_MAX)); 
   usleep(num_usec);
   return; 
}



//main function
int main(int argc, char **argv) {
 
  int seed;
  int num_washer_staff;
  int num_dryer_staff;
  int num_washer_dryer_staff;

  //Process command-line arguments
  if (argc != 5) {
    fprintf(stderr,"Usage: %s <# washers only> <# dryers only> <# both washers and dryers> <seed>\n",argv[0]);
    exit(1);
  }

  if ((sscanf(argv[1],"%d",&num_washer_staff) != 1) || 
      (sscanf(argv[2],"%d",&num_dryer_staff) != 1) || 
      (sscanf(argv[3],"%d",&num_washer_dryer_staff) != 1) || 
      (sscanf(argv[4],"%d",&seed) != 1) ||
      (num_washer_staff < 1) ||
      (num_dryer_staff < 1) ||
      (num_washer_dryer_staff < 1) ||
      (seed < 0)) {
    fprintf(stderr,"Invalid command-line arguments... Aborting\n");
    exit(1);
  }
 
  /* Seed the random number generator */
  srand(seed);

  //IMPLEMENT CODE HERE

  exit(0);
}



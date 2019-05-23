#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

void random_sleep(int a, int b);
void *washer_work(void*);
void *dryer_work(void*);
void *washer_dryer_work(void*);

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

typedef struct mutexLocks {
   pthread_mutex_t washer_check_m;  // to check if washer available
   pthread_mutex_t washer_mod_m;    // to modify  washer

   pthread_mutex_t dryer_check_m;   // to check if dryer available
   pthread_mutex_t dryer_mod_m;     // to modify dryer
}mutexLocks;

typedef struct threadHelper {
   mutexLocks *mutex_locks;
   int thread_id;
   int seed;
   int *washers;
   int *dryers;
}threadHelper;

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

  int washers = 3;
  int dryers = 3;

  /*pthread_mutex_t washer_check_m;
  pthread_mutex_t washer_mod_m;
  pthread_mutex_t dryer_check_m;
  pthread_mutex_t dryer_mod_m;*/
  pthread_attr_t attr;

  mutexLocks *mutex_locks = malloc(sizeof(mutexLocks));
  pthread_mutex_init(&mutex_locks->washer_check_m, NULL);
  pthread_mutex_init(&mutex_locks->washer_mod_m, NULL);
  pthread_mutex_init(&mutex_locks->dryer_check_m, NULL);
  pthread_mutex_init(&mutex_locks->dryer_mod_m, NULL);
  pthread_attr_init(&attr);
  /*mutex_locks->washer_check_m = washer_check_m;
  mutex_locks->washer_mod_m = washer_mod_m;
  mutex_locks->dryer_check_m = dryer_check_m;
  mutex_locks->dryer_mod_m = dryer_mod_m;*/

  threadHelper thread_helper[num_washer_staff + num_dryer_staff + num_washer_dryer_staff];
  pthread_t wash_thread[num_washer_staff];
  pthread_t dry_thread[num_dryer_staff];
  pthread_t wash_dry_thread[num_washer_dryer_staff];
  for (int i = 0; i < num_washer_staff; i++) {
     thread_helper[i].mutex_locks = mutex_locks;
     thread_helper[i].thread_id = i;
     thread_helper[i].seed = seed;
     thread_helper[i].washers = &washers;
     thread_helper[i].dryers = &dryers;
     if (pthread_create(&(wash_thread[i]), NULL, washer_work, &thread_helper[i]))
     {
        fprintf(stderr,"Error while creating washer_work thread #%d\n",i);
        exit(1);
     }
  }
  for (int j = 0; j < num_dryer_staff; j++) {
     thread_helper[j+num_washer_staff].mutex_locks = mutex_locks;
     thread_helper[j+num_washer_staff].thread_id = j;
     thread_helper[j+num_washer_staff].seed = seed;
     thread_helper[j+num_washer_staff].washers = &washers;
     thread_helper[j+num_washer_staff].dryers = &dryers;
     if (pthread_create(&(dry_thread[j]), NULL, dryer_work, &thread_helper[j+num_dryer_staff]))
     {
        fprintf(stderr,"Error while creating dryer_work thread #%d\n",j);
        exit(1);
     }
  }
  for (int k = 0; k < num_washer_dryer_staff; k++) {
     thread_helper[k+num_washer_staff+num_dryer_staff].mutex_locks = mutex_locks;
     thread_helper[k+num_washer_staff+num_dryer_staff].thread_id = k;
     thread_helper[k+num_washer_staff+num_dryer_staff].seed = seed;
     thread_helper[k+num_washer_staff+num_dryer_staff].washers = &washers;
     thread_helper[k+num_washer_staff+num_dryer_staff].dryers = &dryers;
     if (pthread_create(&(wash_dry_thread[k]), NULL, washer_dryer_work, &thread_helper[k+num_dryer_staff+num_dryer_staff]))
     {
        fprintf(stderr,"Error while creating washer_dryer_work thread #%d\n",k);
        exit(1);
     }
  }
  for (int l = 0; l<num_washer_staff; l++)
  {
     pthread_join(wash_thread[l], NULL);
  }
  for (int m = 0; m<num_dryer_staff; m++)
  {
     pthread_join(dry_thread[m+num_washer_staff], NULL);
  }
  for (int n = 0; n<num_washer_dryer_staff; n++)
  {
     pthread_join(wash_dry_thread[n+num_dryer_staff+num_washer_staff], NULL);
  }
  printf("I get here (here being the end of the main function)!\n");
  exit(0);
}

void *washer_work(void *arg) {
   threadHelper *thread_helper = (threadHelper *)arg;
   mutexLocks *mutex_locks = thread_helper->mutex_locks;
   for (int i = 0; i < 10; i++) {
      printf("[Washer housekeeper %d] is working...\n",thread_helper->thread_id);
      usleep(200000);
      //printf("do I get here?\n");
      pthread_mutex_lock(&mutex_locks->washer_check_m);
      //printf("do I get here1?\n");
      while(*(thread_helper->washers) == 0) {
         //printf("do I get here?2\n");
         pthread_mutex_unlock(&mutex_locks->washer_check_m);
         pthread_mutex_lock(&mutex_locks->washer_check_m);
      }
      pthread_mutex_lock(&mutex_locks->washer_mod_m);
      *(thread_helper->washers)--;
      printf("[Washer housekeeper %d] has got a washer...\n",thread_helper->thread_id);
      pthread_mutex_unlock(&mutex_locks->washer_mod_m);
      pthread_mutex_unlock(&mutex_locks->washer_check_m);
      printf("[Washer housekeeper %d] has put laundry in the washer...\n",thread_helper->thread_id);
      usleep(100000);
      printf("[Washer housekeeper %d] has taken articles out of the washer...\n",thread_helper->thread_id);
      pthread_mutex_lock(&mutex_locks->washer_mod_m);
      *(thread_helper->washers)++;
      printf("[Washer housekeeper %d] has finished with the washer...\n",thread_helper->thread_id);
      pthread_mutex_unlock(&mutex_locks->washer_mod_m);
   }
   return NULL;
}
void *dryer_work(void *arg) {
   threadHelper *thread_helper = (threadHelper *)arg;
   mutexLocks *mutex_locks = thread_helper->mutex_locks;
   pthread_mutex_lock(&mutex_locks->dryer_check_m);
   printf("hello from thread %d!\n",thread_helper->thread_id);
   usleep(25000);
   printf("hello?\n");
   pthread_mutex_unlock(&mutex_locks->dryer_check_m);
   return NULL;
}
void *washer_dryer_work(void *arg) {
   threadHelper *thread_helper = (threadHelper *)arg;
   mutexLocks *mutex_locks = thread_helper->mutex_locks;
   return NULL;
}

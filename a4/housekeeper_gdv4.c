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

typedef struct mutexStruct {
   pthread_mutex_t washer_check_m;  // to check if washer available
   pthread_mutex_t washer_mod_m;    // to modify  washer

   pthread_mutex_t dryer_check_m;   // to check if dryer available
   pthread_mutex_t dryer_mod_m;     // to modify dryer
}mutexStruct;

typedef struct threadHelper {
   mutexStruct *locks;
   int thread_id;
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

  pthread_attr_t attr;
  int washers = NUM_WASHERS;
  int dryers = NUM_DRYERS;

  mutexStruct *mutex_locks = malloc(sizeof(mutexStruct));
  pthread_mutex_init(&mutex_locks->washer_check_m, NULL);
  pthread_mutex_init(&mutex_locks->washer_mod_m, NULL);
  pthread_mutex_init(&mutex_locks->dryer_check_m, NULL);
  pthread_mutex_init(&mutex_locks->dryer_mod_m, NULL);
  pthread_attr_init(&attr);

  pthread_t wash_thread[num_washer_staff];
  pthread_t dry_thread[num_dryer_staff];
  pthread_t wash_dry_thread[num_washer_dryer_staff];

  threadHelper wash_helper[num_washer_staff];
  threadHelper dry_helper[num_dryer_staff];
  threadHelper wash_dry_helper[num_washer_dryer_staff];

   for (int i = 0; i < num_washer_staff; i++) {
     wash_helper[i].locks = mutex_locks;
     wash_helper[i].thread_id = i;
     wash_helper[i].washers = &washers;
     wash_helper[i].dryers = &dryers;
     pthread_create(&wash_thread[i], NULL, washer_work, &wash_helper[i]);
  }

   for (int i = 0; i < num_dryer_staff; i++) {
      dry_helper[i].locks = mutex_locks;
      dry_helper[i].thread_id = i;
      dry_helper[i].washers = &washers;
      dry_helper[i].dryers = &dryers;
      pthread_create(&dry_thread[i], NULL, dryer_work, &dry_helper[i]);
   }

   for (int i = 0; i < num_washer_dryer_staff; i++) {
      wash_dry_helper[i].locks = mutex_locks;
      wash_dry_helper[i].thread_id = i;
      wash_dry_helper[i].washers = &washers;
      wash_dry_helper[i].dryers = &dryers;
      pthread_create(&wash_dry_thread[i], NULL, washer_dryer_work, &wash_dry_helper[i]);
   }

   for (int i = 0; i<num_washer_staff; i++) {
      pthread_join(wash_thread[i], NULL);
   }

   for (int i = 0; i<num_dryer_staff; i++) {
      pthread_join(dry_thread[i], NULL);
   }

   for (int i = 0; i<num_washer_dryer_staff; i++) {
      pthread_join(wash_dry_thread[i], NULL);
   }
  exit(0);
}

void *washer_work(void *arg) {
   threadHelper *helper = arg;
   mutexStruct *locks = helper->locks;
   for (int i = 0; i < NUM_ITERATIONS; i++) {
      printf("[Washer housekeeper %d] is working...\n",helper->thread_id);
      random_sleep(2, 5);
      printf("[Washer housekeeper %d] wants a washer...\n",helper->thread_id);
      pthread_mutex_lock(&locks->washer_check_m);
      while((*(helper->washers) == 0)) {
         pthread_mutex_unlock(&locks->washer_check_m);
         pthread_mutex_lock(&locks->washer_check_m);
      }
      pthread_mutex_lock(&locks->washer_mod_m);
      *(helper->washers)-=1;
      printf("[Washer housekeeper %d] has got a washer...\n",helper->thread_id);
      pthread_mutex_unlock(&locks->washer_mod_m);
      pthread_mutex_unlock(&locks->washer_check_m);
      printf("[Washer housekeeper %d] has put laundry in the washer...\n",helper->thread_id);
      random_sleep(2, 5);
      printf("[Washer housekeeper %d] has taken articles out of the washer...\n",helper->thread_id);
      pthread_mutex_lock(&locks->washer_mod_m);
      *(helper->washers)+=1;
      printf("[Washer housekeeper %d] has finished with the washer...\n",helper->thread_id);
      pthread_mutex_unlock(&locks->washer_mod_m);

   }
   return NULL;
}
void *dryer_work(void *arg) {
   threadHelper *helper = arg;
   mutexStruct *locks = helper->locks;
   for (int i = 0; i < NUM_ITERATIONS; i++) {
      printf("[Dryer housekeeper %d] is working...\n",helper->thread_id);
      random_sleep(2, 5);
      printf("[Dryer housekeeper %d] wants a dryer...\n",helper->thread_id);
      pthread_mutex_lock(&locks->dryer_check_m);
      while((*(helper->dryers) == 0)) {
         pthread_mutex_unlock(&locks->dryer_check_m);
         pthread_mutex_lock(&locks->dryer_check_m);
      }
      pthread_mutex_lock(&locks->dryer_mod_m);
      *(helper->dryers)-=1;
      printf("[Dryer housekeeper %d] has got a dryer...\n",helper->thread_id);
      pthread_mutex_unlock(&locks->dryer_mod_m);
      pthread_mutex_unlock(&locks->dryer_check_m);
      printf("[Dryer housekeeper %d] has put laundry in the dryer...\n",helper->thread_id);
      random_sleep(2, 5);
      printf("[Dryer housekeeper %d] has taken articles out of the dryer...\n",helper->thread_id);
      pthread_mutex_lock(&locks->dryer_mod_m);
      *(helper->dryers)+=1;
      printf("[Dryer housekeeper %d] has finished with the dryer...\n",helper->thread_id);
      pthread_mutex_unlock(&locks->dryer_mod_m);

   }
   return NULL;
}
void *washer_dryer_work(void *arg) {
   threadHelper *helper = arg;
   mutexStruct *locks = helper->locks;
   for (int i = 0; i < NUM_ITERATIONS; i++) {
      printf("[Washer/dryer housekeeper %d] is working...\n",helper->thread_id);
      random_sleep(2, 5);
      printf("[Washer/dryer housekeeper %d] wants a washer...\n",helper->thread_id);
      pthread_mutex_lock(&locks->washer_check_m);
      while((*(helper->washers) == 0)) {
         pthread_mutex_unlock(&locks->washer_check_m);
         pthread_mutex_lock(&locks->washer_check_m);
      }
      pthread_mutex_lock(&locks->washer_mod_m);
      *(helper->washers)-=1;
      printf("[Washer/dryer housekeeper %d] has got a washer...\n",helper->thread_id);
      pthread_mutex_unlock(&locks->washer_mod_m);
      pthread_mutex_unlock(&locks->washer_check_m);

      printf("[Washer/dryer housekeeper %d] wants a dryer...\n",helper->thread_id);
      pthread_mutex_lock(&locks->dryer_check_m);
      while((*(helper->dryers) == 0)) {
         pthread_mutex_unlock(&locks->dryer_check_m);
         pthread_mutex_lock(&locks->dryer_check_m);
      }
      pthread_mutex_lock(&locks->dryer_mod_m);
      *(helper->dryers)-=1;
      printf("[Washer/dryer housekeeper %d] has got a dryer...\n",helper->thread_id);
      pthread_mutex_unlock(&locks->dryer_mod_m);
      pthread_mutex_unlock(&locks->dryer_check_m);

      printf("[Washer/dryer housekeeper %d] has started using both the washer and dryer...\n",helper->thread_id);
      random_sleep(2, 5);
      printf("[Washer/dryer housekeeper %d]has taken articles out of the washer and dryer...\n",helper->thread_id);
      pthread_mutex_lock(&locks->washer_mod_m);
      *(helper->washers)+=1;
      printf("[Washer/dryer housekeeper %d] has finished with the washer...\n",helper->thread_id);
      pthread_mutex_unlock(&locks->washer_mod_m);
      pthread_mutex_lock(&locks->dryer_mod_m);
      *(helper->dryers)+=1;
      printf("[Washer/dryer housekeeper %d] has finished with the dryer...\n",helper->thread_id);
      pthread_mutex_unlock(&locks->dryer_mod_m);
   }
}

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

void *increment_work(void *);
void *decrement_work(void *);

pthread_mutex_t mutex;
pthread_cond_t ready;
pthread_attr_t attr;
int counter;
int total_increments;
int total_decrements;

int main(int argc, char* argv[])
{
   pthread_mutex_init(&mutex, NULL);
   pthread_attr_init(&attr);
   pthread_t thread[2];

   pthread_cond_init(&ready, NULL);

   counter = 0;
   total_decrements = 0;
   total_increments = 0;
   if (pthread_create(&(thread[0]), NULL, increment_work, NULL))
   {
      fprintf(stderr,"Error while creating thread increment_work\n");
      exit(1);
   }
   if (pthread_create(&(thread[1]), NULL, decrement_work, NULL))
   {
      fprintf(stderr,"Error while creating thread decrement_work\n");
      exit(1);
   }

   for (int i = 0; i<2; i++)
   {
      pthread_join(thread[i], NULL);
   }
   return 1;
}

void *increment_work(void *arg)
{
   pthread_mutex_lock(&mutex);
   while(1)
   {
      while( counter == 10 )
      {
         pthread_cond_signal(&ready);
         pthread_cond_wait(&ready, &mutex);
      }
      counter++;
      total_increments++;
      printf("Count is now (inc fn): %d\n", counter);
      usleep(500000);
      if (total_increments == 30)
      {
         break;
      }
   }
   pthread_mutex_unlock(&mutex);
   return NULL;
}

void *decrement_work(void *arg)
{
   pthread_mutex_lock(&mutex);
   while(1)
   {
      while( counter == 0 )
      {
         pthread_cond_signal(&ready);
         pthread_cond_wait(&ready, &mutex);
      }
      counter--;
      total_decrements++;
      printf("Count is now (dec fn): %d\n", counter);
      usleep(500000);
      if (total_decrements == 20)
      {
         pthread_cond_signal(&ready);
         break;
      }
   }
   pthread_mutex_unlock(&mutex);
   return NULL;
}

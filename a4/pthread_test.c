#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

void *do_work(void *);

typedef struct mutexStruct {
   pthread_mutex_t mutex;
}mutexStruct;

int main(int argc, char* argv[]) {
   pthread_attr_t attr;

   mutexStruct *mutex_locks = malloc(sizeof(mutexStruct));
   pthread_mutex_init(&mutex_locks->mutex, NULL);
   pthread_attr_init(&attr);

   pthread_t threads[10];
   for (int i = 0; i < 10; i++) {
      printf("%d\n",i);
      pthread_create(&(threads[i]), NULL, do_work, mutex_locks);
   }
   for (int i = 0; i<10; i++)
   {
      pthread_join(threads[i], NULL);
   }
   return 1;

}

void *do_work(void *arg)
{
   mutexStruct *locks = (mutexStruct *) arg;
   pthread_mutex_lock(&locks->mutex);
   printf("hello!\n");
   usleep(500000);
   pthread_mutex_unlock(&locks->mutex);
}

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

void *do_work(void *);

pthread_mutex_t mutex;
pthread_attr_t attr;
int total;
int previousNum;

int main(int argc, char* argv[])
{
   pthread_mutex_init(&mutex, NULL);
   pthread_attr_init(&attr);
   pthread_t thread[10];
   total = 0;
   previousNum = 9;
   if (pthread_create(&(thread[0]), NULL, do_work, (void *)(long)0))
   {
      fprintf(stderr,"Error while creating thread #%d\n",0);
      exit(1);
   }
   if (pthread_create(&(thread[1]), NULL, do_work, (void *)(long)1))
   {
      fprintf(stderr,"Error while creating thread #%d\n",1);
      exit(1);
   }
   if (pthread_create(&(thread[2]), NULL, do_work, (void *)(long)2))
   {
      fprintf(stderr,"Error while creating thread #%d\n",2);
      exit(1);
   }
   if (pthread_create(&(thread[3]), NULL, do_work, (void *)(long)3))
   {
      fprintf(stderr,"Error while creating thread #%d\n",3);
      exit(1);
   }
   if (pthread_create(&(thread[4]), NULL, do_work, (void *)(long)4))
   {
      fprintf(stderr,"Error while creating thread #%d\n",4);
      exit(1);
   }
   if (pthread_create(&(thread[5]), NULL, do_work, (void *)(long)5))
   {
      fprintf(stderr,"Error while creating thread #%d\n",5);
      exit(1);
   }
   if (pthread_create(&(thread[6]), NULL, do_work, (void *)(long)6))
   {
      fprintf(stderr,"Error while creating thread #%d\n",6);
      exit(1);
   }
   if (pthread_create(&(thread[7]), NULL, do_work, (void *)(long)7))
   {
      fprintf(stderr,"Error while creating thread #%d\n",7);
      exit(1);
   }
   if (pthread_create(&(thread[8]), NULL, do_work, (void *)(long)8))
   {
      fprintf(stderr,"Error while creating thread #%d\n",8);
      exit(1);
   }
   if (pthread_create(&(thread[9]), NULL, do_work, (void *)(long)9))
   {
      fprintf(stderr,"Error while creating thread #%d\n",9);
      exit(1);
   }

   for (int i = 0; i<10; i++)
   {
      pthread_join(thread[i], NULL);
   }

   printf("Total: %d\n", total);
   return 1;
}

void *do_work(void *arg)
{
   int myNum = (long) arg;
   while(1)
   {
      //wait at mutex
      pthread_mutex_lock(&mutex);
      if(total == 990)
      {
         pthread_mutex_unlock(&mutex);
         return NULL;
      }
      if(previousNum == myNum - 1 || previousNum == 9 && myNum == 0)
      {
         total += myNum;
         previousNum = myNum;
         printf("my num: %d, total: %d\n",myNum, total);
      }
      pthread_mutex_unlock(&mutex);
   }
   return NULL;
}

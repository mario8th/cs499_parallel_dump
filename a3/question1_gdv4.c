#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

void *do_work(void *);

typedef struct threadHelper
{
   int myID;
   int *index;
   int *buffer;
   int *sequenceTotal;
   int *sequenceCorrect;
}threadHelper;

pthread_mutex_t mutex1;
pthread_mutex_t mutex2;
pthread_attr_t attr;

int main(int argc, char* argv[])
{
   int buffer[3];
   int index = 0;
   int sequenceTotal = 0;
   int sequenceCorrect = 0;

   threadHelper *helper1 = malloc(sizeof(threadHelper));
   threadHelper *helper2 = malloc(sizeof(threadHelper));
   threadHelper *helper3 = malloc(sizeof(threadHelper));
   helper1->myID = 1;
   helper2->myID = 2;
   helper3->myID = 3;
   helper1->index = &index;
   helper2->index = &index;
   helper3->index = &index;
   helper1->buffer = buffer;
   helper2->buffer = buffer;
   helper3->buffer = buffer;
   helper1->sequenceTotal = &sequenceTotal;
   helper2->sequenceTotal = &sequenceTotal;
   helper3->sequenceTotal = &sequenceTotal;
   helper1->sequenceCorrect = &sequenceCorrect;
   helper2->sequenceCorrect = &sequenceCorrect;
   helper3->sequenceCorrect = &sequenceCorrect;

   pthread_mutex_init(&mutex1, NULL);
   pthread_mutex_init(&mutex2, NULL);
   pthread_attr_init(&attr);
   pthread_t thread[3];
   if (pthread_create(&(thread[0]), NULL, do_work, helper1))
   {
      fprintf(stderr,"Error while creating thread #%d\n",1);
      exit(1);
   }
   if (pthread_create(&(thread[1]), NULL, do_work, helper2))
   {
      fprintf(stderr,"Error while creating thread #%d\n",2);
      exit(1);
   }
   if (pthread_create(&(thread[2]), NULL, do_work, helper3))
   {
      fprintf(stderr,"Error while creating thread #%d\n",3);
      exit(1);
   }

   for (int i; i<3; i++)
   {
      pthread_join(thread[i], NULL);
   }
   printf("Total sequences generated: %d\n", sequenceTotal);
   printf("Number of correct sequences: %d\n", sequenceCorrect);
   return 1;
}

void *do_work(void *arg)
{
   threadHelper *helper = (threadHelper *)arg;
   while(1)
   {
      //wait at mutex1
      pthread_mutex_lock(&mutex1);
      //add to buffer
      printf("My id: %d\n", helper->myID);
      helper->buffer[*(helper->index)] = helper->myID;
      //incriment index
      *(helper->index)+= 1;
      //check if full
      if (*(helper->index) == 3)
      {
         printf("%d%d%d\n", helper->buffer[0],helper->buffer[1],helper->buffer[2]);
         if (helper->buffer[0] == 1 && helper->buffer[1] == 2 && helper->buffer[2] == 3 )
         {
            *(helper->sequenceCorrect) += 1;
         }
         *(helper->index) = 0;
         *(helper->sequenceTotal) += 1;
         helper->buffer[0] = 0;
         helper->buffer[1] = 0;
         helper->buffer[2] = 0;

      }
      //unlock mutex1
      pthread_mutex_unlock(&mutex1);
      pthread_mutex_lock(&mutex2);
      //check if 10 wins
      if (*(helper->sequenceCorrect) == 10)
      {
         pthread_mutex_unlock(&mutex2);
         return NULL;
      }
      pthread_mutex_unlock(&mutex2);
      usleep(500000);

   }
   return NULL;
}

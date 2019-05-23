#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

void *do_work(void *);
void *do_work2(void *); //electric boogaloo

typedef struct threadHelper
{
   int myID;
   int *index;
   int *buffer;
   int *sequenceTotal;
   int *sequenceCorrect;
   int *winState;
}threadHelper;

pthread_mutex_t mutex2;
pthread_mutex_t mutex3;
pthread_attr_t attr;

int main(int argc, char* argv[])
{
   int buffer[3];
   int index = 0;
   int sequenceTotal = 0;
   int sequenceCorrect = 0;
   int buffer2[3];
   int index2 = 0;
   int sequenceTotal2 = 0;
   int sequenceCorrect2 = 0;
   int winState = 0;

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
   helper1->winState = &winState;
   helper2->winState = &winState;
   helper3->winState = &winState;

   threadHelper *helper4 = malloc(sizeof(threadHelper));
   threadHelper *helper5 = malloc(sizeof(threadHelper));
   threadHelper *helper6 = malloc(sizeof(threadHelper));
   helper4->myID = 4;
   helper5->myID = 5;
   helper6->myID = 6;
   helper4->index = &index2;
   helper5->index = &index2;
   helper6->index = &index2;
   helper4->buffer = buffer2;
   helper5->buffer = buffer2;
   helper6->buffer = buffer2;
   helper4->sequenceTotal = &sequenceTotal2;
   helper5->sequenceTotal = &sequenceTotal2;
   helper6->sequenceTotal = &sequenceTotal2;
   helper4->sequenceCorrect = &sequenceCorrect2;
   helper5->sequenceCorrect = &sequenceCorrect2;
   helper6->sequenceCorrect = &sequenceCorrect2;
   helper4->winState = &winState;
   helper5->winState = &winState;
   helper6->winState = &winState;

   pthread_mutex_init(&mutex1, NULL);
   pthread_mutex_init(&mutex2, NULL);
   pthread_mutex_init(&mutex3, NULL);
   pthread_attr_init(&attr);
   pthread_t thread[6];
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
   if (pthread_create(&(thread[3]), NULL, do_work2, helper4))
   {
      fprintf(stderr,"Error while creating thread #%d\n",4);
      exit(1);
   }
   if (pthread_create(&(thread[4]), NULL, do_work2, helper5))
   {
      fprintf(stderr,"Error while creating thread #%d\n",5);
      exit(1);
   }
   if (pthread_create(&(thread[5]), NULL, do_work2, helper6))
   {
      fprintf(stderr,"Error while creating thread #%d\n",6);
      exit(1);
   }

   for (int i; i<6; i++)
   {
      pthread_join(thread[i], NULL);
   }
   printf("Total sequences generated team1: %d\n", sequenceTotal);
   printf("Number of correct sequences team1: %d\n", sequenceCorrect);
   printf("Total sequences generated team2: %d\n", sequenceTotal2);
   printf("Number of correct sequences team2: %d\n", sequenceCorrect2);
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
         if (*(helper->sequenceCorrect) == 10)
         {
            *(helper->winState) = 1;
            printf("Team 1 won!\n");
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
      //check winState
      //    if so, unlock mutex2 return
      //unlock mutex2
      if (*(helper->winState) == 1)
      {
         pthread_mutex_unlock(&mutex2);
         return NULL;
      }
      pthread_mutex_unlock(&mutex2);
      usleep(500000);

   }
   return NULL;
}

void *do_work2(void *arg)
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
         if (helper->buffer[0] == 4 && helper->buffer[1] == 5 && helper->buffer[2] == 6 )
         {
            *(helper->sequenceCorrect) += 1;
         }
         if (*(helper->sequenceCorrect) == 10)
         {
            *(helper->winState) = 1;
            printf("Team 2 won!\n");
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
      //check if winState
      //    if so, unlock mutex3 return
      //unlock mutex3
      if (*(helper->winState) == 1)
      {
         pthread_mutex_unlock(&mutex2);
         return NULL;
      }
      pthread_mutex_unlock(&mutex2);
      usleep(500000);

   }
   return NULL;
}

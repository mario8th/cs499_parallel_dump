//compilation instructions/examples:
//gcc -fopenmp point_epsilon_starter.c -o point_epsilon_starter
//sometimes you need to link against the math library with -lm:
//gcc -fopenmp point_epsilon_starter.c -lm -o point_epsilon_starter

//math library needed for the square root

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "omp.h"

//N is 100000 for the submission. However, you may use a smaller value of testing/debugging.
//#define N 100000
//#define N 1000
#define N 100000
//Do not change the seed, or your answer will not be correct
#define SEED 72


struct pointData{
double x;
double y;
};



void generateDataset(struct pointData * data);


void printPointArray(struct pointData * data);
void SortPointArray_ByX(struct pointData * data);
void quickSortPointArray_ByX(struct pointData arr[], int low, int high);
void swap_pointData(struct pointData * a, struct pointData * b);

int main(int argc, char *argv[])
{


	//Read epsilon distance from command line
	if (argc!=2)
	{
	printf("\nIncorrect number of input parameters. Please input an epsilon distance.\n");
	return 0;
	}


	char inputEpsilon[20];
	strcpy(inputEpsilon,argv[1]);
	double epsilon=atof(inputEpsilon);


	printf("\nN = %i, epsilon = %f, \n ", N, epsilon);

	//generate dataset:
	struct pointData * data;
	data=(struct pointData*)malloc(sizeof(struct pointData)*N);
	printf("\nSize of dataset (MiB): %f",(2.0*sizeof(double)*N*1.0)/(1024.0*1024.0));
	generateDataset(data);

   int i, j,ToRight,ToLeft;
	 int total_count = 0;
   double x_1, x_2, y_1, y_2, distance;


	//change OpenMP settings:
	omp_set_num_threads(2);

	double tstart = omp_get_wtime();
	//Write your code here:
	//The data you need to use is stored in the variable "data",
	//which is of type pointData

	//Sort
	//printPointArray(data);
	SortPointArray_ByX(data);
	//printPointArray(data);








	#pragma omp parallel private(i,j,x_1,x_2,y_1,y_2,distance,ToRight,ToLeft) shared(data,epsilon,total_count)
	{
		 #pragma omp for schedule(dynamic)
		 for (i = 0; i < N; i++)
		 {
			  //instead of checking all elements, we will just check for close by elements from our sourted array
				//so we will check points within an E <= array[].X length away from this point
//printf("%d\n",i,ToRight);
				int ToRight = i;

				x_1 = data[i].x;
				y_1 = data[i].y;
				//
				double RightExtent = data[i].x + epsilon;
				//check forward points
				//(data[ToRight].x <= RightExtent) &&
				while((data[ToRight].x <= (data[i].x + epsilon))  &&(ToRight < N-1)) //(data[ToRight].x > (data[i].x + epsilon))  &&
				{

				  	//check point above
						 ToRight++;
						 if((data[ToRight].x < RightExtent))
						 {
							 //printf("%d,%d .x:%f,Ex:%f ->%d\n",i,ToRight,data[ToRight].x,RightExtent,(data[ToRight].x < RightExtent));
						 }
//
						 x_2 = data[ToRight].x;
						 y_2 = data[ToRight].y;
						 distance = sqrt(pow(x_1 - x_2, 2) + pow(y_1 - y_2, 2));
						 if (distance <= epsilon)
						 {
								#pragma omp atomic
								total_count+=2;
						 }
				}


				//check backward points
				//while((data[i].x-epsilon < data[ToRight].x))
				//{

				//}





		 }
	}


	total_count += N;

	double tend=omp_get_wtime();

	printf("\n\nTotal time (s): %f",tend-tstart);
  printf("\nN = %i, epsilon = %f, total count: %i", N, epsilon, total_count);


	free(data);
	printf("\n");
	return 0;
}

/* Function to print an array */
void printPointArray(struct pointData * data)
{
	  printf("Data Array:\n");
    for (unsigned int i = 0; i < N; i++)
		{
			printf("Data[%d] = (%f,%f) Located At: %p\n",i, data[i].x, data[i].y, &data[i]);
		}


}

//Do not modify the dataset generator or you will get the wrong answer
void generateDataset(struct pointData * data)
{
	//seed RNG
	srand(SEED);

	for (unsigned int i=0; i<N; i++){
		data[i].x=1000.0*((double)(rand()) / RAND_MAX);
		data[i].y=1000.0*((double)(rand()) / RAND_MAX);
	}

}



//Sort Array Functions
void SortPointArray_ByX(struct pointData* data)
{
	int length = N-1;
	quickSortPointArray_ByX(data, 0, length);
}

void swap_pointData(struct pointData * a, struct pointData * b)
{
    struct pointData t = *a;
    *a = *b;
    *b = t;
}

int partition(struct pointData arr[], int low, int high)
{
    double pivot = arr[high].x;    // pivot
    int i = (low - 1);  // Index of smaller element

    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot

				//printf("%f\n",arr[j]->x);
        if (arr[j].x <= pivot)
        {
						//printf("%f\n",arr[j].x);
            i++;    // increment index of smaller element
            swap_pointData(&arr[i], &arr[j]);
        }
    }
    swap_pointData(&arr[i + 1], &arr[high]);
    return (i + 1);
}

//Example: quickSort(arr, 0, n-1);
void quickSortPointArray_ByX(struct pointData arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSortPointArray_ByX(arr, low, pi - 1);
        quickSortPointArray_ByX(arr, pi + 1, high);
    }
}

int ArrayinOrder(struct pointData data[])
{
	for (int i = 0; i < N-1; i+=2)
	{
				double x_1 = data[i].x;

				double x_2 = data[i+1].x;

				if (x_1 > x_2)
				{
					 printf("\n%f, %f :Fix that sort algo!!!\n",x_1,x_2);
					 printf("\n%d, %d :Fix that sort algo!!!\n",i,i+1);
					 printf("Fix that sort algo!!!\n");
					 printf("Fix that sort algo!!!\n");
					 return 0;

				}
		}
		return 1;
}
//Set Range double = is 1

int PointArray_BinarySearch(struct pointData arr[], double SearchFor, double low,double high,int E)
{
	if (high >= low)
	{
			 int mid = low + (high - low)/2;

			 // If the element is present at the middle
			 // itself
			 if ((SearchFor-E <= arr[mid].x) && (arr[mid].x<= SearchFor+E))
					 return mid;

			 // If element is smaller than mid, then
			 // it can only be present in left subarray
			 if (arr[mid].x > SearchFor)
			 {
					 return PointArray_BinarySearch(arr, SearchFor, low, mid-1,E);
				 //					 return binarySearch(arr, low, mid-1, SearchFor);
			 }


			 // Else the element can only be present
			 // in right subarray
			 return PointArray_BinarySearch(arr, SearchFor, mid+1, high,E);
	}

	// We reach here when element is not present in array
	return -1;
}

//Returns location of element
int binarySearch(int arr[], int l, int r, int x)
{
   if (r >= l)
   {
        int mid = l + (r - l)/2;

        // If the element is present at the middle
        // itself
        if (arr[mid] == x)
            return mid;

        // If element is smaller than mid, then
        // it can only be present in left subarray
        if (arr[mid] > x)
            return binarySearch(arr, l, mid-1, x);

        // Else the element can only be present
        // in right subarray
        return binarySearch(arr, mid+1, r, x);
   }

   // We reach here when element is not
   // present in array
   return -1;
}

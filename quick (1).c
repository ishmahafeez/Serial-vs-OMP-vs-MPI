#include<stdio.h>
#include<omp.h>
#include<pthread.h>
#include<stdlib.h>

#define ARRAY_MAX_SIZE 50000

int arr[ARRAY_MAX_SIZE];

int n;
struct values{
	
	int low;
	int high;
};



void *quick_sort( void *__v1)
{
struct values *v1=(struct values*) __v1;
     if (v1->low < v1->high)
    {

        int p = partition(arr, v1->low, v1->high);
        pthread_t tid1,tid2;
        struct values t;
        t.low=v1->low;
        t.high=p-1;
        pthread_create(&tid1,NULL,quick_sort,&t);
	pthread_join(tid1, NULL);

	t.low=p+1;
        t.high=v1->high;
        pthread_create(&tid2,NULL,quick_sort,&t);
      	pthread_join(tid2, NULL);
    }

}




void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

int partition (int arr[], int low, int high)
{
	int pivot = arr[high]; 
	int i = (low - 1); 
	for (int j = low; j <= high- 1; j++)
	{
		
		if (arr[j] <= pivot)
		{
			i++;
			swap(&arr[i], &arr[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	return (i + 1);
}


void quicksortomp(int arr[], int low, int high)
{
	if (low < high)
	{

		int pi = partition(arr, low, high);

		#pragma omp task firstprivate(arr,low,pi)
		{
			quicksortomp(arr,low, pi - 1);

		}

		#pragma omp task firstprivate(arr, high,pi)
		{
			quicksortomp(arr, pi + 1, high);

		}


	}
}

void quicksortserial( int arr[], int low, int high)
{
    if (low < high)
    {

        int p = partition( arr,low, high);
        quicksortserial( arr,low, p - 1);
        quicksortserial(arr,p + 1, high);
    }
}

void printarray(int a[],int s){

for (int i = 0; i < s; i++){
        printf("%d ",a[i]);
    }

	printf("\n");
}

void randomize() {
    srand(time(NULL));
    int i;
    for (i = 0; i < n; i++) {
        arr[i] = rand() % 1000 + 1;
        
    }
}


int main()
{
	srand(time(NULL));
	printf("---500 elements---\n");
	n=500;
	double start_time, run_time;
	randomize();


	start_time = omp_get_wtime();
	#pragma omp parallel
	{
		#pragma omp single
		 quicksortomp(arr,0, n-1);

	}
	run_time = omp_get_wtime() - start_time;
	printf("\n OMP Execution time was %lf seconds\n ",run_time);
	printarray(arr,n);


	randomize();
	start_time = omp_get_wtime();
	quicksortserial(arr,0,n-1);
	run_time = omp_get_wtime() - start_time;
	printf("\n Serial Execution time was %lf seconds\n ",run_time);
	printarray(arr,n);

	struct values val;
	val.low=0;
	val.high=n-1;
	pthread_t tid3;
	randomize();
	start_time = omp_get_wtime();
	pthread_create(&tid3,NULL,quick_sort,&val);
	pthread_join(tid3, NULL);
	run_time = omp_get_wtime() - start_time;
	printf("\n Pthread Execution time was %lf seconds\n ",run_time);
	
	//printarray(arr,n);
	
	
	
	
	
	
	
	
	
	printf("\n\n---5000 elements---\n");
	n=5000;
	randomize();


	start_time = omp_get_wtime();
	#pragma omp parallel
	{
		#pragma omp single
		 quicksortomp(arr,0, n-1);

	}
	run_time = omp_get_wtime() - start_time;
	printf("\n OMP Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);


	randomize();
	start_time = omp_get_wtime();
	quicksortserial(arr,0,n-1);
	run_time = omp_get_wtime() - start_time;
	printf("\n Serial Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);

	val.low=0;
	val.high=n-1;
	randomize();
	start_time = omp_get_wtime();
	pthread_create(&tid3,NULL,quick_sort,&val);
	pthread_join(tid3, NULL);
	run_time = omp_get_wtime() - start_time;
	printf("\n Pthread Execution time was %lf seconds\n ",run_time);
	
	//printarray(arr,n);
	
	
	






	printf("\n\n---50000 elements---\n");
	n=50000;
	randomize();


	start_time = omp_get_wtime();
	#pragma omp parallel 
	{
		#pragma omp single
		 quicksortomp(arr,0, n-1);

	}
	run_time = omp_get_wtime() - start_time;
	printf("\n OMP Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);


	randomize();
	start_time = omp_get_wtime();
	quicksortserial(arr,0,n-1);
	run_time = omp_get_wtime() - start_time;
	printf("\n Serial Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);

	val.low=0;
	val.high=n-1;
	randomize();
	start_time = omp_get_wtime();
	pthread_create(&tid3,NULL,quick_sort,&val);
	pthread_join(tid3, NULL);
	run_time = omp_get_wtime() - start_time;
	printf("\n Pthread Execution time was %lf seconds\n ",run_time);
	
	//printarray(arr,n);












}


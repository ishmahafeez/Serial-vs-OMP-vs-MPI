#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <time.h>
#include<pthread.h>

#define max_num 100000
int n;
int arr[max_num];

struct values{

int low;
int high;


};

void printarray(int a[],int s){

for (int i = 0; i < s; i++){
        printf("%d ",a[i]);
    }

	printf("\n");
}

void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

void selectionsortserial()
{
    for (int i = n - 1; i > 0; --i)
    {
    
        int val = arr[i];
        int index = i;
        for (int j = i - 1; j >= 0; --j)
        {
            if (arr[j] > val)
            {
                val = arr[j];
                index = j;
            }
        }
        int tmp = arr[i];
        arr[i] = val;
        arr[index] = tmp;
    }
}






void selectionsortomp()
{
    
    
    for (int i = n - 1; i > 0; --i)
    {
      
        int val = arr[i];
        int index = i;
        #pragma omp parallel for
        for (int j = i - 1; j >= 0; --j)
        {
            if (arr[j] > val)
            {
                val = arr[j];
                index = j;
            }
        }
        int tmp = arr[i];
        arr[i] = val;
        arr[index] = tmp;
    }
}


void randomize() {
    srand(time(NULL));
    int i;
    for (i = 0; i < n; i++) {
        arr[i] = rand() % 1000 + 1;
     
    }
}

int main(){



	printf("---500 elements---\n");
	n=500;
	double start_time, run_time;
	randomize();


	start_time = omp_get_wtime();

		 selectionsortomp();

	run_time = omp_get_wtime() - start_time;
	printf("\n OMP Execution time was %lf seconds\n ",run_time);
	printarray(arr,n);


	randomize();
	start_time = omp_get_wtime();
	selectionsortserial();
	run_time = omp_get_wtime() - start_time;
	printf("\n Serial Execution time was %lf seconds\n ",run_time);
	printarray(arr,n);


	pthread_t tid3;
	randomize();
	start_time = omp_get_wtime();
	pthread_create(&tid3,NULL,(void*)selectionsortserial,NULL);
	pthread_join(tid3, NULL);
	run_time = omp_get_wtime() - start_time;
	printf("\n Pthread Execution time was %lf seconds\n ",run_time);
	
	printarray(arr,n);
	
	
	
	
	
	
	
	
	
	printf("\n\n---5000 elements---\n");
	n=5000;
	randomize();


	start_time = omp_get_wtime();

		 selectionsortomp();

	run_time = omp_get_wtime() - start_time;
	printf("\n OMP Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);


	randomize();
	start_time = omp_get_wtime();
	selectionsortserial();
	run_time = omp_get_wtime() - start_time;
	printf("\n Serial Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);


	randomize();
	start_time = omp_get_wtime();
	pthread_create(&tid3,NULL,(void*)selectionsortserial,NULL);
	pthread_join(tid3, NULL);
	run_time = omp_get_wtime() - start_time;
	printf("\n Pthread Execution time was %lf seconds\n ",run_time);
	
	//printarray(arr,n);
	
	
	






	printf("\n\n---50000 elements---\n");
	n=50000;
	randomize();


	start_time = omp_get_wtime();

		 selectionsortomp();

	run_time = omp_get_wtime() - start_time;
	printf("\n OMP Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);


	randomize();
	start_time = omp_get_wtime();
	selectionsortserial();
	run_time = omp_get_wtime() - start_time;
	printf("\n Serial Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);

	
	randomize();
	start_time = omp_get_wtime();
	pthread_create(&tid3,NULL,(void*)selectionsortserial,NULL);
	pthread_join(tid3, NULL);
	run_time = omp_get_wtime() - start_time;
	printf("\n Pthread Execution time was %lf seconds\n ",run_time);
	
	//printarray(arr,n);




}


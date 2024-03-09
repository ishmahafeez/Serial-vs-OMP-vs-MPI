#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <time.h>
#include<pthread.h>

#define max_num 100000
int n;
int a[max_num], sorted[max_num];



void countsortomp() {
    int i, j, count;
    
    #pragma omp parallel private(i, j, count)
    {
        #pragma omp for
        for (i = 0; i < n; i++) {
            count = 0;
            for (j = 0; j < n; j++) {
                if (a[i] > a[j])
                    count++;
            }
            while (sorted[count] != 0)
                count++;
            sorted[count] = a[i];
        }
    }   
}



void countsortserial(){    

    int i, j, count;

    for (i = 0; i < n; i++) {
        count = 0;
        for (j = 0; j < n; j++) {
            if (a[i] > a[j]) {
                count++;
            }
        }
        while (sorted[count] != 0)
            count++;
        sorted[count] = a[i];
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
        a[i] = rand() % 1000 + 1;
        sorted[i] = 0;
    }
}

int main(){



	printf("---500 elements---\n");
	n=500;
	double start_time, run_time;
	randomize();


	start_time = omp_get_wtime();

		 countsortomp();

	run_time = omp_get_wtime() - start_time;
	printf("\n OMP Execution time was %lf seconds\n ",run_time);
	printarray(sorted,n);


	randomize();
	start_time = omp_get_wtime();
	countsortserial();
	run_time = omp_get_wtime() - start_time;
	printf("\n Serial Execution time was %lf seconds\n ",run_time);
	printarray(sorted,n);


	pthread_t tid3;
	randomize();
	start_time = omp_get_wtime();
	pthread_create(&tid3,NULL,(void*)countsortserial,NULL);
	pthread_join(tid3, NULL);
	run_time = omp_get_wtime() - start_time;
	printf("\n Pthread Execution time was %lf seconds\n ",run_time);
	
	printarray(sorted,n);
	
	
	
	
	
	
	
	
	
	printf("\n\n---5000 elements---\n");
	n=5000;
	randomize();


	start_time = omp_get_wtime();

		 countsortomp();

	run_time = omp_get_wtime() - start_time;
	printf("\n OMP Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);


	randomize();
	start_time = omp_get_wtime();
	countsortserial();
	run_time = omp_get_wtime() - start_time;
	printf("\n Serial Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);


	randomize();
	start_time = omp_get_wtime();
	pthread_create(&tid3,NULL,(void*)countsortserial,NULL);
	pthread_join(tid3, NULL);
	run_time = omp_get_wtime() - start_time;
	printf("\n Pthread Execution time was %lf seconds\n ",run_time);
	
	//printarray(arr,n);
	
	
	






	printf("\n\n---50000 elements---\n");
	n=50000;
	randomize();


	start_time = omp_get_wtime();

		 countsortomp();

	run_time = omp_get_wtime() - start_time;
	printf("\n OMP Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);


	randomize();
	start_time = omp_get_wtime();
	countsortserial();
	run_time = omp_get_wtime() - start_time;
	printf("\n Serial Execution time was %lf seconds\n ",run_time);
	//printarray(arr,n);

	
	randomize();
	start_time = omp_get_wtime();
	pthread_create(&tid3,NULL,(void*)countsortserial,NULL);
	pthread_join(tid3, NULL);
	run_time = omp_get_wtime() - start_time;
	printf("\n Pthread Execution time was %lf seconds\n ",run_time);
	
	//printarray(arr,n);




}




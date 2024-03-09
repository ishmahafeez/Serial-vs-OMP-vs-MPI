#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <mpi.h>

// Structure to represent a point in 2D space
typedef struct {
    double x, y;
} Point;

// Function to calculate the Euclidean distance between two points
double euclideanDistance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// Function to compare two points based on their x-coordinate
int compareX(const void* a, const void* b) {
    return ((Point*)a)->x - ((Point*)b)->x;
}

// Function to compare two points based on their y-coordinate
int compareY(const void* a, const void* b) {
    return ((Point*)a)->y - ((Point*)b)->y;
}

// Function to find the minimum of two double values
double min(double a, double b) {
    return (a < b) ? a : b;
}

// Function to find the closest pair of points in a strip
double stripClosest(Point strip[], int size, double d) {
    double minDist = d;

    // Check for points closer than d in the strip
    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size && (strip[j].y - strip[i].y) < minDist; ++j) {
            double dist = euclideanDistance(strip[i], strip[j]);
            minDist = min(minDist, dist);
        }
    }

    return minDist;
}

// Function to find the closest pair of points using divide and conquer with MPI parallelization
double closestPairMPI(Point points[], int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Broadcast the number of points to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the number of points each process will handle
    int local_size = n / size;
    int remainder = n % size;

    // Calculate the starting and ending indices for the current process
    int start = rank * local_size + (rank < remainder ? rank : remainder);
    int end = start + local_size + (rank < remainder ? 1 : 0);
    



    // Allocate memory for local points
    Point* local_points = (Point*)malloc(local_size * sizeof(Point));

    // Scatter the points to all processes
    MPI_Scatter(points, local_size * sizeof(Point), MPI_BYTE, local_points, local_size * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Sort local points by x-coordinate
    qsort(local_points, local_size, sizeof(Point), compareX);

    // Find the closest pair in the local subset
    double local_min_dist = DBL_MAX;
    for (int i = 0; i < local_size; ++i) {
        for (int j = i + 1; j < local_size; ++j) {
            double dist = euclideanDistance(local_points[i], local_points[j]);
            local_min_dist = min(local_min_dist, dist);
        }
    }

    // Gather the local minimum distances from all processes
    double* all_min_dists = (double*)malloc(size * sizeof(double));
    MPI_Gather(&local_min_dist, 1, MPI_DOUBLE, all_min_dists, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Process 0 finds the minimum distance among all gathered distances
    double global_min_dist = DBL_MAX;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            global_min_dist = min(global_min_dist, all_min_dists[i]);
        }
    }

    // Broadcast the global minimum distance to all processes
    MPI_Bcast(&global_min_dist, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Free allocated memory
    free(local_points);
    free(all_min_dists);

    return global_min_dist;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Different dataset sizes
    int datasetSizes[] = {10, 100, 1000, 10000, 50000};

    for (int datasetIndex = 0; datasetIndex < sizeof(datasetSizes) / sizeof(datasetSizes[0]); ++datasetIndex) {
        int n = datasetSizes[datasetIndex];

        Point* points = NULL;
        int local_size = n / size;
        int remainder = n % size;

        if (rank == 0) {
            points = (Point*)malloc(n * sizeof(Point));
            srand(time(NULL));
            for (int i = 0; i < n; ++i) {
                points[i].x = (double)rand() / RAND_MAX * 100.0; // Adjust the range as needed
                points[i].y = (double)rand() / RAND_MAX * 100.0;
            }
        }

        // Broadcast the number of points to all processes
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate the starting and ending indices for the current process
    int start = rank * local_size + (rank < remainder ? rank : remainder);
    int end = start + local_size + (rank < remainder || local_size == 0 ? 1 : 0);


        // Allocate memory for local points
        Point* local_points = (Point*)malloc(local_size * sizeof(Point));

        // Scatter the points to all processes
        MPI_Scatter(points, local_size * sizeof(Point), MPI_BYTE, local_points, local_size * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
        

        // Measure the time taken
        double start_time, end_time;
        MPI_Barrier(MPI_COMM_WORLD); 
        start_time = MPI_Wtime();

        
        double result = closestPairMPI(local_points, local_size);

      
        MPI_Barrier(MPI_COMM_WORLD);
        end_time = MPI_Wtime();
        if (rank == 0) {
            printf("\nDataset Size: %d\n", n);
            printf("Result: %lf\n", result);
            printf("MPI time: %lf seconds\n", end_time - start_time);
        }

        // Free allocated memory
        free(local_points);
        if (rank == 0) {
            free(points);
        }
    }

    MPI_Finalize();

    return 0;
}


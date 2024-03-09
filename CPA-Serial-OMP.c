#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <omp.h>

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

    // Sort the strip based on y-coordinate
    qsort(strip, size, sizeof(Point), compareY);

    // Check for points closer than d in the strip (parallelized)
    #pragma omp parallel for schedule(dynamic) //if one thread is done it will take another task
    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size && (strip[j].y - strip[i].y) < minDist; ++j) {
            double dist = euclideanDistance(strip[i], strip[j]);
            #pragma omp critical //only one thread can access this at a time
            {
                minDist = min(minDist, dist);
            }
        }
    }//when we have a lot of threads, we don't want to have a lot of threads accessing the same variable at the same time  

    return minDist;
}

// Function to find the closest pair of points using divide and conquer
double closestPairSerial(Point points[], int n) {
    // If there are fewer than 3 points, simply calculate the distance
    if (n <= 3) {
        double minDist = DBL_MAX;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dist = euclideanDistance(points[i], points[j]);
                minDist = min(minDist, dist);
            }
        }
        return minDist;
    }


    // Sort points by x-coordinate
    qsort(points, n, sizeof(Point), compareX);

    // Divide the points into two halves
    int mid = n / 2;
    Point midPoint = points[mid];

    // Recursively find the closest pair in each half
    double leftMinDist = closestPairSerial(points, mid);
    double rightMinDist = closestPairSerial(points + mid, n - mid);
    double minDist = min(leftMinDist, rightMinDist);

    // Find the closest pair in the strip , too find how many points
    Point* strip = (Point*)malloc(n * sizeof(Point));
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(points[i].x - midPoint.x) < minDist) {
            strip[j] = points[i];
            j++;
        }
    }

    // Calculate the closest pair in the strip
    double stripMinDist = stripClosest(strip, j, minDist);
    free(strip);

    return min(minDist, stripMinDist);
}

// Function to find the closest pair of points using divide and conquer;parallelized
double closestPairParallel(Point points[], int n, int threshold) {
    // If there are fewer than the threshold points, simply calculate the distance
    if (n <= threshold) {
        double minDist = DBL_MAX;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dist = euclideanDistance(points[i], points[j]);
                minDist = min(minDist, dist);
            }
        }
        return minDist;
    }


    // Sort points by x-coordinate
    qsort(points, n, sizeof(Point), compareX);

    // Divide the points into two halves
    int mid = n / 2;
    Point midPoint = points[mid];

    // Recursively find the closest pair in each half (in parallel)
    double leftMinDist, rightMinDist;
#pragma omp parallel sections //parallise sections  
    {
#pragma omp section
        leftMinDist = closestPairParallel(points, mid, threshold);
#pragma omp section
        rightMinDist = closestPairParallel(points + mid, n - mid, threshold);
    }

    // Find the closest pair in the strip
    Point* strip = (Point*)malloc(n * sizeof(Point));
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(points[i].x - midPoint.x) < min(leftMinDist, rightMinDist)) {
            strip[j] = points[i];
            j++;
        }
    }

    // Calculate the closest pair in the strip
    double stripMinDist = stripClosest(strip, j, min(leftMinDist, rightMinDist));
    free(strip);
  

    return min(min(leftMinDist, rightMinDist), stripMinDist);
}

// Function to copy an array of points
Point* copyPoints(const Point* source, int n) {
    Point* copy = (Point*)malloc(n * sizeof(Point));
    if (copy == NULL) {
        // Handle memory allocation failure
        exit(EXIT_FAILURE);
    }

    // Copy each point
    for (int i = 0; i < n; ++i) {
        copy[i] = source[i];
    }

    return copy;
}

int main() {

	omp_set_num_threads(8);
    // Array of dataset sizes
    int datasetSizes[] = {10, 100, 1000, 10000, 50000};
    
    // Loop through different dataset sizes
    for (int i = 0; i < sizeof(datasetSizes) / sizeof(datasetSizes[0]); ++i) {
        int n = datasetSizes[i];

        // Generate a new dataset of random points
        Point* points = (Point*)malloc(n * sizeof(Point));
        srand(time(NULL));
        for (int j = 0; j < n; ++j) {
            points[j].x = (double)rand() / RAND_MAX * 100.0; // Adjust the range as needed
            points[j].y = (double)rand() / RAND_MAX * 100.0;
        }

        // Create a copy of the original array
        Point* pointsCopy = copyPoints(points, n);

        // Serial version
        clock_t start_time = clock();
        double serialResult = closestPairSerial(pointsCopy, n);
        clock_t end_time = clock();
        double serialTime = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

        // Parallel version
        int threshold = 200; // what is the best threshold?
        start_time = clock();
        double parallelResult = closestPairParallel(pointsCopy, n, threshold);
        end_time = clock();
        double parallelTime = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

        // Print results and times
        printf("Dataset size: %d\n", n);
        printf("Serial result: %lf\n", serialResult);
        printf("OpenMP result: %lf\n", parallelResult);
        printf("Serial time: %lf seconds\n", serialTime);
        printf("OpenMP time: %lf seconds\n", parallelTime);
        printf("\n");

        // Clean up
        free(points);
        free(pointsCopy);
    }




    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
// Define point structure
typedef struct {
    double x, y;
} Point;

// Compare function for qsort
int compare(const void* a, const void* b) {
    Point *p1 = (Point*)a, *p2 = (Point*)b;
    if (p1->x < p2->x) return -1;
    if (p1->x > p2->x) return 1;
    return 0;
}

// Orientation function
double orientation(Point p, Point q, Point r) {
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
}

// Graham Scan algorithm (serial version)
void grahamScanSerial(Point* points, int n, int* hull_size, Point* hull) {
    // Sort the points based on x-coordinate
    qsort(points, n, sizeof(Point), compare);

    // Initialize the convex hull
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        while (idx >= 2 && orientation(hull[idx - 2], hull[idx - 1], points[i]) <= 0) {
            idx--;
        }
        hull[idx++] = points[i];
    }

    // Build upper hull
    for (int i = n - 2, t = idx + 1; i >= 0; --i) {
        while (idx >= t && orientation(hull[idx - 2], hull[idx - 1], points[i]) <= 0) {
            idx--;
        }
        hull[idx++] = points[i];
    }

    // Set the size of the convex hull
    *hull_size = idx - 1;
}

// Graham Scan algorithm (OpenMP version)
void grahamScanOpenMP(Point* points, int n, int* hull_size, Point* hull) {
    // Sort the points based on x-coordinate using OpenMP parallelization
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        hull[i] = points[i];
    }

    // Initialize the convex hull with OpenMP parallelization
    int idx = 0;
    #pragma omp parallel
    {
        int local_idx = 0;
        Point local_hull[n];

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            while (local_idx >= 2 && orientation(local_hull[local_idx - 2], local_hull[local_idx - 1], points[i]) <= 0) {
                local_idx--;
            }
            local_hull[local_idx++] = points[i];
        }

        // Merge local hulls into the global hull
        #pragma omp critical
        {
            for (int k = 0; k < local_idx; ++k) {
                while (idx >= 2 && orientation(hull[idx - 2], hull[idx - 1], local_hull[k]) <= 0) {
                    idx--;
                }
                hull[idx++] = local_hull[k];
            }
        }
    }

    // Set the size of the convex hull
    *hull_size = idx - 1;
}

void generateRandomPoints(Point points[], int n) {
    for (int i = 0; i < n; i++) {
        points[i].x = rand() % 1000;
        points[i].y = rand() % 1000;
    }
}

int main() {
    // Set the number of OpenMP threads
    omp_set_num_threads(8);

    // Define dataset sizes
    int dataset_sizes[5] = {10, 100, 1000, 10000, 50000};

    for (int d = 0; d < 5; ++d) {
        int n = dataset_sizes[d];

        // Allocate memory for the current dataset
        Point* all_points = (Point*)malloc(n * sizeof(Point));

        // Generate random points for the current dataset
        generateRandomPoints(all_points, n);

	 printf("For n = %d:\n", n);
        // Serial version
        double start_time_serial = omp_get_wtime();
        int final_hull_size_serial;
        Point* final_hull_serial = (Point*)malloc(n * sizeof(Point));
        grahamScanSerial(all_points, n, &final_hull_size_serial, final_hull_serial);
        double end_time_serial = omp_get_wtime();


        printf("Serial Time: %f seconds\n",end_time_serial - start_time_serial);

        // OpenMP version
        double start_time_openmp = omp_get_wtime();
        int final_hull_size_openmp;
        Point* final_hull_openmp = (Point*)malloc(n * sizeof(Point));
        grahamScanOpenMP(all_points, n, &final_hull_size_openmp, final_hull_openmp);
        double end_time_openmp = omp_get_wtime();

        // Print execution time for the OpenMP version
        printf("OpenMP Time: %f seconds\n\n", end_time_openmp - start_time_openmp);

        // Free allocated memory for the current dataset
        free(all_points);
        free(final_hull_serial);
        free(final_hull_openmp);
    }

    return 0;
}


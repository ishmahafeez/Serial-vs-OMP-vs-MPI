#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

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

// Graham Scan algorithm
void grahamScan(Point* points, int n, int* hull_size, Point* hull) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Dataset sizes
    int dataset_sizes[] = {10, 100, 1000, 10000, 50000};
    int num_datasets = sizeof(dataset_sizes) / sizeof(dataset_sizes[0]);

    for (int d = 0; d < num_datasets; ++d) {
        int n = dataset_sizes[d];

        // Initialize random points for the current dataset size
        srand(12345 + rank);
        Point* all_points = (Point*)malloc(n * sizeof(Point));
        for (int i = 0; i < n; ++i) {
            all_points[i].x = rand() % 1000;
            all_points[i].y = rand() % 1000;
        }

        // Distribute points among processes
        int local_n = n / size;
        Point* local_points = (Point*)malloc(local_n * sizeof(Point));
        MPI_Scatter(all_points, local_n * 2, MPI_DOUBLE, local_points, local_n * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Perform Graham Scan locally
        int local_hull_size;
        Point* local_hull = (Point*)malloc(local_n * sizeof(Point));
        double start_time = MPI_Wtime();
        grahamScan(local_points, local_n, &local_hull_size, local_hull);
        double end_time = MPI_Wtime();

        // Gather local hulls to process 0
        Point* all_hull = NULL;
        int* all_hull_sizes = NULL;
        if (rank == 0) {
            all_hull = (Point*)malloc(n * sizeof(Point));
            all_hull_sizes = (int*)malloc(size * sizeof(int));
        }

        MPI_Gather(local_hull, local_n * 2, MPI_DOUBLE, all_hull, local_n * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_hull_size, 1, MPI_INT, all_hull_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Combine local hulls into the final convex hull
        if (rank == 0) {
            int final_hull_size;
            Point* final_hull = (Point*)malloc(n * sizeof(Point));
            grahamScan(all_hull, n, &final_hull_size, final_hull);

           //printf("Dataset Size: %d, Execution Time: %f seconds\n", n, end_time - start_time);
            printf("Dataset Size: %d, Execution Time: %f seconds\n", n, end_time - start_time);

            free(final_hull);
            free(all_hull_sizes);
            free(all_hull);
        }

        free(all_points);
        free(local_points);
        free(local_hull);
    }

    MPI_Finalize();
    return 0;
}



#include "kmeans_worker.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

void assign_clusters(const double *local_data, int local_N,
                     const double *centroids, int K, int D,
                     int *labels, double *local_sums, int *local_counts)
{
    if (local_N < 0 || K <= 0 || D <= 0 ||
        local_sums == NULL || local_counts == NULL) {
        fprintf(stderr, "kmeans_worker: invalid dimensions or output buffers\n");
        abort();
    }
    if (K > INT_MAX / D || local_N > INT_MAX / D) {
        fprintf(stderr, "kmeans_worker: array dimensions exceed int indexing limit\n");
        abort();
    }
    if (local_N > 0 &&
        (local_data == NULL || centroids == NULL || labels == NULL)) {
        fprintf(stderr, "kmeans_worker: missing data, centroids or labels\n");
        abort();
    }

    /* Start fresh for each K-Means iteration, including ranks with no points. */
    for (int cluster = 0; cluster < K; ++cluster) {
        local_counts[cluster] = 0;
        for (int dim = 0; dim < D; ++dim) {
            local_sums[cluster * D + dim] = 0.0;
        }
    }

    for (int point = 0; point < local_N; ++point) {
        int nearest = 0;
        double best_distance = 0.0;

        for (int cluster = 0; cluster < K; ++cluster) {
            double distance = 0.0;
            for (int dim = 0; dim < D; ++dim) {
                double difference = local_data[point * D + dim]
                                  - centroids[cluster * D + dim];
                distance += difference * difference;
            }
            /* Strict comparison keeps the lowest cluster index on a tie. */
            if (cluster == 0 || distance < best_distance) {
                best_distance = distance;
                nearest = cluster;
            }
        }

        labels[point] = nearest;
        ++local_counts[nearest];
        for (int dim = 0; dim < D; ++dim) {
            local_sums[nearest * D + dim] += local_data[point * D + dim];
        }
    }
}

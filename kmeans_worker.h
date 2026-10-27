#ifndef KMEANS_WORKER_H
#define KMEANS_WORKER_H

/* Row-major data and centroids; sums and counts need K * D and K elements.
 * Input coordinates must be finite, with squared distances fitting in double.
 * Invalid dimensions or missing required buffers abort the program.
 * With local_N == 0, local_data, labels and centroids may be NULL.
 */
void assign_clusters(const double *local_data, int local_N,
                     const double *centroids, int K, int D,
                     int *labels, double *local_sums, int *local_counts);

#endif

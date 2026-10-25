#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H

typedef struct {
    int N;          // Number of data points
    int K;          // Number of clusters
    int D;          // Number of features per point
    int max_iter;   // Maximum number of iterations
} KMeansConfig;

/* Read and validate project configuration */
int read_config(KMeansConfig *config);

/* Generate a dataset containing N points, each with D features */
int generate_dataset(double *data, int N, int D);

/* Initialize K centroids using the first K data points */
int initialize_centroids(
    const double *data,
    double *centroids,
    int N,
    int K,
    int D
);

#endif
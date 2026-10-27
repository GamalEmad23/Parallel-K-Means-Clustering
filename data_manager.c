#include <stdio.h>
#include <stdlib.h>
#include "data_manager.h"



int read_config(KMeansConfig *config)
{
    if (config == NULL)
        return 0;

    printf("Enter number of data points (N): ");
    fflush(stdout);

    if (scanf("%d", &config->N) != 1)
    {
        printf("\nInvalid input for N.\n");
        return 0;
    }

    printf("Enter number of clusters (K): ");
    fflush(stdout);

    if (scanf("%d", &config->K) != 1)
    {
        printf("\nInvalid input for K.\n");
        return 0;
    }

    printf("Enter number of features (D): ");
    fflush(stdout);

    if (scanf("%d", &config->D) != 1)
    {
        printf("\nInvalid input for D.\n");
        return 0;
    }

    printf("Enter maximum iterations: ");
    fflush(stdout);

    if (scanf("%d", &config->max_iter) != 1)
    {
        printf("\nInvalid input for max_iter.\n");
        return 0;
    }

    if (config->N <= 0 ||
        config->K <= 0 ||
        config->D <= 0 ||
        config->max_iter <= 0 ||
        config->K > config->N)
    {
        printf("\nError: Invalid configuration.\n");
        printf("N, K, D, and max_iter must be positive, ");
        printf("and K must be <= N.\n");
        return 0;
    }

    return 1;
}



int generate_dataset(double *data, int N, int D)
{
    if (data == NULL || N <= 0 || D <= 0)
        return 0;

    /*
       Generate N data points.
       Each point contains D features.
       Feature values are random numbers between 0 and 100.
    */

    for (int i = 0; i < N * D; i++)
    {
        data[i] = ((double)rand() / RAND_MAX) * 100.0;
    }

    return 1;
}

int initialize_centroids(
    const double *data,
    double *centroids,
    int N,
    int K,
    int D)
{
    if (data == NULL ||
        centroids == NULL ||
        N <= 0 ||
        K <= 0 ||
        D <= 0 ||
        K > N)
    {
        return 0;
    }

    /*
       Use the first K data points as initial centroids.
       Each centroid contains D features.
    */

    for (int k = 0; k < K; k++)
    {
        for (int j = 0; j < D; j++)
        {
            centroids[k * D + j] = data[k * D + j];
        }
    }

    return 1;
}
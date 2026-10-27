
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "data_manager.h"
#include "workload.h"
#include "kmeans_worker.h"
#include "coordinator.h"
#include "performance.h"

#define CONVERGENCE_TOLERANCE 1e-4

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    KMeansConfig config = {0};
    int valid = 1;

    /* Read configuration */
    if (rank == 0)
    {
        printf("=== K-Means ===\n");
        fflush(stdout);
        valid = read_config(&config);
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!valid)
    {
        MPI_Finalize();
        return 1;
    }

    /* Broadcast configuration */
    int settings[4];

    if (rank == 0)
    {
        settings[0] = config.N;
        settings[1] = config.K;
        settings[2] = config.D;
        settings[3] = config.max_iter;
    }

    MPI_Bcast(settings, 4, MPI_INT, 0, MPI_COMM_WORLD);

    config.N = settings[0];
    config.K = settings[1];
    config.D = settings[2];
    config.max_iter = settings[3];

    /* Allocate global data and centroids */
    double *data = NULL;
    double *centroids = NULL;

    if (rank == 0)
    {
        data = malloc(
            (size_t)config.N * config.D * sizeof(double));
    }

    centroids = malloc(
        (size_t)config.K * config.D * sizeof(double));

    int allocation_ok =
        (rank != 0 || data != NULL) && centroids != NULL;

    int all_allocations_ok = 0;

    MPI_Allreduce(
        &allocation_ok, &all_allocations_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (!all_allocations_ok)
    {
        if (rank == 0)
            free(data);

        free(centroids);

        if (rank == 0)
            fprintf(stderr, "Error: memory allocation failed.\n");

        MPI_Finalize();
        return 1;
    }

    /* Generate dataset and initialize centroids */
    int setup_ok = 1;

    if (rank == 0)
    {
        srand(42);

        setup_ok = generate_dataset(
            data, config.N, config.D);

        if (setup_ok)
        {
            setup_ok = initialize_centroids(
                data, centroids,
                config.N, config.K, config.D);
        }
    }

    MPI_Bcast(&setup_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!setup_ok)
    {
        if (rank == 0)
            free(data);

        free(centroids);

        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(
        centroids,
        config.K * config.D,
        MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    /* Allocate local workload */
    int local_N = config.N / size + (rank < config.N % size ? 1 : 0);

    double *local_data = NULL;
    int *labels = NULL;
    double *local_sums = NULL;
    int *local_counts = NULL;

    if (local_N > 0)
    {
        local_data = malloc(
            (size_t)local_N * config.D * sizeof(double));

        labels = malloc(
            (size_t)local_N * sizeof(int));
    }

    local_sums = malloc(
        (size_t)config.K * config.D * sizeof(double));

    local_counts = malloc(
        (size_t)config.K * sizeof(int));

    allocation_ok =
        (local_N == 0 ||
         (local_data != NULL && labels != NULL)) &&
        local_sums != NULL && local_counts != NULL;

    MPI_Allreduce(
        &allocation_ok, &all_allocations_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (!all_allocations_ok)
    {
        if (rank == 0)
            free(data);

        free(centroids);
        free(local_data);
        free(labels);
        free(local_sums);
        free(local_counts);

        if (rank == 0)
            fprintf(stderr,
                    "Error: local memory allocation failed.\n");

        MPI_Finalize();
        return 1;
    }

    /* Start parallel execution timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    distribute_data(
        data, local_data,
        config.N, config.D,
        rank, size, MPI_COMM_WORLD);

    int iterations = 0;
    int converged = 0;
    int algorithm_ok = 1;

    /* Run K-Means */
    for (int iter = 0; iter < config.max_iter; iter++)
    {
        assign_clusters(
            local_data, local_N, centroids,
            config.K, config.D,
            labels, local_sums, local_counts);

        int update_result = update_centroids(
            local_sums, local_counts, centroids,
            config.K, config.D,
            CONVERGENCE_TOLERANCE, MPI_COMM_WORLD);

        iterations = iter + 1;

        if (update_result < 0)
        {
            algorithm_ok = 0;
            break;
        }

        if (update_result == 1)
        {
            converged = 1;
            break;
        }
    }

    double local_elapsed = MPI_Wtime() - start;

    /* Gather timing results */
    double *all_elapsed = NULL;

    if (rank == 0)
    {
        all_elapsed = malloc((size_t)size * sizeof(double));

        if (all_elapsed == NULL)
        {
            fprintf(stderr,
                    "Error: failed to allocate timing results.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(
        &local_elapsed, 1, MPI_DOUBLE,
        all_elapsed, 1, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    double parallel_elapsed =
        measure_parallel_time(
            local_elapsed, MPI_COMM_WORLD);

    /* Print output in the requested order */

    if (rank == 0)
    {
        /* 1. K-Means Results */
        printf("\n--- K-Means Results ---\n");
        printf("Points (N): %d\n", config.N);
        printf("Clusters (K): %d\n", config.K);
        printf("Dimensions (D): %d\n", config.D);
        printf("MPI processes: %d\n", size);
        printf("Iterations: %d\n", iterations);
        printf("Converged: %s\n", converged ? "Yes" : "No");

        /* 2. First data points */
        printf("\nFirst data points:\n");

        int points_to_print =
            config.N < 5 ? config.N : 5;

        for (int i = 0; i < points_to_print; i++)
        {
            printf("Point %d: ", i);

            for (int j = 0; j < config.D; j++)
            {
                printf("%.2f ", data[i * config.D + j]);
            }

            printf("\n");
        }

        /* 3. Initial centroids */
        printf("\nInitial centroids:\n");

        for (int k = 0; k < config.K; k++)
        {
            printf("Centroid %d: ", k);

            for (int j = 0; j < config.D; j++)
            {
                printf("%.2f ", centroids[k * config.D + j]);
            }

            printf("\n");
        }

        /* 4. Per-process timing results */
        printf("\n--- Per-process Timing Results ---\n");

        for (int i = 0; i < size; i++)
        {
            printf("Rank %d elapsed time: %.6f seconds\n",
                   i, all_elapsed[i]);
        }

        /* 5. Maximum parallel execution time */
        printf("\nMaximum parallel execution time: %.6f seconds\n",
               parallel_elapsed);

        if (!algorithm_ok)
        {
            fprintf(stderr, "Error: centroid update failed.\n");
        }

        free(all_elapsed);
    }

    /* Free memory */
    if (rank == 0)
        free(data);

    free(centroids);
    free(local_data);
    free(labels);
    free(local_sums);
    free(local_counts);

    MPI_Finalize();

    return algorithm_ok ? 0 : 1;
}

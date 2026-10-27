#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "data_manager.h"

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    KMeansConfig config;
    int valid = 1;

    /* Step 1: Root process reads the configuration */
    if (rank == 0)
    {
        printf("=== K-Means: Member 1 Test ===\n");
        fflush(stdout);

        valid = read_config(&config);
    }

    /* Inform all processes whether the input is valid */
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!valid)
    {
        MPI_Finalize();
        return 1;
    }

    /* Step 2: Broadcast the configuration to all processes */
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

    /* Step 3: Allocate memory dynamically */
    double *data = NULL;
    double *centroids = NULL;

    if (rank == 0)
    {
        data = malloc((size_t)config.N * config.D * sizeof(double));
    }

    centroids = malloc((size_t)config.K * config.D * sizeof(double));

    int allocation_ok =
        (rank != 0 || data != NULL) && centroids != NULL;

    int all_allocations_ok;

    MPI_Allreduce(
        &allocation_ok,
        &all_allocations_ok,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    if (!all_allocations_ok)
    {
        if (rank == 0)
            free(data);

        free(centroids);

        if (rank == 0)
            printf("Error: Memory allocation failed.\n");

        MPI_Finalize();
        return 1;
    }

    /* Step 4: Root generates the dataset and initial centroids */
    int setup_ok = 1;

    if (rank == 0)
    {
        srand(42); /* Fixed seed for reproducible testing */

        setup_ok = generate_dataset(
            data,
            config.N,
            config.D
        );

        if (setup_ok)
        {
            setup_ok = initialize_centroids(
                data,
                centroids,
                config.N,
                config.K,
                config.D
            );
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

    /* Step 5: Share the initial centroids with all processes */
    MPI_Bcast(
        centroids,
        config.K * config.D,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /* Step 6: Print test results */
    if (rank == 0)
    {
        printf("\nConfiguration:\n");
        printf("N = %d\n", config.N);
        printf("K = %d\n", config.K);
        printf("D = %d\n", config.D);
        printf("Max iterations = %d\n", config.max_iter);
        printf("Number of MPI processes = %d\n", size);

        printf("\nFirst data points:\n");

        int points_to_print =
            config.N < 5 ? config.N : 5;

        for (int i = 0; i < points_to_print; i++)
        {
            printf("Point %d: ", i);

            for (int j = 0; j < config.D; j++)
            {
                printf(
                    "%.2f ",
                    data[i * config.D + j]
                );
            }

            printf("\n");
        }

        printf("\nInitial centroids:\n");

        for (int k = 0; k < config.K; k++)
        {
            printf("Centroid %d: ", k);

            for (int j = 0; j < config.D; j++)
            {
                printf(
                    "%.2f ",
                    centroids[k * config.D + j]
                );
            }

            printf("\n");
        }

        printf("\nMember 1 setup completed successfully!\n");
    }

    /* Step 7: Free allocated memory */
    if (rank == 0)
        free(data);

    free(centroids);

    MPI_Finalize();

    return 0;
}
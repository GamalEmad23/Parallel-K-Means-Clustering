

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "performance.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Start measuring after all processes are ready */
    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    /* Simulated workload for testing */
    volatile long result = 0;
    for (long i = 0; i < 10000000L; i++)
    {
        result += i % 10;
    }

    double end = MPI_Wtime();
    double local_elapsed = end - start;

    /* Get the maximum elapsed time using our Performance module */
    double maximum_elapsed =
        measure_parallel_time(local_elapsed, MPI_COMM_WORLD);

    /* Collect each process's elapsed time on Root */
    double *all_elapsed = NULL;

    if (rank == 0)
    {
        all_elapsed = malloc((size_t)world_size * sizeof(double));

        if (all_elapsed == NULL)
        {
            fprintf(stderr, "Failed to allocate timing results.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(
        &local_elapsed,
        1,
        MPI_DOUBLE,
        all_elapsed,
        1,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        int slowest_rank = 0;

        printf("\n--- Per-process timing results ---\n");

        for (int i = 0; i < world_size; i++)
        {
            printf("Rank %d local elapsed: %.6f seconds\n",
                   i, all_elapsed[i]);

            if (all_elapsed[i] > all_elapsed[slowest_rank])
            {
                slowest_rank = i;
            }
        }

        printf("\nMaximum parallel elapsed time: %.6f seconds\n",
               maximum_elapsed);

        printf("Slowest process: Rank %d\n", slowest_rank);

        /* Test speedup and efficiency calculations */
        double speedup = calculate_speedup(10.0, 5.0);
        double efficiency = calculate_efficiency(speedup, 2);

        printf("\n--- Performance metrics tests ---\n");
        printf("Speedup: %.2f (expected 2.00)\n", speedup);
        printf("Efficiency: %.2f%% (expected 100.00%%)\n",
               efficiency);

        if (speedup < 1.99 || speedup > 2.01 ||
            efficiency < 99.99 || efficiency > 100.01)
        {
            fprintf(stderr, "Performance metrics test failed.\n");
            free(all_elapsed);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Performance metrics tests passed.\n");

        free(all_elapsed);
    }

    MPI_Finalize();
    return 0;
}

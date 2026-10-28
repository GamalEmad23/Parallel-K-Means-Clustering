#include "workload.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void calculate_distribution(int N, int D, int world_size,
                            int *sendcounts, int *displacements)
{
    if (N < 0 || D <= 0 || world_size <= 0 ||
        sendcounts == NULL || displacements == NULL) {
        fprintf(stderr, "calculate_distribution: invalid dimensions or NULL output arrays.\n");
        fflush(stderr);
        abort();
    }

    if (N > 0 && D > INT_MAX / N) {
        fprintf(stderr, "calculate_distribution: N * D exceeds the MPI int-count limit.\n");
        fflush(stderr);
        abort();
    }

    int points_per_process = N / world_size;
    int extra_points = N % world_size;
    int first_point = 0;

    for (int process = 0; process < world_size; process++) {
        int points = points_per_process;

        if (process < extra_points) {
            points++;
        }

        sendcounts[process] = points * D;
        displacements[process] = first_point * D;
        first_point += points;
    }
}

void distribute_data(const double *all_data, double *local_data,
                     int N, int D, int rank, int world_size, MPI_Comm comm)
{
    int actual_rank, actual_size;
    MPI_Comm_rank(comm, &actual_rank);
    MPI_Comm_size(comm, &actual_size);

    if (rank != actual_rank || world_size != actual_size) {
        fprintf(stderr,
                "Rank %d: supplied rank=%d and world_size=%d; communicator has rank=%d and size=%d.\n",
                actual_rank, rank, world_size, actual_rank, actual_size);
        fflush(stderr);
        MPI_Abort(comm, 1);
    }

    if (N < 0 || D <= 0) {
        fprintf(stderr, "Rank %d: N must be >= 0 and D must be > 0.\n", rank);
        fflush(stderr);
        MPI_Abort(comm, 1);
    }

    if (N > 0 && D > INT_MAX / N) {
        fprintf(stderr, "Rank %d: N * D exceeds the MPI int-count limit.\n", rank);
        fflush(stderr);
        MPI_Abort(comm, 1);
    }

    if ((size_t)world_size > SIZE_MAX / sizeof(int)) {
        fprintf(stderr, "Rank %d: distribution arrays exceed the allocation size limit.\n", rank);
        fflush(stderr);
        MPI_Abort(comm, 1);
    }

    if (rank == 0 && N > 0 && all_data == NULL) {
        fprintf(stderr, "Rank 0: all_data is NULL for a nonempty dataset.\n");
        fflush(stderr);
        MPI_Abort(comm, 1);
    }

    int local_N = N / world_size;
    if (rank < N % world_size) {
        local_N++;
    }

    if (local_N > 0 && local_data == NULL) {
        fprintf(stderr, "Rank %d: local_data is NULL; space for %d points is required.\n",
                rank, local_N);
        fflush(stderr);
        MPI_Abort(comm, 1);
    }

    int *sendcounts = NULL;
    int *displacements = NULL;

    if (rank == 0) {
        sendcounts = malloc((size_t)world_size * sizeof(int));
        displacements = malloc((size_t)world_size * sizeof(int));

        if (sendcounts == NULL) {
            fprintf(stderr, "Rank 0: malloc failed for sendcounts (%zu bytes).\n",
                    (size_t)world_size * sizeof(int));
            fflush(stderr);
            MPI_Abort(comm, 1);
        }

        if (displacements == NULL) {
            fprintf(stderr, "Rank 0: malloc failed for displacements (%zu bytes).\n",
                    (size_t)world_size * sizeof(int));
            fflush(stderr);
            MPI_Abort(comm, 1);
        }

        calculate_distribution(N, D, world_size,
                              sendcounts, displacements);
    }

    double dummy = 0.0;
    const double *send_buffer = all_data;
    if (send_buffer == NULL) {
        send_buffer = &dummy;
    }
    double *receive_buffer = local_data;
    if (local_N == 0) {
        receive_buffer = &dummy;
    }

    MPI_Scatterv(send_buffer, sendcounts, displacements, MPI_DOUBLE,
                 receive_buffer, local_N * D, MPI_DOUBLE, 0, comm);

    if (rank == 0) {
        free(sendcounts);
        free(displacements);
    }
}

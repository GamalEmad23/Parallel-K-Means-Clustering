#include "workload.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static void fail(MPI_Comm comm, const char *message)
{
    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    fprintf(stderr, "workload: rank %d: %s\n", rank, message);
    MPI_Abort(comm, EXIT_FAILURE);
}

void calculate_distribution(int N, int D, int world_size,
                            int *sendcounts, int *displacements)
{
    if (N < 0 || D <= 0 || world_size <= 0 ||
        sendcounts == NULL || displacements == NULL) {
        fprintf(stderr, "workload: invalid arguments to calculate_distribution\n");
        abort();
    }
    if (N > 0 && D > INT_MAX / N) {
        fprintf(stderr, "Error: N * D exceeds MPI limit\n");
        abort();
    }

    const int base = N / world_size;
    const int remainder = N % world_size;
    int point_offset = 0;

    for (int process = 0; process < world_size; ++process) {
        const int local_n = base + (process < remainder ? 1 : 0);
        sendcounts[process] = local_n * D;
        displacements[process] = point_offset * D;
        point_offset += local_n;
    }
}

void distribute_data(const double *all_data, double *local_data,
                     int N, int D, int rank, int world_size, MPI_Comm comm)
{
    int actual_rank = -1;
    int actual_world_size = 0;
    MPI_Comm_rank(comm, &actual_rank);
    MPI_Comm_size(comm, &actual_world_size);

    if (N < 0 || D <= 0 || world_size <= 0 ||
        actual_rank != rank || actual_world_size != world_size) {
        fail(comm, "invalid dimensions or rank/world_size do not match communicator");
    }
    if (N > 0 && D > INT_MAX / N) {
        fail(comm, "N * D exceeds MPI int-count limit");
    }
    if (rank == 0 && N > 0 && all_data == NULL) {
        fail(comm, "rank 0 must provide all_data when N is positive");
    }

    int *sendcounts = (int *)malloc((size_t)world_size * sizeof(int));
    int *displacements = (int *)malloc((size_t)world_size * sizeof(int));
    int *point_counts = (int *)malloc((size_t)world_size * sizeof(int));
    if (sendcounts == NULL || displacements == NULL || point_counts == NULL) {
        free(sendcounts);
        free(displacements);
        free(point_counts);
        fail(comm, "could not allocate distribution metadata");
    }
    calculate_distribution(N, D, world_size, sendcounts, displacements);
    for (int process = 0; process < world_size; ++process) {
        point_counts[process] = sendcounts[process] / D;
    }

    int local_n = 0;
    MPI_Scatter(point_counts, 1, MPI_INT, &local_n, 1, MPI_INT, 0, comm);
    if (local_n > 0 && local_data == NULL) {
        free(sendcounts);
        free(displacements);
        free(point_counts);
        fail(comm, "local_data must be allocated for this rank's points");
    }

    double dummy = 0.0;
    MPI_Scatterv(all_data != NULL ? all_data : &dummy,
                 sendcounts, displacements, MPI_DOUBLE,
                 local_data != NULL ? local_data : &dummy,
                 local_n * D, MPI_DOUBLE, 0, comm);

    /* Confirm the per-rank counts cover exactly N points. */
    int *observed_counts = (int *)malloc((size_t)world_size * sizeof(int));
    if (observed_counts == NULL) {
        free(sendcounts);
        free(displacements);
        free(point_counts);
        fail(comm, "could not allocate workload validation metadata");
    }
    MPI_Allgather(&local_n, 1, MPI_INT, observed_counts, 1, MPI_INT, comm);
    int total_points = 0;
    for (int process = 0; process < world_size; ++process) {
        if (observed_counts[process] != point_counts[process]) {
            free(observed_counts);
            free(sendcounts);
            free(displacements);
            free(point_counts);
            fail(comm, "MPI_Allgather workload validation failed");
        }
        total_points += observed_counts[process];
    }
    free(observed_counts);
    free(sendcounts);
    free(displacements);
    free(point_counts);

    if (total_points != N) {
        fail(comm, "distributed point counts do not sum to N");
    }
}

#ifndef WORKLOAD_H
#define WORKLOAD_H

#include <mpi.h>

/* Calculate element counts and displacements for MPI_Scatterv. */
void calculate_distribution(int N, int D, int world_size,
                            int *sendcounts, int *displacements);

/*
 * Scatter row-major N-by-D point data from rank 0. Each rank must provide
 * local_data with room for local_N * D doubles, where
 * local_N = N / world_size + (rank < N % world_size ? 1 : 0).
 */
void distribute_data(const double *all_data, double *local_data,
                     int N, int D, int rank, int world_size, MPI_Comm comm);

#endif

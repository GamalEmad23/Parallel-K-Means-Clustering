#ifndef WORKLOAD_H
#define WORKLOAD_H

#include <mpi.h>

/* Counts and displacements are measured in double elements. */
void calculate_distribution(int N, int D, int world_size,
                            int *sendcounts, int *displacements);

/* Rank 0 supplies all_data. Each rank allocates its local_data buffer. */
void distribute_data(const double *all_data, double *local_data,
                     int N, int D, int rank, int world_size, MPI_Comm comm);

#endif

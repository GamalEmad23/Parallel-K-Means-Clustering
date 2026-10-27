#ifndef COORDINATOR_H
#define COORDINATOR_H

#include <mpi.h>

/*
 * Member 4: centroid update and convergence check.
 *
 * COLLECTIVE: every rank of `comm` must call this function in the same
 * iteration, with the same K, D and tolerance. Ranks with zero local points
 * must still call it (their local_sums are all 0.0 and local_counts all 0,
 * exactly what assign_clusters() produces for local_N == 0).
 *
 * Inputs (not modified):
 *   local_sums    K * D doubles, row-major: local_sums[c * D + d]
 *   local_counts  K ints:                   local_counts[c]
 * In/out:
 *   centroids     K * D doubles, row-major. Holds the old centroids on entry
 *                 and the updated centroids on a successful return. Empty
 *                 clusters (global count 0) keep their old centroid. It is
 *                 left untouched when -1 is returned.
 *   tolerance     finite and >= 0.
 *
 * Return value (identical on all ranks):
 *    1  converged:      max centroid shift <= tolerance
 *    0  not converged:  run another iteration (unless max_iter is reached)
 *   -1  error:          bad argument or allocation failure on ANY rank;
 *                       reported on every rank, centroids unchanged.
 *
 * The caller owns all buffers; this function allocates and frees only
 * temporary scratch memory.
 */
int update_centroids(const double *local_sums,
                     const int *local_counts,
                     double *centroids,
                     int K,
                     int D,
                     double tolerance,
                     MPI_Comm comm);

#endif

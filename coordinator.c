#include "coordinator.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* Checks only what a rank can verify on its own. */
static int arguments_ok(const double *local_sums, const int *local_counts,
                        const double *centroids, int K, int D,
                        double tolerance)
{
    if (local_sums == NULL || local_counts == NULL || centroids == NULL)
        return 0;
    if (K <= 0 || D <= 0 || K > INT_MAX / D)   /* K * D must fit in int: MPI counts are int */
        return 0;
    if (!isfinite(tolerance) || tolerance < 0.0)
        return 0;
    for (int c = 0; c < K; ++c)
        if (local_counts[c] < 0)
            return 0;
    return 1;
}

int update_centroids(const double *local_sums,
                     const int *local_counts,
                     double *centroids,
                     int K,
                     int D,
                     double tolerance,
                     MPI_Comm comm)
{
    double *global_sums = NULL;
    int *global_counts = NULL;
    int ok = arguments_ok(local_sums, local_counts, centroids, K, D, tolerance);

    /* Scratch allocation, with size_t overflow checks (K * D <= INT_MAX here). */
    if (ok) {
        size_t kd = (size_t)K * (size_t)D;
        if (kd > SIZE_MAX / sizeof(double) || (size_t)K > SIZE_MAX / sizeof(int)) {
            ok = 0;
        } else {
            global_sums = (double *)malloc(kd * sizeof(double));
            global_counts = (int *)malloc((size_t)K * sizeof(int));
            if (global_sums == NULL || global_counts == NULL)
                ok = 0;
        }
    }

    /*
     * Agreement step: EVERY rank reaches this collective, even a rank whose
     * arguments are bad. If any rank is bad, all ranks return -1 here, before
     * the data reductions, so nobody blocks in a collective others skipped.
     */
    int all_ok = 0;
    if (MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm) != MPI_SUCCESS)
        all_ok = 0;
    if (!all_ok) {
        free(global_sums);
        free(global_counts);
        return -1;
    }

    /* Aggregate: all ranks receive identical global sums and counts. */
    int rc_sums = MPI_Allreduce(local_sums, global_sums, K * D,
                                MPI_DOUBLE, MPI_SUM, comm);
    int rc_counts = MPI_Allreduce(local_counts, global_counts, K,
                                  MPI_INT, MPI_SUM, comm);
    if (rc_sums != MPI_SUCCESS || rc_counts != MPI_SUCCESS) {
        free(global_sums);
        free(global_counts);
        return -1;
    }

    /* Update centroids; track the largest squared shift (no sqrt needed). */
    double max_shift_sq = 0.0;
    for (int c = 0; c < K; ++c) {
        if (global_counts[c] <= 0)
            continue;                       /* globally empty: keep old centroid */
        double shift_sq = 0.0;
        for (int d = 0; d < D; ++d) {
            double updated = global_sums[c * D + d] / (double)global_counts[c];
            double diff = updated - centroids[c * D + d];
            shift_sq += diff * diff;
            centroids[c * D + d] = updated;
        }
        if (!(shift_sq <= max_shift_sq))     /* also propagates NaN => not converged */
            max_shift_sq = shift_sq;
    }

    free(global_sums);
    free(global_counts);

    /* max_shift <= tolerance  <=>  max_shift^2 <= tolerance^2 (both >= 0). */
    return (max_shift_sq <= tolerance * tolerance) ? 1 : 0;
}

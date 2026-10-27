/*
 * Tests for update_centroids(). Run with any process count >= 1:
 *   mpirun -np 1 ./test_coordinator ... mpirun -np 5 ./test_coordinator
 *
 * Every rank runs every test and calls the same collectives in the same order.
 * Failures are counted, never exit()ed, so no rank leaves others blocked.
 */
#include "coordinator.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_rank, g_size;
static int g_failures = 0;
static int g_checks = 0;

static void check(int cond, const char *name)
{
    ++g_checks;
    if (!cond) {
        ++g_failures;
        fprintf(stderr, "FAIL (rank %d): %s\n", g_rank, name);
    }
}

/*
 * Split a known integer-valued global statistic across ranks. Modes:
 *   0: every rank holds ~1/P of every count and sum (remainder on rank 0)
 *   1: only the LAST rank holds data; all other ranks have zero local points
 * All values are small integers, so double sums are exact in any order.
 */
static void split_stats(const double *gs, const int *gc, int K, int D, int mode,
                        double *ls, int *lc)
{
    for (int c = 0; c < K; ++c) {
        if (mode == 1) {
            lc[c] = (g_rank == g_size - 1) ? gc[c] : 0;
            for (int d = 0; d < D; ++d)
                ls[c * D + d] = (g_rank == g_size - 1) ? gs[c * D + d] : 0.0;
        } else {
            lc[c] = gc[c] / g_size + (g_rank == 0 ? gc[c] % g_size : 0);
            for (int d = 0; d < D; ++d) {
                double v = gs[c * D + d];
                double base = floor(v / g_size);
                ls[c * D + d] = base + (g_rank == 0 ? v - base * g_size : 0.0);
            }
        }
    }
}

/* All ranks must hold bit-identical centroids (compared against rank 0's). */
static void check_same_on_all_ranks(const double *cent, int n, int ret, const char *name)
{
    double *ref = malloc((size_t)n * sizeof(double));
    int ref_ret = ret;
    memcpy(ref, cent, (size_t)n * sizeof(double));
    MPI_Bcast(ref, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ref_ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    check(memcmp(ref, cent, (size_t)n * sizeof(double)) == 0 && ref_ret == ret, name);
    free(ref);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    double ls[32];
    int lc[8], r;

    /* Test 1+5+10: K=3, D=2, known sums/counts, exact values, row-major layout.
     * cluster 0: sum (6,12) count 3 -> (2,4)
     * cluster 1: sum (20,10) count 4 -> (5,2.5)
     * cluster 2: sum (9,-3) count 1 -> (9,-3)                                  */
    for (int mode = 0; mode < 2; ++mode) {
        const double gs[6] = {6, 12, 20, 10, 9, -3};
        const int gc[3] = {3, 4, 1};
        double cent[6] = {0, 0, 0, 0, 0, 0};
        split_stats(gs, gc, 3, 2, mode, ls, lc);
        r = update_centroids(ls, lc, cent, 3, 2, 1e-9, MPI_COMM_WORLD);
        check(r == 0, "update: not converged when centroids move a lot");
        check(cent[0] == 2.0 && cent[1] == 4.0, "update: cluster 0 value/layout");
        check(cent[2] == 5.0 && cent[3] == 2.5, "update: cluster 1 value/layout");
        check(cent[4] == 9.0 && cent[5] == -3.0, "update: cluster 2 value/layout");
        check_same_on_all_ranks(cent, 6, r, "update: identical on all ranks");
    }

    /* Test 2: converge when shift is below tolerance. Old (2,4),(5,2.5),(9,-3.0)
     * moved by 0.25 in one coordinate only. */
    {
        const double gs[6] = {6, 12, 20, 10, 9, -3};
        const int gc[3] = {3, 4, 1};
        double cent[6] = {2.25, 4, 5, 2.5, 9, -3};
        split_stats(gs, gc, 3, 2, 0, ls, lc);
        r = update_centroids(ls, lc, cent, 3, 2, 0.5, MPI_COMM_WORLD);
        check(r == 1, "converge: shift 0.25 < tol 0.5");
        check(cent[0] == 2.0, "converge: centroid still updated");
        check_same_on_all_ranks(cent, 6, r, "converge: identical on all ranks");
    }

    /* Test 3: non-convergence when the shift exceeds the tolerance. */
    {
        const double gs[2] = {4, 8};
        const int gc[1] = {2};              /* new centroid (2,4) */
        double cent[2] = {2.5, 4};          /* shift 0.5 */
        split_stats(gs, gc, 1, 2, 0, ls, lc);
        r = update_centroids(ls, lc, cent, 1, 2, 0.25, MPI_COMM_WORLD);
        check(r == 0, "no converge: shift 0.5 > tol 0.25");
    }

    /* Test 9: exact tolerance boundary (3-4-5 triangle) and zero tolerance. */
    {
        const double gs[2] = {3, 4};
        const int gc[1] = {1};              /* new centroid (3,4) */
        double cent[2] = {0, 0};            /* shift exactly 5 */
        split_stats(gs, gc, 1, 2, 0, ls, lc);
        r = update_centroids(ls, lc, cent, 1, 2, 5.0, MPI_COMM_WORLD);
        check(r == 1, "boundary: shift == tol converges");

        double cent2[2] = {0, 0};
        r = update_centroids(ls, lc, cent2, 1, 2, 4.999999, MPI_COMM_WORLD);
        check(r == 0, "boundary: shift just above tol does not converge");

        double cent3[2] = {3, 4};           /* zero shift, tol 0 */
        r = update_centroids(ls, lc, cent3, 1, 2, 0.0, MPI_COMM_WORLD);
        check(r == 1, "zero tol: zero shift converges");
        double cent4[2] = {3, 4.125};
        r = update_centroids(ls, lc, cent4, 1, 2, 0.0, MPI_COMM_WORLD);
        check(r == 0, "zero tol: any nonzero shift does not converge");
    }

    /* Test 4: empty cluster keeps its old centroid, no NaN/Inf, and it does not
     * count toward the shift. Cluster 1 is empty on every rank. */
    {
        const double gs[4] = {10, 10, 0, 0};
        const int gc[2] = {5, 0};
        double cent[4] = {2, 2, 7.5, -1.25};
        split_stats(gs, gc, 2, 2, 0, ls, lc);
        r = update_centroids(ls, lc, cent, 2, 2, 0.0, MPI_COMM_WORLD);
        check(cent[0] == 2.0 && cent[1] == 2.0, "empty: non-empty cluster updated");
        check(cent[2] == 7.5 && cent[3] == -1.25, "empty: old centroid preserved");
        check(r == 1, "empty: empty cluster does not block convergence");
        check_same_on_all_ranks(cent, 4, r, "empty: identical on all ranks");
    }

    /* Test 4b/7: a cluster is empty only GLOBALLY. Rank 0 alone has points of
     * cluster 1 (count 2); other ranks report 0 for it. Must still be updated. */
    {
        double sums[2] = {0, 0};
        int counts[1] = {0};
        if (g_rank == 0) { sums[0] = 8; sums[1] = 12; counts[0] = 2; }
        double cent[2] = {0, 0};
        r = update_centroids(sums, counts, cent, 1, 2, 1e-9, MPI_COMM_WORLD);
        check(cent[0] == 4.0 && cent[1] == 6.0, "global-empty: locally-empty ranks do not suppress update");
        check_same_on_all_ranks(cent, 2, r, "global-empty: identical on all ranks");
    }

    /* Test 7: all ranks have zero points for every cluster (N = 0 locally and
     * globally): centroids unchanged, converged. */
    {
        double sums[4] = {0, 0, 0, 0};
        int counts[2] = {0, 0};
        double cent[4] = {1, 2, 3, 4};
        r = update_centroids(sums, counts, cent, 2, 2, 0.0, MPI_COMM_WORLD);
        check(r == 1 && cent[0] == 1 && cent[1] == 2 && cent[2] == 3 && cent[3] == 4,
              "all-zero: centroids unchanged, converged");
    }

    /* Test 6/7: only last rank has data (all other ranks have zero local points). */
    {
        const double gs[3] = {9, 18, 27};   /* K=1, D=3, count 3 -> (3,6,9) */
        const int gc[1] = {3};
        double cent[3] = {3, 6, 9};
        split_stats(gs, gc, 1, 3, 1, ls, lc);
        r = update_centroids(ls, lc, cent, 1, 3, 0.0, MPI_COMM_WORLD);
        check(r == 1 && cent[0] == 3 && cent[1] == 6 && cent[2] == 9,
              "zero-local: only last rank has data");
        check_same_on_all_ranks(cent, 3, r, "zero-local: identical on all ranks");
    }

    /* Test 8: invalid input, same on all ranks. All ranks must return -1,
     * and centroids must remain untouched. */
    {
        double sums[2] = {2, 2};
        int counts[1] = {1};
        double cent[2] = {7, 8};
        check(update_centroids(sums, counts, cent, 0, 2, 0.1, MPI_COMM_WORLD) == -1, "invalid: K = 0");
        check(update_centroids(sums, counts, cent, 1, 0, 0.1, MPI_COMM_WORLD) == -1, "invalid: D = 0");
        check(update_centroids(sums, counts, cent, 1, 2, -0.1, MPI_COMM_WORLD) == -1, "invalid: negative tol");
        check(update_centroids(sums, counts, cent, 1, 2, NAN, MPI_COMM_WORLD) == -1, "invalid: NaN tol");
        check(update_centroids(sums, counts, cent, 1, 2, INFINITY, MPI_COMM_WORLD) == -1, "invalid: inf tol");
        check(update_centroids(NULL, counts, cent, 1, 2, 0.1, MPI_COMM_WORLD) == -1, "invalid: NULL sums");
        check(update_centroids(sums, NULL, cent, 1, 2, 0.1, MPI_COMM_WORLD) == -1, "invalid: NULL counts");
        check(update_centroids(sums, counts, NULL, 1, 2, 0.1, MPI_COMM_WORLD) == -1, "invalid: NULL centroids");
        int neg[1] = {-1};
        check(update_centroids(sums, neg, cent, 1, 2, 0.1, MPI_COMM_WORLD) == -1, "invalid: negative count");
        check(update_centroids(sums, counts, cent, 2000000000, 2, 0.1, MPI_COMM_WORLD) == -1,
              "invalid: K * D overflows int");
        check(cent[0] == 7 && cent[1] == 8, "invalid: centroids untouched");
    }

    /* Test 8b: only ONE rank has a bad argument. All ranks must still return -1
     * and none may hang (this is the deadlock the agreement step prevents). */
    {
        double sums[2] = {2, 2};
        int counts[1] = {1};
        double cent[2] = {7, 8};
        int bad_rank = g_size - 1;
        r = update_centroids(g_rank == bad_rank ? NULL : sums, counts, cent, 1, 2, 0.1, MPI_COMM_WORLD);
        check(r == -1, "one-bad-rank: every rank returns -1");
        check(cent[0] == 7 && cent[1] == 8, "one-bad-rank: centroids untouched");
        /* The communicator must still be usable afterwards. */
        const double gs[2] = {2, 2};
        const int gc[1] = {1};
        split_stats(gs, gc, 1, 2, 0, ls, lc);
        r = update_centroids(ls, lc, cent, 1, 2, 0.0, MPI_COMM_WORLD);
        check(cent[0] == 2.0 && cent[1] == 2.0, "one-bad-rank: later call still works");
    }

    /* Test 10: K=1, D=1 and D=4 layout (strided indexing). */
    {
        const double gs[8] = {1, 2, 3, 4, 10, 20, 30, 40};
        const int gc[2] = {1, 10};          /* (1,2,3,4) and (1,2,3,4) */
        double cent[8] = {0};
        split_stats(gs, gc, 2, 4, 0, ls, lc);
        update_centroids(ls, lc, cent, 2, 4, 0.0, MPI_COMM_WORLD);
        int idx_ok = 1;
        for (int d = 0; d < 4; ++d)
            if (cent[d] != d + 1.0 || cent[4 + d] != d + 1.0) idx_ok = 0;
        check(idx_ok, "layout: K=2, D=4 row-major indexing");
    }

    /* Combine results: every rank must agree that nothing failed anywhere. */
    int total_failures = 0;
    MPI_Allreduce(&g_failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) {
        if (total_failures == 0)
            printf("PASS: test_coordinator (%d checks per rank, %d processes)\n", g_checks, g_size);
        else
            printf("FAIL: test_coordinator (%d failed checks across %d processes)\n", total_failures, g_size);
    }
    MPI_Finalize();
    return total_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

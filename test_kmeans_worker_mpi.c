#include "kmeans_worker.h"
#include "workload.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int N = 3, D = 2, K = 2;
    const double data[] = {0.0, 0.0, 2.0, 2.0, 10.0, 10.0};
    const double centroids[] = {0.0, 0.0, 10.0, 10.0};
    const int expected_labels[] = {0, 0, 1};
    const int local_N = N / world_size + (rank < N % world_size ? 1 : 0);
    double local_data[6];
    int labels[3];
    double sums[4], total_sums[4];
    int counts[2], total_counts[2];

    distribute_data(rank == 0 ? data : NULL, local_N > 0 ? local_data : NULL,
                    N, D, rank, world_size, MPI_COMM_WORLD);
    assign_clusters(local_N > 0 ? local_data : NULL, local_N, centroids, K, D,
                    local_N > 0 ? labels : NULL, sums, counts);

    int local_ok = 1;
    const int start = rank * (N / world_size) + (rank < N % world_size ? rank : N % world_size);
    for (int point = 0; point < local_N; ++point) {
        if (labels[point] != expected_labels[start + point]) local_ok = 0;
    }
    if (local_N == 0) {
        for (int i = 0; i < K * D; ++i) {
            if (sums[i] != 0.0) local_ok = 0;
        }
        for (int i = 0; i < K; ++i) {
            if (counts[i] != 0) local_ok = 0;
        }
    }

    MPI_Allreduce(sums, total_sums, K * D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(counts, total_counts, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (total_sums[0] != 2.0 || total_sums[1] != 2.0 ||
        total_sums[2] != 10.0 || total_sums[3] != 10.0 ||
        total_counts[0] != 2 || total_counts[1] != 1) local_ok = 0;

    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%s: worker after distribution, N=%d, processes=%d.\n",
               global_ok ? "PASS" : "FAIL", N, world_size);
    }
    MPI_Finalize();
    return global_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

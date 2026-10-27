#include "workload.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int parse_int(const char *text, int *value)
{
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (text[0] == '\0' || end == text || *end != '\0' ||
        errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX) {
        return 0;
    }
    *value = (int)parsed;
    return 1;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = 103;
    int D = 3;
    int valid = argc <= 3;
    if (valid && argc > 1) valid = parse_int(argv[1], &N);
    if (valid && argc > 2) valid = parse_int(argv[2], &D);
    if (valid && (N < 0 || D <= 0)) valid = 0;
    if (valid && N > 0 && D > INT_MAX / N) valid = 0;
    if (valid && (size_t)N > SIZE_MAX / sizeof(double) / (size_t)D) valid = 0;
    if (!valid) {
        if (rank == 0) {
            fprintf(stderr, "Usage: test_workload [N >= 0] [D > 0]\n");
            fprintf(stderr, "N and D must be valid integers and N * D must fit MPI and size_t limits.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int local_n = N / world_size + (rank < N % world_size ? 1 : 0);
    double *all_data = rank == 0 && N > 0
        ? (double *)malloc((size_t)N * D * sizeof(double)) : NULL;
    double *local_data = local_n > 0
        ? (double *)malloc((size_t)local_n * D * sizeof(double)) : NULL;
    if ((rank == 0 && N > 0 && all_data == NULL) ||
        (local_n > 0 && local_data == NULL)) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        for (int point = 0; point < N; ++point)
            for (int dim = 0; dim < D; ++dim)
                all_data[point * D + dim] = point * 1000.0 + dim;
    }

    distribute_data(all_data, local_data, N, D, rank, world_size, MPI_COMM_WORLD);

    const int start = rank * (N / world_size) + (rank < N % world_size ? rank : N % world_size);
    int local_ok = 1;
    for (int point = 0; point < local_n; ++point) {
        for (int dim = 0; dim < D; ++dim) {
            const double expected = (start + point) * 1000.0 + dim;
            if (local_data[point * D + dim] != expected) local_ok = 0;
        }
    }
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    global_ok = (global_ok == world_size);

    if (rank == 0) {
        if (global_ok) {
            printf("PASS: N=%d points, D=%d, processes=%d; every point arrived once in order.\n",
                   N, D, world_size);
        } else {
            fprintf(stderr, "FAIL: received point data did not match the expected partition.\n");
        }
    }
    free(all_data);
    free(local_data);
    MPI_Finalize();
    return global_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

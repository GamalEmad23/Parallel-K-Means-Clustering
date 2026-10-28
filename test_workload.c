#include "workload.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Convert the whole argument to int, rejecting text and out-of-range values. */
static int read_int(const char *text, int *result)
{
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);

    if (end == text || *end != '\0' || errno == ERANGE ||
        value < INT_MIN || value > INT_MAX) {
        return 0;
    }

    *result = (int)value;
    return 1;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int settings[3] = {103, 3, 1};

    if (rank == 0) {
        if (argc > 3) {
            fprintf(stderr, "Usage: test_workload [N >= 0] [D > 0]\n");
            settings[2] = 0;
        }

        if (settings[2] && argc > 1) {
            if (!read_int(argv[1], &settings[0])) {
                fprintf(stderr, "Invalid N: expected an integer within int range.\n");
                settings[2] = 0;
            }
        }

        if (settings[2] && argc > 2) {
            if (!read_int(argv[2], &settings[1])) {
                fprintf(stderr, "Invalid D: expected an integer within int range.\n");
                settings[2] = 0;
            }
        }

        int n = settings[0];
        int d = settings[1];

        if (settings[2] && (n < 0 || d <= 0)) {
            fprintf(stderr, "Invalid dimensions: N must be >= 0 and D must be > 0.\n");
            settings[2] = 0;
        }

        if (settings[2] && n > 0 && d > INT_MAX / n) {
            fprintf(stderr, "N * D exceeds the MPI int-count limit.\n");
            settings[2] = 0;
        }

        if (settings[2] &&
            (size_t)n > SIZE_MAX / sizeof(double) / (size_t)d) {
            fprintf(stderr, "N * D doubles exceed the allocation size limit.\n");
            settings[2] = 0;
        }
    }

    MPI_Bcast(settings, 3, MPI_INT, 0, MPI_COMM_WORLD);

    if (!settings[2]) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int N = settings[0];
    int D = settings[1];

    /* Start with an equal number of points for every process. */
    int local_N = N / world_size;
    /* Give one extra point to the first processes. */
    if (rank < N % world_size) {
        local_N++;
    }

    /* Process 0 stores the full dataset. */
    double *all_data = NULL;
    if (rank == 0 && N > 0) {
        all_data = malloc((size_t)N * D * sizeof(double));
    }

    /* Each process stores only its own points. */
    double *local_data = NULL;
    if (local_N > 0) {
        local_data = malloc((size_t)local_N * D * sizeof(double));
    }

    if (rank == 0 && N > 0 && all_data == NULL) {
        fprintf(stderr, "Rank %d: malloc failed for all_data (%zu bytes).\n",
                rank, (size_t)N * D * sizeof(double));
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (local_N > 0 && local_data == NULL) {
        fprintf(stderr, "Rank %d: malloc failed for local_data (%zu bytes).\n",
                rank, (size_t)local_N * D * sizeof(double));
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        for (int point = 0; point < N; point++) {
            for (int dim = 0; dim < D; dim++) {
                all_data[point * D + dim] = point * 1000.0 + dim;
            }
        }
    }

    distribute_data(all_data, local_data, N, D, rank,
                    world_size, MPI_COMM_WORLD);

    /* Find where this process starts in the full dataset. */
    int first_point = rank * (N / world_size);
    if (rank < N % world_size) {
        first_point += rank;
    } else {
        first_point += N % world_size;
    }

    int local_ok = 1;
    for (int point = 0; point < local_N; point++) {
        for (int dim = 0; dim < D; dim++) {
            double expected = (first_point + point) * 1000.0 + dim;
            if (local_data[point * D + dim] != expected) {
                local_ok = 0;
            }
        }
    }

    int passed_processes = 0;
    MPI_Allreduce(&local_ok, &passed_processes, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        if (passed_processes == world_size) {
            printf("PASS: each point was distributed once.\n");
        } else {
            printf("FAIL: the distributed data is incorrect.\n");
        }
    }

    free(all_data);
    free(local_data);
    MPI_Finalize();

    if (passed_processes == world_size) {
        return EXIT_SUCCESS;
    }

    return EXIT_FAILURE;
}

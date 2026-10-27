# Member 2: Workload Distribution Using MPI

This module implements Member 2's assigned task: distribute row-major `N x D` point data across MPI processes. The first `N % P` ranks receive one extra point, so the distribution also works when `N` is not divisible by the process count or there are more processes than points.

`calculate_distribution` fills `sendcounts` and `displacements` in **double elements**, as required by `MPI_Scatterv` with `MPI_DOUBLE`. `distribute_data` uses `MPI_Scatter` to give each process its point count, `MPI_Scatterv` to send the coordinates, and `MPI_Allgather` to check the per-process workload. `MPI_Scatterv` is used because the assignment specifically requires uneven data distribution.

The module checks that `N * D` fits the `int` counts used by `MPI_Scatterv`. The test checks command-line values with `strtol` and verifies allocation-size limits before allocating data. When a process has no points, the scatter call uses a one-value dummy buffer for its zero-count argument.

## Files

- `workload.h` — declarations shared with the integration `main`.
- `workload.c` — workload calculation and MPI distribution.
- `test_workload.c` — standalone correctness check.
- `CMakeLists.txt` — builds the project library and workload test.

## Calling the module

All processes call `distribute_data`. Rank 0 passes the full row-major dataset; other ranks pass `NULL` for `all_data`. Each process allocates `local_N * D` doubles for `local_data`, where:

```c
local_N = N / world_size + (rank < N % world_size ? 1 : 0);
```

Example integration:

```c
int local_N = N / world_size + (rank < N % world_size ? 1 : 0);
double *local_data = malloc((size_t)local_N * D * sizeof(double));
distribute_data(all_data, local_data, N, D, rank, world_size, MPI_COMM_WORLD);
```

On rank 0, `all_data` contains `N * D` doubles. The local array can be `NULL` when `local_N` is zero.

## Build and run in CLion on Windows

Open the repository root in CLion and reload the CMake project. Select a C toolchain that works with the installed Microsoft MPI SDK. The project uses `find_package(MPI COMPONENTS C)` to find and link MPI.

Run the built test from CLion's terminal (or Windows Terminal). These cases cover uneven division, even division, more processes than points, zero points, and a single process:

```powershell
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -n 5 .\cmake-build-debug\test_workload.exe 103 3
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -n 4 .\cmake-build-debug\test_workload.exe 100 2
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -n 5 .\cmake-build-debug\test_workload.exe 3 2
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -n 5 .\cmake-build-debug\test_workload.exe 0 3
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -n 1 .\cmake-build-debug\test_workload.exe 103 3
```

Adjust `cmake-build-debug` if CLion uses a different build directory.

## Member 3: Distance and cluster assignment

`kmeans_worker.c` and `kmeans_worker.h` provide `assign_clusters`. Each process
calls it after `distribute_data`, using the current centroids and its local points.
The function compares squared Euclidean distances, stores the nearest cluster in
`labels`, and fills local coordinate sums and point counts. Equal distances keep
the lowest cluster index. Sums and counts are reset on every call, even when the
process has no points. The worker makes no MPI calls; the next module can combine
its results using MPI reductions.

Arrays use the same row-major layout as the distribution module. Provide space
for `local_N` labels, `K * D` sums and `K` counts, using separate input and output
buffers. `K` and `D` must be positive and `local_N` nonnegative; products used for
indexing must fit in `int`. Invalid dimensions or missing required pointers abort
with an error message. For zero local points, data, labels and centroids may be
`NULL`. Coordinates and squared distances must fit in finite `double` values.

The standalone test needs only a C compiler:

```sh
gcc -std=c11 -Wall -Wextra -Werror -pedantic kmeans_worker.c test_kmeans_worker.c -o test_kmeans_worker
./test_kmeans_worker
```

With MPI installed, build using CMake and run `ctest --test-dir <build-directory>
--output-on-failure`. This also runs `test_kmeans_worker_mpi` with five processes
and three points to check distribution, assignment and reduction with empty ranks.

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

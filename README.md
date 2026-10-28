# Member 2: Workload Distribution Using MPI

N is the number of points, D is the number of coordinates per point,
and world_size is the number of MPI processes.

## Distribution

1. Divide N by world_size to find the basic number of points per process.
2. Give one extra point to each of the first N % world_size processes.
3. Rank 0 calculates sendcounts and displacements in double elements.
4. MPI_Scatterv sends each process its coordinates.

For example, 103 points and 5 processes produce counts of
21, 21, 21, 20, 20 points. If D is 3, sendcounts contains
63, 63, 63, 60, 60 double elements.

Each caller calculates its local_N before allocating local_data:

```c
int local_N = N / world_size;
if (rank < N % world_size) {
    local_N++;
}
```

Rank 0 owns the full dataset. Each process owns its local buffer.
The data is stored point by point: x0, y0, x1, y1, and so on.
Processes with zero points use a dummy buffer for the zero-count call.

## Test program

Rank 0 reads N and D from the command line and sends the settings with
MPI_Bcast. The defaults are 103 points and 3 dimensions.
Memory is allocated at runtime. The number of processes comes from
MPI_Comm_size and is selected with mpiexec -n.

The test generates recognizable values, checks every received coordinate,
and combines the success flags with MPI_Allreduce and MPI_SUM.
The size checks prevent overflowing the MPI int counts and allocation size.
Input uses strtol and rejects empty arguments, trailing text, and values outside int range. Distribution checks rank and world_size against the supplied communicator. Allocation errors identify the rank, buffer, and requested bytes. All ranks must supply identical N and D and call the function collectively using a valid intracommunicator.

## Build and run

Open this folder in CLion and build the test_workload target.
CMake needs an installed MPI runtime and development SDK.

Run from the project folder after building:

```powershell
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -n 5 .\cmake-build-debug\test_workload.exe 103 3
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -n 4 .\cmake-build-debug\test_workload.exe 100 2
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -n 5 .\cmake-build-debug\test_workload.exe 3 2
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -n 5 .\cmake-build-debug\test_workload.exe 0 3
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -n 1 .\cmake-build-debug\test_workload.exe 103 3
```

Adjust the build directory if your CLion profile uses a different one.

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

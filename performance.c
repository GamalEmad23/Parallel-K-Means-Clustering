#include "performance.h"

double measure_parallel_time(double local_elapsed, MPI_Comm comm)
{
    double parallel_elapsed = 0.0;

    MPI_Reduce(
        &local_elapsed,       
        &parallel_elapsed,    
        1,                   
        MPI_DOUBLE,
        MPI_MAX,              
        0,                   
        comm
    );

    return parallel_elapsed;
}

double calculate_speedup(double sequential_time,
                         double parallel_time)
{
    if (sequential_time <= 0.0 || parallel_time <= 0.0)
        return -1.0;

    return sequential_time / parallel_time;
}


double calculate_efficiency(double speedup,
                            int process_count)
{
    if (speedup <= 0.0 || process_count <= 0)
        return -1.0;

    return (speedup / process_count) * 100.0;
}
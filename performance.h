#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <mpi.h>

double measure_parallel_time(double local_elapsed, MPI_Comm comm);

double calculate_speedup(double sequential_time,
                         double parallel_time);

double calculate_efficiency(double speedup,
                            int process_count);

#endif

cmake_minimum_required(VERSION 3.28)
project(ParallelK_MeansClustering C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS C)

add_library(ParallelK_MeansClustering STATIC
        Main.c
        data_manager.h
        data_manager.c
        workload.h
        workload.c
        kmeans_worker.h
        kmeans_worker.c)

target_link_libraries(ParallelK_MeansClustering PUBLIC MPI::MPI_C)

add_executable(test_workload
        test_workload.c
        workload.c
        workload.h)
target_link_libraries(test_workload PRIVATE MPI::MPI_C)

add_executable(test_kmeans_worker test_kmeans_worker.c kmeans_worker.c)

add_executable(test_kmeans_worker_mpi test_kmeans_worker_mpi.c)
target_link_libraries(test_kmeans_worker_mpi PRIVATE ParallelK_MeansClustering)

enable_testing()
add_test(NAME worker COMMAND test_kmeans_worker)
add_test(NAME worker_mpi_empty_ranks
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 5
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_kmeans_worker_mpi> ${MPIEXEC_POSTFLAGS})

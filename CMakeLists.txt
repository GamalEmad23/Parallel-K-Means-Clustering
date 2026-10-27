cmake_minimum_required(VERSION 3.28)

project(ParallelK_MeansClustering LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# MPI configuration
find_package(MPI REQUIRED COMPONENTS C)

# Shared library for data management
add_library(data_manager STATIC
data_manager.c
)

target_include_directories(data_manager PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
)

# Shared library for workload functionality
add_library(workload STATIC
workload.c
)

target_include_directories(workload PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(workload PUBLIC
MPI::MPI_C
)

# Shared library for K-Means worker functionality
add_library(kmeans_worker STATIC
kmeans_worker.c
)

target_include_directories(kmeans_worker PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
)


# Shared library for coordinator functionality
add_library(coordinator STATIC coordinator.c)

target_include_directories(coordinator PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(coordinator PUBLIC MPI::MPI_C)

# Shared functionality for Performance measurement
add_library(performance STATIC performance.c)
target_include_directories(performance PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(performance PUBLIC MPI::MPI_C)

# Main application
add_executable(ParallelK_MeansClustering
Main.c
)

target_link_libraries(ParallelK_MeansClustering PRIVATE
data_manager
workload
kmeans_worker
coordinator
performance
MPI::MPI_C
)

# Workload tests
add_executable(test_workload
test_workload.c
)

target_link_libraries(test_workload PRIVATE
workload
MPI::MPI_C
)

# K-Means worker tests
add_executable(test_kmeans_worker
test_kmeans_worker.c
)

target_link_libraries(test_kmeans_worker PRIVATE
kmeans_worker
)

# MPI K-Means worker tests
add_executable(test_kmeans_worker_mpi
test_kmeans_worker_mpi.c
)

target_link_libraries(test_kmeans_worker_mpi PRIVATE
kmeans_worker
workload
MPI::MPI_C
)

# Performance tests
add_executable(test_performance test_performance.c)

target_link_libraries(test_performance PRIVATE
    performance
    MPI::MPI_C
)



# Testing configuration
enable_testing()

add_test(
NAME workload
COMMAND test_workload
)

add_test(
NAME worker
COMMAND test_kmeans_worker
)

add_test(
NAME worker_mpi_empty_ranks
COMMAND ${MPIEXEC_EXECUTABLE}
${MPIEXEC_NUMPROC_FLAG} 5
${MPIEXEC_PREFLAGS}
$<TARGET_FILE:test_kmeans_worker_mpi>
${MPIEXEC_POSTFLAGS}
)

add_test(
 NAME performance_mpi
 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
 ${MPIEXEC_PREFLAGS}
 $<TARGET_FILE:test_performance>
 ${MPIEXEC_POSTFLAGS}
)
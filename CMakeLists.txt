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
        workload.c)

target_link_libraries(ParallelK_MeansClustering PUBLIC MPI::MPI_C)

add_executable(test_workload
        test_workload.c
        workload.c
        workload.h)
target_link_libraries(test_workload PRIVATE MPI::MPI_C)

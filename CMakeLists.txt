cmake_minimum_required(VERSION 3.28)

project(ParallelK_MeansClustering LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS C)

add_executable(ParallelK_MeansClustering
        Main.c
        data_manager.c
        data_manager.h
)

target_link_libraries(ParallelK_MeansClustering PRIVATE MPI::MPI_C)
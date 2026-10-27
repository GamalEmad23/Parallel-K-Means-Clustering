cmake_minimum_required(VERSION 3.28)
project(ParallelK_MeansClustering C)

set(CMAKE_C_STANDARD 11)

add_library(ParallelK_MeansClustering STATIC
        Main.c
        data_manager.h
        data_manager.c)

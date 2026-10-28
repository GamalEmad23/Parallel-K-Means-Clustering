# CMake generated Testfile for 
# Source directory: E:/Parallel Project/Parallel-K-Means-Clustering
# Build directory: E:/Parallel Project/Parallel-K-Means-Clustering/build-member5
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[workload]=] "E:/Parallel Project/Parallel-K-Means-Clustering/build-member5/test_workload.exe")
set_tests_properties([=[workload]=] PROPERTIES  _BACKTRACE_TRIPLES "E:/Parallel Project/Parallel-K-Means-Clustering/CMakeLists.txt;115;add_test;E:/Parallel Project/Parallel-K-Means-Clustering/CMakeLists.txt;0;")
add_test([=[worker]=] "E:/Parallel Project/Parallel-K-Means-Clustering/build-member5/test_kmeans_worker.exe")
set_tests_properties([=[worker]=] PROPERTIES  _BACKTRACE_TRIPLES "E:/Parallel Project/Parallel-K-Means-Clustering/CMakeLists.txt;120;add_test;E:/Parallel Project/Parallel-K-Means-Clustering/CMakeLists.txt;0;")
add_test([=[worker_mpi_empty_ranks]=] "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe" "-n" "5" "E:/Parallel Project/Parallel-K-Means-Clustering/build-member5/test_kmeans_worker_mpi.exe")
set_tests_properties([=[worker_mpi_empty_ranks]=] PROPERTIES  _BACKTRACE_TRIPLES "E:/Parallel Project/Parallel-K-Means-Clustering/CMakeLists.txt;125;add_test;E:/Parallel Project/Parallel-K-Means-Clustering/CMakeLists.txt;0;")
add_test([=[performance_mpi]=] "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe" "-n" "4" "E:/Parallel Project/Parallel-K-Means-Clustering/build-member5/test_performance.exe")
set_tests_properties([=[performance_mpi]=] PROPERTIES  _BACKTRACE_TRIPLES "E:/Parallel Project/Parallel-K-Means-Clustering/CMakeLists.txt;134;add_test;E:/Parallel Project/Parallel-K-Means-Clustering/CMakeLists.txt;0;")

#include <iostream>
#include <string>
#include <chrono>
#include <cstring>

#include "csvUtils.hpp"
#include "meanShift.hpp"
#include <mpi.h>

int main(const int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<Point> points;
    if (rank == 0) {
        points = getPointsFromCsv("data_csv.csv");
        std::cout << "Process " << rank << " began. Number of points : " << points.size() << "; Number of dimensions : " << points[0].dimensions() << std::endl;
    }

    float bandwidth = 2.7;

    
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    std::vector<Cluster> clusters = meanShift(points, bandwidth, rank, size);

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    float elapsedTime = std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    if (rank == 0) {
        std::cout << "0: Number of clusters: " << clusters.size() << std::endl;
        std::cout << "0: Elapsed time: " << elapsedTime << " s" << std::endl;
    }


    MPI_Finalize();
    return 0;
}

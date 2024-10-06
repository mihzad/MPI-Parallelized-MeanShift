#include <vector>
#include <iostream>
#include <cmath>
#include "Point.hpp"
#include "Cluster.hpp"
#include "ClustersBuilder.hpp"
#include "meanShift.hpp"
#include <mpi.h>

std::vector<Cluster> meanShift(const std::vector<Point> &points, float bandwidth, int rank, int proc_count) {
    std::cout << "Process " << rank << " entered Meanshift" << std::endl;
    
    // distributing dots between processes
    //local info:
    int total_points_count = -1;
    int local_points_count = -1;
    int start = -1;
    int dimensions = -1;
    std::vector<Point> got_points;
    if (rank == 0)
    {
        local_points_count = points.size() / proc_count + 1;
        start = local_points_count * rank;
        dimensions = (int)points[0].dimensions();
        total_points_count = points.size();

        for (int rank = 1; rank < proc_count; rank++)
        {
            int res[2] = { total_points_count, dimensions };
            MPI_Send(res, 2, MPI_INT, rank, 0, MPI_COMM_WORLD);
            std::vector<float> flattened_data;
            for(int i = 0; i < total_points_count; i++)
                flattened_data.insert(flattened_data.end(), points[i].values.begin(), points[i].values.end());
            MPI_Send(flattened_data.data(), total_points_count*dimensions, MPI_FLOAT, rank, 1, MPI_COMM_WORLD);
        }
        
        got_points = points;
    }
    else
    {
        int res[2];
        MPI_Recv(&res, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        total_points_count = res[0];
        dimensions = res[1];
        local_points_count = total_points_count / proc_count + 1;
        start = local_points_count * rank;

        float* flattened_data = new float[total_points_count * dimensions];
        MPI_Recv(flattened_data, total_points_count*dimensions, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < total_points_count; ++i) {
            std::vector<float> vals;
            for (int j = 0; j < dimensions; j++)
                vals.push_back(flattened_data[i * dimensions + j]);

            got_points.push_back(Point(vals));
        }
        delete[] flattened_data;
    }

    ClustersBuilder builder = ClustersBuilder(got_points, start, local_points_count, 0.4);

    std::cout << "Process " << rank << " built the clusterbuilder" << std::endl;
    std::cout << "Process " << rank << " start = " << start << "; builder size : " << builder.originalPoints.size() << std::endl;

    long iterations = 0;
    float radius = bandwidth * 3;
    float doubledSquaredBandwidth = 2 * bandwidth * bandwidth;
    while (!builder.allPointsHaveStoppedShifting() && iterations < MAX_ITERATIONS) {

        std::cout << "Process " << rank << " is building... iteration " << iterations << ";" << std::endl;
        //parralelize for using MPI
        for (long i = 0; i < builder.originalPoints.size(); ++i) {
            if (builder.hasStoppedShifting(i))
                continue;

            Point newPosition(dimensions);
            Point pointToShift = builder.getShiftedPoint(i);
            float totalWeight = 0.0;
            for (auto &point : got_points) {
                float distance = pointToShift.euclideanDistance(point);
                if (distance <= radius) {
                    float gaussian = std::exp(-(distance * distance) / doubledSquaredBandwidth);
                    newPosition += point * gaussian;
                    totalWeight += gaussian;
                }
            }

            // the new position of the point is the weighted average of its neighbors
            newPosition /= totalWeight;
            builder.shiftPoint(i, newPosition);
        }
        ++iterations;
    }

    //============merging data===========
    std::cout << "Process " << rank << " went to merging phase; " << std::endl;

    std::vector<float> flattened_data;
    for (const auto& point : builder.shiftedPoints) {
        flattened_data.insert(flattened_data.end(), point.values.begin(), point.values.end());
    }

    float* received_flattened_data = nullptr;
    if (rank == 0) {
        received_flattened_data = new float[local_points_count * proc_count * dimensions];
        std::cout << "Process 0 allocated memory for gathering;" << std::endl;
        std::cout << "Process 0 size: " << total_points_count * dimensions <<  std::endl;
    }

    std::cout << "Process " << rank << "is waiting at the barrier." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "barrier passed. performing gathering..." << std::endl;
    MPI_Gather(flattened_data.data(), local_points_count*dimensions, MPI_FLOAT, received_flattened_data, local_points_count*dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "0: successfully received data. Processing it..." << std::endl;

        std::vector<Point> shiftedPoints;
        for (int i = 0; i < total_points_count; ++i) {
            std::vector<float> vals;
            //std::cout << "stage " << i << std::endl;
            for (int j = 0; j < dimensions; j++) {
                vals.push_back(received_flattened_data[i * dimensions + j]);
                //std::cout << "  substage " << j << std::endl;
            }
            //std::cout << "  stageend " << std::endl;
            shiftedPoints.push_back(Point(vals));
            //std::cout << "0: point " << i << " was successfully got." << std::endl;
        }
        delete[] received_flattened_data;
        std::cout << "0: points size = " << points.size() << "; shifted = " << shiftedPoints.size() << std::endl;
        builder = ClustersBuilder(points, shiftedPoints, 0.4);
        std::cout << "0: Final builder created. Forming clusters..." << std::endl;
        return builder.buildClusters();
    }
    else
        return std::vector<Cluster>();


}


#include <vector>
#include <algorithm>

#include "Point.hpp"
#include "Cluster.hpp"
#include "ClustersBuilder.hpp"

ClustersBuilder::ClustersBuilder(const std::vector<Point> &originalPoints, int start, int count, float clusterEps) {
    //builder manages chunk of points: [start, end), where end = start+count
    int n = start+count < originalPoints.size() ? count : originalPoints.size()-start;
    this->originalPoints = std::vector<Point>(n);
    for (int i = 0; i < n; i++)
        this->originalPoints[i] = originalPoints[start + i];

    shiftedPoints = this->originalPoints;
    // vector of booleans such that the element in position i is false if the i-th point
    // has stopped to shift

    shifting = std::vector<bool>(n, true);
    this->clusterEps = clusterEps;
    this->shiftingEps = clusterEps / 10;
}

ClustersBuilder::ClustersBuilder(const std::vector<Point>& originalPoints, const std::vector<Point>& shiftedPoints, float clusterEps) {
    //constructor for final builder state. is used for merging builders only.
    this->originalPoints = originalPoints;

    this->shiftedPoints = shiftedPoints;
    // vector of booleans such that the element in position i is false if the i-th point
    // has stopped to shift
    shifting = std::vector<bool>(originalPoints.size(), false);
    this->clusterEps = clusterEps;
    this->shiftingEps = clusterEps / 10;
}


Point &ClustersBuilder::getShiftedPoint(long index) {
    return shiftedPoints[index];
}


void ClustersBuilder::shiftPoint(const long index, const Point &newPosition) {
    if (newPosition.euclideanDistance(shiftedPoints[index]) <= shiftingEps)
        shifting[index] = false;
    else
        shiftedPoints[index] = newPosition;
}


bool ClustersBuilder::hasStoppedShifting(long index) {
    return !shifting[index];
}


bool ClustersBuilder::allPointsHaveStoppedShifting() {
    return std::none_of(shifting.begin(), shifting.end(), [](bool v) { return v; });
}


std::vector<Point>::iterator ClustersBuilder::begin() {
    return shiftedPoints.begin();
}


std::vector<Point>::iterator ClustersBuilder::end() {
    return shiftedPoints.end();
}


std::vector<Cluster> ClustersBuilder::buildClusters() {
    std::vector<Cluster> clusters;

    // shifted points with distance minor or equal than clusterEps will go in the same cluster
    for (int i = 0; i < shiftedPoints.size(); ++i) {
        Point shiftedPoint = shiftedPoints[i];
        auto it = clusters.begin();
        auto itEnd = clusters.end();
        while (it != itEnd) {
            if (it->getCentroid().euclideanDistance(shiftedPoint) <= clusterEps) {
                // the point belongs to a cluster already created
                it->addPoint(originalPoints[i]);
                break;
            }
            ++it;
        }
        if (it == itEnd) {
            // create a new cluster
            Cluster cluster(shiftedPoint);
            cluster.addPoint(originalPoints[i]);
            clusters.emplace_back(cluster);
        }
    }
    return clusters;
}
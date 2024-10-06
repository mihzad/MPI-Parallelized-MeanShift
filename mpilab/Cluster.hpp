#ifndef MEANSHIFT_CLUSTER_HPP
#define MEANSHIFT_CLUSTER_HPP

#include <vector>

#include "Point.hpp"


class Cluster {
public:
    explicit Cluster(Point centroid);

    Point getCentroid() const;

    void addPoint(Point point);

    long getSize() const;

    std::vector<Point>::iterator begin();

    std::vector<Point>::iterator end();

    float getSse() const;

private:
    std::vector<Point> points;
    Point centroid;
};


#endif //MEANSHIFT_CLUSTER_HPP

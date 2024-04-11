#pragma once

#include <models/point.h>
#include <common-includes/cgal.h>

#include <vector>

class Triangulation
{
public:
    Triangulation(const std::vector<Point>& points);
    ~Triangulation();

    inline const DT2& getTriangulation() const { return m_dt; }
    inline const std::vector<Point>& getPoints() const { return m_points; }

private:
    void triangulate();

private:
    std::vector<Point> m_points{};
    DT2 m_dt{};
};

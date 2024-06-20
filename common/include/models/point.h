#pragma once

struct Point
{
    double x{};
    double y{};
    double z{};
    Point(){}
    Point(double xVal, double yVal, double zVal) : x(xVal), y(yVal), z(zVal) {}
};

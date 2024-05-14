#pragma once

#include <models/layer.h>

#include <common-includes/cgal.h>

struct Mesh
{
    CDT2 dt{};
    Polyhedron surfaceMesh{};
    Polyhedron test{};
    std::vector<Polyhedron> layerBody{};
    Layer layer{};
};

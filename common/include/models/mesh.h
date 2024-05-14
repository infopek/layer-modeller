#pragma once

#include <models/layer.h>

#include <common-includes/cgal.h>

struct Mesh
{
    CDT2 dt{};
    SurfaceMesh surfaceMesh{};
    SurfaceMesh test{};
    std::vector<SurfaceMesh> layerBody{};
    Layer layer{};
};

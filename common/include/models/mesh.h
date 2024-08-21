#pragma once

#include <models/layer.h>

#include <common-includes/cgal.h>

struct Mesh
{
    SurfaceMesh surfaceMesh{};
    SurfaceMesh layerBody{};
    Layer layer{};
};

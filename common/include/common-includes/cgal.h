#pragma once

// CGAL headers
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>

#include <CGAL/property_map.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>

#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/extrude.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Triangulation_incremental_builder_3.h>

#include <CGAL/convex_hull_3.h>

#include <CGAL/Surface_mesh.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Implicit_surface_3.h>

// Types
using K = CGAL::Exact_predicates_inexact_constructions_kernel;

using Point2 = K::Point_2;
using Point3 = K::Point_3;
using Vector3 = K::Vector_3;

using CDT2 = CGAL::Constrained_Delaunay_triangulation_2<K>;
using EdgeIterator = CDT2::Edge_iterator;
using FaceIterator = CDT2::Finite_faces_iterator;
using VertexHandle = CDT2::Vertex_handle;
using FaceHandle = CDT2::Face_handle;

using DT2 = CGAL::Delaunay_triangulation_2<K>;
using DT3 = CGAL::Delaunay_triangulation_3<K>;

using Polyhedron = CGAL::Polyhedron_3<K>;
using Sphere3 = K::Sphere_3;
using SurfaceMesh = CGAL::Surface_mesh<Point3>;
using Plane3 = K::Plane_3;

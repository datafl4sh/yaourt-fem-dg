#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>

#include <blaze/Math.h>

#include "point.hpp"

namespace dg2d {

namespace priv {

template<typename T>
void sort_uniq(std::vector<T>& vec)
{
    std::sort(vec.begin(), vec.end());
    vec.erase( std::unique(vec.begin(), vec.end()), vec.end() );
}

} //namespace priv

struct edge
{
    edge(){}

    edge(size_t ap0, size_t ap1, bool bnd = false)
    {
        assert (ap0 != ap1);
        p0                  = ap0;
        p1                  = ap1;
        if (p0 > p1)
            std::swap(p0, p1);

        is_boundary         = bnd;
        is_broken           = false;
    }

    edge(size_t ap0, size_t ap1, size_t bid, bool bnd)
    {
        assert (ap0 != ap1);
        p0                  = ap0;
        p1                  = ap1;
        if (p0 > p1)
            std::swap(p0, p1);

        is_boundary         = bnd;
        is_broken           = false;
        boundary_id         = bid;
    }

    friend bool operator<(const edge& a, const edge& b) {
        assert(a.p0 < a.p1);
        assert(b.p0 < b.p1);
        return ( a.p0 < b.p0 or (a.p0 == b.p0 and a.p1 < b.p1) );
    }

    friend bool operator==(const edge& a, const edge& b) {
        return ( a.p0 == b.p0 and a.p1 == b.p1 );
    }

    friend std::ostream& operator<<(std::ostream& os, const edge& e) {
        os << "Edge: " << e.p0 << " " << e.p1;
        if (e.is_broken) os << ", broken";
        if (e.is_boundary) os << ", boundary";
        return os;
    }

    size_t  p0, p1, pb;
    bool    is_boundary, is_broken;
    size_t  boundary_id;

    auto point_ids() const { return std::array<size_t,2>({p0, p1}); }
};


struct triangle
{
    triangle() {}

    triangle(size_t ap0, size_t ap1, size_t ap2)
        : p{ap0, ap1, ap2}
    {
        //std::sort(p.begin(), p.end());
    }

    std::array<size_t, 3> p;

    friend bool operator<(const triangle& a, const triangle& b) {
        return std::lexicographical_compare( a.p.begin(), a.p.end(),
                                             b.p.begin(), b.p.end() );
    }

    friend std::ostream& operator<<(std::ostream& os, const triangle& t) {
        os << "Triangle: " << t.p[0] << " " << t.p[1] << " " << t.p[2];
        return os;
    }

    auto point_ids() const { return p; }
};

struct quadrangle
{
    quadrangle() {}

    quadrangle(size_t ap0, size_t ap1, size_t ap2, size_t ap3)
        : p{ap0, ap1, ap2, ap3}
    {
        //std::sort(p.begin(), p.end());
    }

    std::array<size_t, 4> p;

    friend bool operator<(const quadrangle& a, const quadrangle& b) {
        return std::lexicographical_compare( a.p.begin(), a.p.end(),
                                             b.p.begin(), b.p.end() );
    }

    friend std::ostream& operator<<(std::ostream& os, const quadrangle& t) {
        os  << "Quadrangle: " << t.p[0] << " " << t.p[1] << " " << t.p[2];
        os  << " " << t.p[3];
        return os;
    }

    auto point_ids() const { return p; }
};

#define NO_OWNER (~0)

template<typename T, size_t DIM, typename CellT, typename FaceT>
class mesh;

template<typename T, typename CellT, typename FaceT>
class mesh<T, 2, CellT, FaceT>
{
public:
    using coordinate_type   = T;
    using face_type         = FaceT;
    using cell_type         = CellT;
    using point_type        = point<T,2>;

    std::vector<point_type>     points;
    std::vector<face_type>      faces;
    std::vector<cell_type>      cells;

    std::vector<std::array<size_t,2>> face_owners;

    mesh()
    {}

    void compute_connectivity()
    {
        face_owners.resize( faces.size() );

        for (auto& fo : face_owners)
        {
            fo[0] = NO_OWNER;
            fo[1] = NO_OWNER;
        }

        size_t cell_id = 0;
        for (auto& cl : cells)
        {
            auto fcids = face_ids(*this, cl);

            for (auto& fcid : fcids)
            {
                auto& fo = face_owners.at(fcid);
                if (fo[0] == NO_OWNER)
                    fo[0] = cell_id;
                else if (fo[1] == NO_OWNER)
                    fo[1] = cell_id;
                else
                    throw std::logic_error("BUG: a face has max 2 owners"); 
            }

            ++cell_id;
        }
    }
};

template<typename T, typename CellT, typename FaceT>
class mesh<T, 3, CellT, FaceT>
{
public:
    using coordinate_type   = T;
    using edge_type         = edge;
    using face_type         = FaceT;
    using cell_type         = CellT;
    using point_type        = point<T,3>;

    std::vector<point_type>     points;
    std::vector<edge_type>      edges;
    std::vector<face_type>      faces;
    std::vector<cell_type>      cells;

    std::vector<std::array<size_t,2>> face_owners;

    mesh()
    {}
};

template<template<typename, size_t, typename, typename> class Mesh,
         typename T, typename CellT, typename FaceT>
bool is_boundary(const Mesh<T, 2, CellT, FaceT>& msh,
                 const typename Mesh<T,2,CellT,FaceT>::face_type& f)
{
    return f.is_boundary;
}

template<typename T>
using simplicial_mesh = mesh<T, 2, triangle, edge>;

template<typename T>
using quad_mesh = mesh<T, 2, quadrangle, edge>;

template<typename Mesh>
std::pair<typename Mesh::cell_type, bool>
neighbour_via(const Mesh& msh,
              const typename Mesh::cell_type& cl,
              const typename Mesh::face_type& fc)
{
    if ( msh.face_owners.size() != msh.faces.size() )
        throw std::logic_error("No neighbour information.");

    auto cl_ofs = offset(msh, cl);
    auto fc_ofs = offset(msh, fc);

    auto fo = msh.face_owners.at( fc_ofs );

    if ( fo[0] != cl_ofs )
        std::swap(fo[0], fo[1]);

    assert(fo[0] == cl_ofs);

    if (fo[1] == NO_OWNER)
        return std::make_pair(msh.cells[0], false);

    return std::make_pair(msh.cells.at(fo[1]), true);
}

template<typename Mesh>
size_t
offset(const Mesh& msh, const typename Mesh::cell_type& cl)
{
    auto itor = std::lower_bound(msh.cells.begin(), msh.cells.end(), cl);
    if (itor == msh.cells.end())
        throw std::invalid_argument("Mesh cell not found");

    return std::distance(msh.cells.begin(), itor);
}


template<typename Mesh>
size_t
offset(const Mesh& msh, const typename Mesh::face_type& fc)
{
    auto itor = std::lower_bound(msh.faces.begin(), msh.faces.end(), fc);
    if (itor == msh.faces.end())
        throw std::invalid_argument("Mesh face not found");

    return std::distance(msh.faces.begin(), itor);
}

template<typename T>
std::array<typename simplicial_mesh<T>::face_type, 3>
faces(const simplicial_mesh<T>& msh, const typename simplicial_mesh<T>::cell_type& cell)
{
    using face_type = typename simplicial_mesh<T>::face_type;

    std::array<face_type, 3> ret;
    ret[0] = msh.faces.at( offset(msh, face_type(cell.p[0], cell.p[1])) );
    ret[1] = msh.faces.at( offset(msh, face_type(cell.p[1], cell.p[2])) );
    ret[2] = msh.faces.at( offset(msh, face_type(cell.p[0], cell.p[2])) );
    return ret;
}

template<typename T>
std::array<typename quad_mesh<T>::face_type, 4>
faces(const quad_mesh<T>& msh, const typename quad_mesh<T>::cell_type& cell)
{
    using face_type = typename quad_mesh<T>::face_type;

    std::array<face_type, 4> ret;
    ret[0] = msh.faces.at( offset(msh, face_type(cell.p[0], cell.p[1])) );
    ret[1] = msh.faces.at( offset(msh, face_type(cell.p[1], cell.p[2])) );
    ret[2] = msh.faces.at( offset(msh, face_type(cell.p[2], cell.p[3])) );
    ret[3] = msh.faces.at( offset(msh, face_type(cell.p[0], cell.p[3])) );
    return ret;
}


template<typename T>
std::array<size_t, 3>
face_ids(const simplicial_mesh<T>& msh, const typename simplicial_mesh<T>::cell_type& cell)
{
    using face_type = typename simplicial_mesh<T>::face_type;

    std::array<size_t, 3> ret;
    ret[0] = offset(msh, face_type(cell.p[0], cell.p[1]));
    ret[1] = offset(msh, face_type(cell.p[1], cell.p[2]));
    ret[2] = offset(msh, face_type(cell.p[0], cell.p[2]));
    return ret;
}

template<typename T>
std::array<size_t, 4>
face_ids(const quad_mesh<T>& msh, const typename quad_mesh<T>::cell_type& cell)
{
    using face_type = typename quad_mesh<T>::face_type;

    std::array<size_t, 4> ret;
    ret[0] = offset(msh, face_type(cell.p[0], cell.p[1]));
    ret[1] = offset(msh, face_type(cell.p[1], cell.p[2]));
    ret[2] = offset(msh, face_type(cell.p[2], cell.p[3]));
    ret[3] = offset(msh, face_type(cell.p[0], cell.p[3]));
    return ret;
}

template<typename T>
std::array<point<T,2>, 2>
points(const simplicial_mesh<T>& msh,
       const typename simplicial_mesh<T>::face_type& fc)
{
    std::array<point<T,2>, 2> ret;

    ret[0] = *std::next(msh.points.begin(), fc.p0);
    ret[1] = *std::next(msh.points.begin(), fc.p1);

    return ret;
}


template<typename T>
std::array<point<T,2>, 3>
points(const simplicial_mesh<T>& msh,
       const typename simplicial_mesh<T>::cell_type& cl)
{
    std::array<point<T,2>, 3> ret;
    ret[0] = msh.points.at(cl.p[0]);
    ret[1] = msh.points.at(cl.p[1]);
    ret[2] = msh.points.at(cl.p[2]);
    return ret;
}

template<typename T>
std::array<point<T,2>, 2>
points(const quad_mesh<T>& msh,
       const typename quad_mesh<T>::face_type& fc)
{
    std::array<point<T,2>, 2> ret;

    ret[0] = *std::next(msh.points.begin(), fc.p0);
    ret[1] = *std::next(msh.points.begin(), fc.p1);

    return ret;
}

template<typename T>
std::array<point<T,2>, 4>
points(const quad_mesh<T>& msh,
       const typename quad_mesh<T>::cell_type& cl)
{
    std::array<point<T,2>, 4> ret;
    ret[0] = msh.points.at(cl.p[0]);
    ret[1] = msh.points.at(cl.p[1]);
    ret[2] = msh.points.at(cl.p[2]);
    ret[3] = msh.points.at(cl.p[3]);

    return ret;
}

template<template<typename, size_t, typename, typename> class Mesh,
         typename T, typename CellT, typename FaceT>
typename Mesh<T,2,CellT,FaceT>::point_type
barycenter(const Mesh<T,2,CellT,FaceT>& msh,
           const FaceT& fc)
{
    auto pts = points(msh, fc);
    return (pts[0] + pts[1]) / 2.0;
}

template<typename T>
typename simplicial_mesh<T>::point_type
barycenter(const simplicial_mesh<T>& msh,
           const typename simplicial_mesh<T>::cell_type& cl)
{
    auto pts = points(msh, cl);
    return (pts[0] + pts[1] + pts[2]) / 3.0;
}


template<typename T>
typename quad_mesh<T>::point_type
barycenter(const quad_mesh<T>& msh,
           const typename quad_mesh<T>::cell_type& cl)
{
    auto pts = points(msh, cl);
    return (pts[0] + pts[1] + pts[2]+ pts[3]) / 4.0;
}

template<template<typename, size_t, typename, typename> class Mesh,
         typename T, typename CellT, typename FaceT>
T
measure(const Mesh<T,2,CellT,FaceT>& msh, const CellT& cl)
{
    auto pts = points(msh, cl);

    T acc = 0.0;
    for (size_t i = 1; i < pts.size() - 1; i++)
    {
        auto d0 = pts.at(i) - pts.at(0);
        auto d1 = pts.at(i+1) - pts.at(0);
        acc += std::abs(d0.x()*d1.y() - d1.x()*d0.y())/T(2);
    }

    return acc;
}

template<template<typename, size_t, typename, typename> class Mesh,
         typename T, typename CellT, typename FaceT>
T
measure(const Mesh<T,2,CellT,FaceT>& msh, const FaceT& fc)
{
    auto pts = points(msh, fc);
    assert(pts.size() == 2);
    return distance(pts[0], pts[1]);
}

template<typename Mesh, typename Element>
typename Mesh::coordinate_type
diameter(const Mesh& msh, const Element& elem)
{
    const auto pts = points(msh, elem);

    typename Mesh::coordinate_type diam = 0.;

    for (size_t i = 0; i < pts.size(); i++)
        for (size_t j = i+1; j < pts.size(); j++)
            diam = std::max( distance(pts[i], pts[j]), diam );

    return diam;
}

template<template<typename, size_t, typename, typename> class Mesh,
         typename T, typename CellT, typename FaceT>
blaze::StaticVector<T,2>
normal(const Mesh<T,2,CellT,FaceT>& msh, const CellT& cl, const FaceT& fc)
{
    auto pts = points(msh, fc);
    assert(pts.size() == 2);

    auto v = pts[1] - pts[0];

    blaze::StaticVector<T,2> n;
    n[0] = -v.y();
    n[1] = v.x();

    auto cell_bar = barycenter(msh, cl);
    auto face_bar = barycenter(msh, fc);
    auto ov_temp = face_bar - cell_bar;
    blaze::StaticVector<T,2> outward_vector;
    outward_vector[0] = ov_temp.x();
    outward_vector[1] = ov_temp.y();

    if ( dot(n,outward_vector) < T(0) )
        return normalize(-n);

    return normalize(n);
}


enum class boundary_condition {
    NONE,
    DIRICHLET,
    NEUMANN,
    ROBIN
};

} //namespace dg2d

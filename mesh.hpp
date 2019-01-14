#pragma once

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
};


struct triangle
{
    triangle() {}

    triangle(size_t ap0, size_t ap1, size_t ap2)
        : p{ap0, ap1, ap2}
    {
        std::sort(p.begin(), p.end());
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

template<typename T, typename CellT>
class mesh
{
public:
    using coordinate_type   = T;
    using face_type         = edge;
    using cell_type         = CellT;
    using point_type        = point<T,2>;

    std::vector<point_type>     points;
    std::vector<face_type>      faces;
    std::vector<cell_type>      cells;

    std::vector<std::array<size_t,2>> face_owners;

    mesh()
    {}
};

template<typename T>
using simplicial_mesh = mesh<T, triangle>;


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
typename simplicial_mesh<T>::point_type
barycenter(const simplicial_mesh<T>& msh,
           const typename simplicial_mesh<T>::cell_type& cl)
{
    auto pts = points(msh, cl);
    return (pts[0] + pts[1] + pts[2]) / 3.0;
}

enum class boundary_condition {
    NONE,
    DIRICHLET,
    NEUMANN,
    ROBIN
};

} //namespace dg2d

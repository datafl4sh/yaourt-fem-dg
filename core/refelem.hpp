/*
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Toy implememtation of Discontinuous Galerkin for teaching purposes
 *
 * Matteo Cicuttin (c) 2018, 2019, 2020
 */

#pragma once

#include <array>
#include "point.hpp"

namespace yaourt::refelem {

template<typename T, size_t N>
struct reference_element;

template<typename T>
struct reference_element<T, 2>
{
    using point_type = point<T,2>;
    std::array<point_type, 2>   points;

    reference_element()
        : points({{point_type(-1,0), point_type(1,0)}})
    {}

    /*
    reference_element(const point_type& p0, const point_type& p1)
        : points({ p0, p1})
    {}
    */
};

template<typename T>
using reference_edge = reference_element<T,2>;

template<typename T>
struct reference_element<T, 3>
{
    using point_type = point<T,2>;
    std::array<point_type, 3>   points;

    reference_element()
        : points({{point_type(0,0), point_type(1,0), point_type(0,1)}})
    {}
};

template<typename T>
using reference_triangle = reference_element<T,3>;

template<typename T>
struct transform
{
    blaze::StaticMatrix<T,2,2>  Tm;
    blaze::StaticVector<T,2>    Tv;
    T                           Tdet;

    blaze::StaticVector<T,2>
    operator()(const blaze::StaticVector<T,2>& v)
    {
        return Tm*v + Tv;
    }

    point<T,2>
    operator()(const point<T,2>& pt)
    {
        blaze::StaticVector<T,2>    vpt;
        vpt[0] = pt.x();
        vpt[1] = pt.y();

        auto pvpt = Tm*vpt + Tv;
        return point<T,2>(pvpt[0], pvpt[1]);
    }
};

template<typename T>
transform<T>
inverse(const transform<T>& t)
{
    struct transform<T> ret;
    ret.Tm      = inv(t.Tm);
    ret.Tv      = -ret.Tm*t.Tv;
    ret.Tdet    = 1./t.Tdet;
    return ret;
}

template<typename T>
transform<T>
make_ref2phys_transform(const simplicial_mesh<T>& msh,
                        const typename simplicial_mesh<T>::cell_type& cl,
                        const reference_triangle<T>& rt)
{
    transform<T>   Tret;

    auto pts = points(msh, cl);
    auto v1 = pts[1] - pts[0];
    auto v2 = pts[2] - pts[0];

    Tret.Tm(0,0) = v1.x();  Tret.Tm(0,1) = v2.x();
    Tret.Tm(1,0) = v1.y();  Tret.Tm(1,1) = v2.y();

    Tret.Tv[0] = pts[0].x();
    Tret.Tv[1] = pts[0].y();

    Tret.Tdet = det(Tret.Tm);

    return Tret;
}

template<typename T>
transform<T>
make_ref2phys_transform(const simplicial_mesh<T>& msh,
                        const typename simplicial_mesh<T>::face_type& fc,
                        const reference_edge<T>& re)
{
    transform<T>   Tret;

    auto pts = points(msh, fc);

    blaze::StaticMatrix<T,2,2>  Tt;
    blaze::StaticVector<T,2>    bt;

    auto v = pts[1] - pts[0];

    Tt(0,0) = v.x();    Tt(0,1) = -v.y();
    Tt(1,0) = v.y();    Tt(1,1) =  v.x(); 

    bt[0] = pts[0].x();
    bt[1] = pts[0].y();

    blaze::StaticMatrix<T,2,2>  iTh;
    blaze::StaticVector<T,2>    bh;

    auto vh = re.points[1] - re.points[0];

    iTh(0,0) =  vh.x();     iTh(0,1) = vh.y();
    iTh(1,0) = -vh.y();     iTh(1,1) = vh.x();

    iTh /= (vh.x()*vh.x() + vh.y()*vh.y());
    bh[0] = re.points[0].x();
    bh[1] = re.points[0].y();

    Tret.Tm = Tt*iTh;
    Tret.Tv = bt - Tret.Tm*bh;
    Tret.Tdet = det(Tret.Tm);

    return Tret;
}

template<typename T>
point<T,2>
ref2phys(const transform<T>& t, const point<T,2>& pt)
{
    blaze::StaticVector<T,2>    vpt;
    vpt[0] = pt.x();
    vpt[1] = pt.y();

    auto pvpt = t.Tm*vpt + t.Tv;
    return point<T,2>(pvpt[0], pvpt[1]);
}

} // namespace yaourt::refelem

namespace yaourt {
template<typename T>
auto barycenter(const refelem::reference_triangle<T>& t)
{
    return (t.points[0] + t.points[1] + t.points[2])/3.0;
}

template<typename T>
auto diameter(const refelem::reference_triangle<T>& t)
{
    T diam = 0.0;

    for (size_t i = 0; i < 3; i++)
        for (size_t j = i+1; j < 3; j++)
            diam = std::max(diam, distance(t.points[i], t.points[j]));

    return diam;
}

template<typename T>
std::vector<point<T,2>>
make_test_points(const refelem::reference_triangle<T>& t,
                 size_t levels)
{
    std::vector<point<T,2>> test_points;
    auto pts = t.points;

    levels += 1;
    auto d0 = (pts[1] - pts[0]) / levels;
    auto d1 = (pts[2] - pts[0]) / levels;

    test_points.push_back( pts[0] );
    for (size_t i = 1; i <= levels; i++)
    {
        auto p0 = pts[0] + (i * d0);
        auto p1 = pts[0] + (i * d1);

        auto d2 = (p0 - p1) / i;

        for (size_t j = 0; j <= i; j++)
            test_points.push_back( p1 + j * d2 );
    }

    return test_points;
}

}



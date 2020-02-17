/*
 * Yaourt-FEM-DG - Yet AnOther Useful Resource for Teaching FEM and DG.
 *
 * Matteo Cicuttin (C) 2019
 *
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
 */

#pragma once

#include <blaze/Math.h>

#include "mesh.hpp"

namespace yaourt {
namespace bases {

/* Perform exponentiation by integer exponent. */
template<typename T>
T
iexp_pow(T x, size_t n)
{
    if (n == 0)
        return 1;

    T y = 1;
    while (n > 1)
    {
        if (n % 2 == 0)
        {
            x = x * x;
            n = n / 2;
        }
        else
        {
            y = x * y;
            x = x * x;
            n = (n - 1) / 2;
        }
    }

    return x * y;
}

/* Compute the size of a scalar basis of degree k in dimension d. */
size_t
scalar_basis_size(size_t k, size_t d)
{
    size_t num = 1;
    size_t den = 1;

    for (size_t i = 1; i <= d; i++)
    {
        num *= k + i;
        den *= i;
    }

    return num / den;
}

template<typename Mesh, typename Element>
class scalar_basis;

template<template<typename, size_t, typename, typename> class Mesh,
         typename T, typename CellT, typename FaceT>
class scalar_basis<Mesh<T,2,CellT,FaceT>, CellT>
{
    typedef Mesh<T,2,CellT,FaceT>           mesh_type;
    typedef CellT                           elem_type;
    typedef typename mesh_type::point_type  point_type;

    point_type      elem_bar;
    size_t          basis_degree, basis_size;
    T               elem_h;

public:
    scalar_basis(const mesh_type& msh, const elem_type& elem, size_t degree)
    {
        elem_bar        = barycenter(msh, elem);
        elem_h          = diameter(msh, elem);
        basis_degree    = degree;
        basis_size      = scalar_basis_size(degree, 2);
    }

    blaze::DynamicVector<T>
    eval(const point_type& pt) const
    {
        blaze::DynamicVector<T> ret(basis_size);

        const auto b = (pt - elem_bar) / (0.5*elem_h);

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t i = 0; i <= k; i++)
            {
                const auto pow_x = k - i;
                const auto pow_y = i;

                const auto px = iexp_pow(b.x(), pow_x);
                const auto py = iexp_pow(b.y(), pow_y);

                ret[pos++] = px * py;
            }
        }

        assert(pos == basis_size);

        return ret;
    }

    blaze::DynamicMatrix<T>
    eval_grads(const point_type& pt) const
    {
        blaze::DynamicMatrix<T> ret(basis_size, 2);

        const auto ih = 2.0 / elem_h;
        const auto b = (pt - elem_bar) / (0.5*elem_h);

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t i = 0; i <= k; i++)
            {
                const auto pow_x = k - i;
                const auto pow_y = i;

                const auto px = iexp_pow(b.x(), pow_x);
                const auto py = iexp_pow(b.y(), pow_y);
                const auto dx = (pow_x == 0) ? 0 : pow_x * ih * iexp_pow(b.x(), pow_x - 1);
                const auto dy = (pow_y == 0) ? 0 : pow_y * ih * iexp_pow(b.y(), pow_y - 1);

                ret(pos, 0) = dx * py;
                ret(pos, 1) = px * dy;
                pos++;
            }
        }

        assert(pos == basis_size);

        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};

#if 0
template<template<typename, size_t, typename, typename> class Mesh,
         typename T, typename FaceT>
class scalar_basis<Mesh<T,2,quadrangle,FaceT>, quadrangle>
{
    typedef Mesh<T,2,quadrangle,FaceT>      mesh_type;
    typedef quadrangle                      elem_type;
    typedef typename mesh_type::point_type  point_type;

    point_type      elem_bar;
    size_t          basis_degree, basis_size;
    T               elem_h;

    T legendre_eval(T x, size_t degree) const
    {
        switch (degree)
        {
            case 0: return 1;
            case 1: return x;
            case 2: return 0.5*(3*x*x - 1);
            case 3: return 0.5*(5*x*x - 3)*x;
        }
    }

    T legendre_eval_deriv(T x, size_t degree) const
    {
        switch (degree)
        {
            case 0: return 0;
            case 1: return 1;
            case 2: return 3*x;
            case 3: return 7.5*x*x - 1.5;
        }
    }


public:
    scalar_basis(const mesh_type& msh, const elem_type& elem, size_t degree)
    {
        elem_bar        = barycenter(msh, elem);
        elem_h          = diameter(msh, elem);
        basis_degree    = degree;
        basis_size      = scalar_basis_size(degree, 2);
    }

    blaze::DynamicVector<T>
    eval(const point_type& pt) const
    {
        blaze::DynamicVector<T> ret(basis_size);

        const auto b = (pt - elem_bar) / (0.5*elem_h);

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t i = 0; i <= k; i++)
            {
                const auto pow_x = k - i;
                const auto pow_y = i;

                const auto px = legendre_eval(b.x(), pow_x);
                const auto py = legendre_eval(b.y(), pow_y);

                ret[pos++] = px * py;
            }
        }

        assert(pos == basis_size);

        return ret;
    }

    blaze::DynamicMatrix<T>
    eval_grads(const point_type& pt) const
    {
        blaze::DynamicMatrix<T> ret(basis_size, 2);

        const auto ih = 2.0 / elem_h;
        const auto b = (pt - elem_bar) / (0.5*elem_h);

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t i = 0; i <= k; i++)
            {
                const auto pow_x = k - i;
                const auto pow_y = i;

                const auto px = legendre_eval(b.x(), pow_x);
                const auto py = legendre_eval(b.y(), pow_y);
                const auto dx = (pow_x == 0) ? 0 : ih * legendre_eval_deriv(b.x(), pow_x);
                const auto dy = (pow_y == 0) ? 0 : ih * legendre_eval_deriv(b.y(), pow_y);

                ret(pos, 0) = dx * py;
                ret(pos, 1) = px * dy;
                pos++;
            }
        }

        assert(pos == basis_size);

        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};
#endif

template<template<typename, size_t, typename, typename> class Mesh,
         typename T, typename CellT, typename FaceT>
class scalar_basis<Mesh<T,2,CellT,FaceT>, FaceT>
{
    typedef Mesh<T,2,CellT,FaceT>   mesh_type;
    typedef FaceT                   elem_type;
    typedef point<T,2>              point_type;

    point_type      elem_bar, p0;
    size_t          basis_degree, basis_size;
    T               elem_h;

public:
    scalar_basis(const mesh_type& msh, const elem_type& elem, size_t degree)
    {
        elem_bar        = barycenter(msh, elem);
        elem_h          = diameter(msh, elem);
        basis_degree    = degree;
        basis_size      = scalar_basis_size(degree, 1);

        auto pts = points(msh, elem);
        p0 = pts[0];
    }

    blaze::DynamicVector<T>
    eval(const point_type& pt) const
    {
        blaze::DynamicVector<T> ret(basis_size);

        const auto vp   = (p0 - elem_bar);
        const auto tp   = (pt - elem_bar);
        blaze::StaticVector<T,2> v({vp.x(), vp.y()});
        blaze::StaticVector<T,2> t({tp.x(), tp.y()});
        const auto d = dot(v,t);
        const auto ep  = 4.0 * d / (elem_h * elem_h);

        for (size_t i = 0; i <= basis_degree; i++)
        {
            const auto bv = iexp_pow(ep, i);
            ret[i]  = bv;
        }
        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};

template<typename Mesh, typename Element>
auto make_basis(const Mesh& msh, const Element& elem, size_t degree)
{
    return scalar_basis<Mesh,Element>(msh, elem, degree);
}

} // namespace bases
} // namespace yaourt

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


/* Exercise number 2 - Using quadratures to compute integrals
 *
 * In this exercise we use numerical integration to compute the value
 * of simple integrals.
 * 
 * The example code computes the integral of sin( \pi x ) in the domain [0,1]^2.
 * More in detail, it does the following:
 *
 *   1) Create a cartesian mesh where to work. The mesh represents the square
 *      domain [0,1]^2.
 *
 *   2) Iterate on each cell and compute the corresponding contribution to the
 *      integral. Accumulate the result in 'int_val' and print.
 *
 * Your task is to study the code and modify it to compute the integral
 * on [0,1]^2 of the monomials 'x^m y^n'. Remember to use the right order for
 * the quadrature. Compare the results with the analytical value, which is
 *
 *                          1./((m+1)*(n+1))
 * 
 */

#include <iostream>
#include <cmath>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/quadratures.hpp"

int main(void)
{
    using T = double;

    size_t      mesh_levels = 3;
    size_t      quad_degree = 1;

    /* Declare a mesh object */
    dg2d::quad_mesh<T> msh;

    /* Ask for a mesher */
    auto mesher = dg2d::get_mesher(msh);

    /* Mesh the domain */
    mesher.create_mesh(msh, mesh_levels);

    /* Define the function to integrate */
    auto f = [](const point<T,2>& pt) -> T {
        return std::sin(M_PI * pt.x());
    };

    T int_val = 0.0;
    for (auto& cl : msh.cells)
    {
        auto qps = dg2d::quadratures::integrate(msh, cl, quad_degree);
        for (auto& qp : qps)
            int_val += qp.weight() * f( qp.point() );
    }

    std::cout << "Integral value: " << int_val << std::endl;

    return 0;
}


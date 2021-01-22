/*
 * Yaourt-FEM-DG - Yet AnOther Useful Resource for Teaching FEM and DG.
 *
 * Matteo Cicuttin (C) 2019-2020
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


/* Exercise number 3 - Using quadratures & bases
 *
 * In this exercise we will solve the problem (u,v) = (f,v) on each mesh cell,
 * in order to project a function in a polynomial space. What this code does is
 * the following:
 *
 *   1) Create a simplicial mesh where to work. The mesh represents the square
 *      domain [0,1]^2.
 *
 *   2) Iterate on each cell and compute the (u,v) term (the mass matrix) and
 *      the (f,v) term (the rhs). Solve (u,v) = (f,v) for each cell and store
 *      the result.
 *
 *   3) Compute the L2-error of the projection. Remember that the solution
 *      obtained in point (2) is a vector of coefficients of a polynomial,
 *      while the vector 'phi' you obtain by calling basis.eval(pt) is the
 *      vector of the basis functions evaluated at the point 'pt'. Suppose
 *      then that the local solution is stored in the vector 'sol': the value
 *      of the unknown function 'u' at the point 'pt' is then obtained by
 *      doing the following call: dot(sol, phi).
 *
 * For this exercise you are required to do the part (3) of the previous list.
 * The tasks are the following:
 *
 *   a) Iterate on the mesh elements and recover the local solution vector 'sol'
 *      from the global solution vector 'proj' filled during part (2).
 *
 *   b) Require a basis of degree k and a quadrature of degree 2*k+2. Recover
 *      the value of the function 'u' at the quadrature point and use it to
 *      compute the projection L2-norm error. Remember that that error is given
 *      by:
 *
 *        \sqrt{ \sum_{t \in T} \int_t (u-f)^2 }.
 *
 *   c) Plot on a log-log graph the convergence rates you find for polynomial
 *      degrees 0 and 1.
 */

#include <iostream>
#include <cmath>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/quadratures.hpp"
#include "core/bases.hpp"

#include "core/blaze_sparse_init.hpp"

int main(int argc, char **argv)
{
    using T = double;
    namespace yb = yaourt::bases;
    namespace yq = yaourt::quadratures;

    if (argc != 3)
    {
        std::cout << "Specify degree and mesh levels!" << std::endl;
        return 1;
    }

    size_t      degree      = atoi(argv[1]);
    size_t      mesh_levels = atoi(argv[2]);

    size_t      basis_size = yb::scalar_basis_size(degree, 2);

    /* Declare a mesh object */
    yaourt::simplicial_mesh<T> msh;

    /* Ask for a mesher */
    auto mesher = yaourt::get_mesher(msh);

    /* Mesh the domain */
    mesher.create_mesh(msh, mesh_levels);

    /* Define the function to project */
    auto f = [](const point<T,2>& pt) -> T {
        return std::sin(M_PI * pt.x());
    };

    blaze::DynamicVector<T> proj(msh.cells.size() * basis_size);

    /* Problem "assembly". Here you would assemble the matrix and the RHS
     * of the global linear system but, as the system here is block-diagonal
     * we can simply solve element-by-element and store the local result in
     * a global solution array */
    for (auto& cl : msh.cells)
    {
        /* Prepare to solve (u,v) = (f,v) on each cell */
        /* Initialize matrices */
        blaze::DynamicMatrix<T> M(basis_size, basis_size, 0.0);
        blaze::DynamicVector<T> rhs(basis_size, 0.0);

        /* Ask for a basis on the current cell */
        auto basis = yb::make_basis(msh, cl, degree);

        /* Ask for the quadrature */
        auto qps = yq::integrate(msh, cl, 2*degree);
        for (auto& qp : qps)
        {
            auto phi = basis.eval(qp.point());          /* Evaluate basis */

            M   += qp.weight() * phi * trans(phi);      /* Build mass matrix */
            rhs += qp.weight() * f(qp.point()) * phi;   /* Build RHS */
        }

        auto proj_loc = blaze::solve_LU(M, rhs);        /* Solve M*sol = rhs */

        auto cl_num = offset(msh, cl);
        for (size_t i = 0; i < basis_size; i++)         /* Store the solution */
            proj[cl_num*basis_size + i] = proj_loc[i];
    }

    /* At this point 'proj' contains the global solution of the problem
     * (u_h, v_h) = (f, v_h). */

    /* Hints:
     *  - Fill 'sol' with the appropriate values of 'proj'
     *  - Once 'sol' is filled, you can use 'dot(proj_loc, phi)' to get the
     *    value of the solution at your quadrature point
     *  - Remember to use a quadrature of degree 2k+2
     */

    /* Postprocessing */
    T L2_errsq = 0.0;
    for (auto& cl : msh.cells)
    {
        auto cl_num = offset(msh, cl);

        /* (1) Recover element-local solution */
        blaze::DynamicVector<T> proj_loc(basis_size);
        for (size_t i = 0; i < basis_size; i++) 
            proj_loc[i] = proj[cl_num*basis_size + i];

        auto basis = yb::make_basis(msh, cl, degree);
        auto qps = yq::integrate(msh, cl, 2*degree+2);
        for (auto& qp : qps)
        {
            /* (2) Evaluate local solution and reference solution at quadrature
             * points, then compute L2-error */
            auto phi = basis.eval(qp.point());
            auto c_val = dot(proj_loc, phi);
            auto r_val = f(qp.point());

            L2_errsq += qp.weight() * (c_val-r_val) * (c_val-r_val);
        }
    }

    std::cout << "Projection L2 error: " << std::sqrt(L2_errsq) << std::endl;

    return 0;
}


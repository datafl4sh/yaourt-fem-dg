/*
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
 * 
 * Toy implememtation of Discontinuous Galerkin for teaching purposes
 *
 * Matteo Cicuttin (c) 2018
 */

#include <iostream>
#include <vector>
#include <list>
#include <blaze/Math.h>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/dataio.hpp"
#include "core/solvers.hpp"
#include "method_cfem/cfem.hpp"

int main(int argc, char **argv)
{

    using T = double;

    dg2d::simplicial_mesh<T> mesh;
    auto mesher = dg2d::get_mesher(mesh);

    mesher.create_mesh(mesh, 6);

#ifdef WITH_SILO
    dg2d::silo_database silo;
    silo.create("test.silo");

    silo.add_mesh(mesh, "test_mesh");
#endif /* WITH_SILO */

    auto assembler = dg2d::cfem::get_assembler(mesh, 1);

    for(auto& cl : mesh.cells)
    {
        auto pts = points(mesh, cl);
        blaze::StaticVector<T,3> local_rhs;

        auto bar = barycenter(mesh, cl);
        auto meas = measure(mesh, cl);
        T bval = 2.0 * M_PI * M_PI * meas * std::sin(M_PI*bar.x()) * std::sin(M_PI*bar.y()) / 3.0;

        local_rhs[0] = bval;
        local_rhs[1] = bval;
        local_rhs[2] = bval;

        auto local_lhs = dg2d::cfem::stiffness_matrix(mesh, cl);
        assembler.assemble(mesh, cl, local_lhs, local_rhs);
    }

    assembler.finalize();

    blaze::DynamicVector<T> sol(assembler.system_size());

    conjugated_gradient_params<T> cgp;
    cgp.verbose = true;
    cgp.max_iter = 2*assembler.system_size();

    conjugated_gradient(cgp, assembler.lhs, assembler.rhs, sol);

    blaze::DynamicVector<T> exp_sol;

    assembler.expand(sol, exp_sol);

#ifdef WITH_SILO
    silo.add_nodal_variable("test_mesh", "solution", exp_sol);
#endif

    return 0;
}




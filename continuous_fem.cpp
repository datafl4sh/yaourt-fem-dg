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

#include <iostream>
#include <vector>
#include <list>

#include <blaze/Math.h>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/dataio.hpp"
#include "core/solvers.hpp"
#include "methods/cfem.hpp"

template<typename T>
blaze::StaticVector<T, 3>
compute_rhs(const yaourt::simplicial_mesh<T>& msh,
            const typename yaourt::simplicial_mesh<T>::cell_type& cl)
{
    auto pts = points(msh, cl);
    blaze::StaticVector<T,3> local_rhs;

    auto bar = barycenter(msh, cl);
    auto meas = measure(msh, cl);
    T bval = std::sin(M_PI*bar.x()) * std::sin(M_PI*bar.y());
    bval = bval * 2.0 * M_PI * M_PI * meas / 3.0;

    local_rhs[0] = bval;
    local_rhs[1] = bval;
    local_rhs[2] = bval;

    return local_rhs;
}

int main(int argc, char **argv)
{

    using T = double;

    yaourt::simplicial_mesh<T> msh;
    auto mesher = yaourt::get_mesher(msh);

    mesher.create_mesh(msh, 2);

    auto assembler = yaourt::cfem::get_assembler(msh, 1);

    for(auto& cl : msh.cells)
    {
        auto local_lhs = yaourt::cfem::stiffness_matrix(msh, cl);
        auto local_rhs = compute_rhs(msh, cl);
        assembler.assemble(msh, cl, local_lhs, local_rhs);
    }

    assembler.finalize();

    blaze::DynamicVector<T> sol(assembler.system_size());

    conjugated_gradient_params<T> cgp;
    cgp.verbose = true;
    cgp.rr_max = 1000;
    cgp.max_iter = 2*assembler.system_size();

    conjugated_gradient(cgp, assembler.lhs, assembler.rhs, sol);

    blaze::DynamicVector<T> exp_sol;

    assembler.expand(sol, exp_sol);
    
    std::ofstream gnuplot_output("cfem_solution.txt");
    
    for (auto& cl : msh.cells)
    {
        auto l2g = cl.point_ids();

        blaze::DynamicVector<T> loc_sol(3);
        for (size_t i = 0; i < 3; i++)
            loc_sol[i] = exp_sol[ l2g[i] ];

        auto tps = yaourt::make_test_points(msh, cl, 7);
        for (auto& tp : tps)
        {
            auto phi = yaourt::cfem::eval_basis(msh, cl, tp);
            T sval = dot(loc_sol, phi);

            gnuplot_output << tp.x() << " " << tp.y() << " " << sval << std::endl;
        }
    }

#ifdef WITH_SILO
    yaourt::dataio::silo_database silo;
    silo.create("test.silo");

    silo.add_mesh(msh, "test_mesh");

    silo.add_nodal_variable("test_mesh", "solution", exp_sol);
#endif

    return 0;
}

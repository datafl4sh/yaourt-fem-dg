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
#include "mesh.hpp"
#include "meshers.hpp"
#include "dataio.hpp"

#include "cfem.hpp"

int main(int argc, char **argv)
{

    using T = double;

    dg2d::simplicial_mesh<T> mesh;
    auto mesher = dg2d::get_mesher(mesh);

    mesher.create_mesh(mesh, 4);

#ifdef WITH_SILO
    dg2d::silo_database silo;
    silo.create("test.silo");

    silo.add_mesh(mesh, "test_mesh");
#endif /* WITH_SILO */

    for(auto& cl : mesh.cells)
    {
        auto bar = barycenter(mesh, cl);
        auto bf = dg2d::cfem::eval_basis(mesh, cl, bar);
        auto bg = dg2d::cfem::eval_basis_grad(mesh, cl);
    }

    return 0;
}




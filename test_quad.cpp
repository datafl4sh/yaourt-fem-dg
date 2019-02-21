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

int main(int argc, char **argv)
{

    using T = double;

    dg2d::quad_mesh<T> msh;
    auto mesher = dg2d::get_mesher(msh);

    mesher.create_mesh(msh, 3);

    dg2d::silo_database silo;
    silo.create("test.silo");

    silo.add_mesh(msh, "test_mesh");
}

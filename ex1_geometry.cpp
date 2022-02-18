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

/* Exercise number 1 - Start exploring the code
 *
 * In this exercise we will see some of the basic functionalities of the code.
 * 
 */

#include <iostream>
#include <cmath>

#include "core/mesh.hpp"
#include "core/meshers.hpp"

int main(void)
{
    using T = double;

    /* Declare a mesh object */
    //yaourt::quad_mesh<T> msh;
    yaourt::simplicial_mesh<T> msh;

    /* Ask for a mesher */
    auto mesher = yaourt::get_mesher(msh);

    /* Mesh the domain */
    mesher.create_mesh(msh, 0);

    T int_val = 0.0;
    for (auto& cl : msh.cells)
    {
        std::cout << cl << std::endl;
        std::cout << "  Measure:    " << measure(msh, cl) << std::endl;
        std::cout << "  Barycenter: " << barycenter(msh, cl) << std::endl;

        auto fcs = faces(msh, cl);
        for (auto& fc : fcs)
        {
            std::cout << "    " << fc << std::endl;
            std::cout << "      Measure:    " << measure(msh, fc) << std::endl;
            std::cout << "      Barycenter: " << barycenter(msh, fc) << std::endl;
            std::cout << "      Normal: " << trans(normal(msh, cl, fc)) << std::endl;
        }
    }

    return 0;
}


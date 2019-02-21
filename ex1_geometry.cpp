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
    //dg2d::quad_mesh<T> msh;
    dg2d::simplicial_mesh<T> msh;

    /* Ask for a mesher */
    auto mesher = dg2d::get_mesher(msh);

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
        }
    }

    return 0;
}


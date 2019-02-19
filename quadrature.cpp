#include <iostream>
#include <cmath>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/quadratures.hpp"

int main(void)
{
    using T = double;

    /* Declare a mesh object */
    dg2d::simplicial_mesh<T> msh;

    /* Ask for a mesher */
    auto mesher = dg2d::get_mesher(msh);

    /* Mesh the domain */
    mesher.create_mesh(msh, 3);

    /* Define the function to integrate */
    auto f = [](const point<T,2>& pt) -> T {
        return std::sin(M_PI * pt.x());
    };

    T int_val = 0.0;
    for (auto& cl : msh.cells)
    {
        auto qps = dg2d::quadratures::integrate(msh, cl, 2);
        for (auto& qp : qps)
            int_val += qp.weight() * f( qp.point() );
    }

    std::cout << "Integral value: " << int_val << std::endl;

    return 0;
}

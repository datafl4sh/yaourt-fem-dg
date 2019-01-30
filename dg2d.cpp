#include <iostream>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/quadratures.hpp"

int main(void)
{
    using T = double;

    dg2d::simplicial_mesh<T> msh;
    auto mesher = dg2d::get_mesher(msh);

    mesher.create_mesh(msh, 6);

    T area = 0;

    for (auto& cl : msh.cells)
    {
        auto qps = dg2d::quadratures::integrate(msh, cl, 2);

        for (auto& qp : qps)
        {
            area += qp.weight();
        }
    }

    std::cout << area << std::endl;

    for (auto& fc : msh.faces)
    {
        auto qps = dg2d::quadratures::integrate(msh, fc, 2);
    }

    return 0;
}
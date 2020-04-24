#include <iostream>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/refelem.hpp"


int main(void)
{
    using T = double;

    //namespace yq = yaourt::quadratures;
    //namespace yb = yaourt::bases;
    namespace yr = yaourt::refelem;

    yaourt::simplicial_mesh<T> msh;

    auto mesher = yaourt::get_mesher(msh);
    mesher.create_mesh(msh, 0);

    yr::reference_triangle<T>   rt;
    yr::reference_edge<T>       re;

    for (auto& cl : msh.cells)
    {
        auto cl_ref2phys = yr::make_ref2phys_transform(msh, cl, rt);
        point<T,2> bar_ref(1./3., 1./3.);
        point<T,2> bar_phys = cl_ref2phys(bar_ref);
        auto bar = barycenter(msh, cl);

        std::cout << bar_phys << bar << std::endl;

        auto fcs = faces(msh, cl);
        for (auto& fc : fcs)
        {
            auto fc_ref2phys = yr::make_ref2phys_transform(msh, fc, re);
            point<T,2> fbar_ref(0., 0);
            point<T,2> fbar_phys = fc_ref2phys(fbar_ref);
            auto fbar = barycenter(msh, fc);

            std::cout << "  " << fbar_phys << fbar << std::endl;
        }
    }

    return 0;
}


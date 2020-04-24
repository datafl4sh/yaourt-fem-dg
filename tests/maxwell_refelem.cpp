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

#define USE_REFERENCE_SIMPLEX

#include <iostream>
#include <fstream>

#include "methods/dg_maxwell_2D.hpp"

namespace ymax = yaourt::maxwell_2D;

int main(void)
{
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);

    using T = double;
    using Mesh = yaourt::simplicial_mesh<T>;

    ymax::maxwell_config<T> cfg;
    cfg.degree = 3;
    cfg.mesh_levels = 2;

    ymax::maxwell_context<Mesh> ctx(cfg);

#ifdef USE_REFERENCE_SIMPLEX
    assemble(ctx);
    std::cout << ctx.rM << std::endl;

    yaourt::refelem::reference_triangle<T> rt;

    std::ofstream ofs("basisplot.dat");
    std::ofstream ofsg("gradplot.dat");

    auto pf = [](const point<T,2>& pt) -> auto {
        return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
    };

    auto rbas = yaourt::bases::make_basis(rt, ctx.cfg.degree);
    auto qps = yaourt::quadratures::integrate(rt, 2*ctx.cfg.degree);

    size_t i = 0;
    for (auto& cl : ctx.msh.cells)
    {
        assert(i < ctx.msh.cells.size());

        auto cl_trans = yaourt::refelem::make_ref2phys_transform(ctx.msh, cl, rt);

        blaze::DynamicVector<T> rhs(rbas.size(), 0.0);
        for (auto& qp : qps)
        {
            auto phi = rbas.eval(qp.point());
            rhs += qp.weight() * pf(cl_trans(qp.point())) * phi;
        }

        /* Yes, indices are swapped because we need trans(Tm) */
        auto b11 = ctx.gGeomCoeffs[i].Tm(0,0);
        auto b12 = ctx.gGeomCoeffs[i].Tm(1,0);

        auto b21 = ctx.gGeomCoeffs[i].Tm(0,1);
        auto b22 = ctx.gGeomCoeffs[i].Tm(1,1);

        auto op_eta = blaze::solve(ctx.rM, ctx.rSeta);
        auto op_xi = blaze::solve(ctx.rM, ctx.rSxi);

        auto op_x = b11*op_eta + b12*op_xi;
        auto op_y = b21*op_eta + b22*op_xi;

        blaze::DynamicVector<T> temp = solve(ctx.rM, rhs);
        blaze::DynamicVector<T> proj = op_x*temp;

        auto tps = yaourt::make_test_points(rt, 10);
        for (auto& tp : tps)
        {
            auto phi = rbas.eval(tp);
            auto val = dot(proj, phi);
            auto pp = cl_trans(tp);
            ofs << pp.x() << " " << pp.y() << " " << val << std::endl;
        }


        /* LAST */
        i++;
    }
#else
    assemble(ctx);

    for (size_t i = 0; i < ctx.msh.cells.size(); i++)
    {
    	auto bs = ctx.basis_size;
    	auto offset = bs * i;
    	std::cout << "Element mass matrix" << std::endl;
    	std::cout << submatrix(ctx.gM, offset, 0, bs, bs) << std::endl;
    }
#endif

    return 0;
}
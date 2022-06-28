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
#include <fstream>
#include <sstream>

#include <cstring>
#include <cmath>

#include <unistd.h>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/quadratures.hpp"
#include "core/bases.hpp"
#include "core/solvers.hpp"
#include "core/blaze_sparse_init.hpp"
#include "core/dataio.hpp"

#include "methods/dg.hpp"

namespace params {
/* Diffusion term coefficient */
template<typename T>
T
epsilon(const point<T,2>& pt)
{
    return 1.0;
}

} // namespace params

namespace data {
template<typename T>
T
rhs(const point<T,2>& pt)
{
    auto sx = std::sin(M_PI*pt.x());
    auto sy = std::sin(M_PI*pt.y());

    return 2.0 * M_PI * M_PI * sx * sy;
};

template<typename T>
T
dirichlet(const point<T,2>& pt)
{
    return 0.0;
};

template<typename T>
T
diffusion_ref_sol(const point<T,2>& pt)
{
    auto sx = std::sin(M_PI*pt.x());
    auto sy = std::sin(M_PI*pt.y());

    return sx * sy;
};

} // namespace data


template<typename T>
struct dg_config
{
    T               eta;
    int             degree;
    int             ref_levels;
    bool            use_preconditioner;
    bool            shatter;

    dg_config()
        : eta(1.0), degree(1), ref_levels(4), use_preconditioner(false),
          shatter(false)
    {}
};

template<typename T>
struct solver_status
{
    T   mesh_h;
    T   L2_errsq_qp;
    T   L2_errsq_mm;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const solver_status<T>& s)
{
    os << "Convergence results: " << std::endl;
    os << "  mesh size (h):         " << s.mesh_h << std::endl;
    os << "  L2-norm error (qp):    " << std::sqrt(s.L2_errsq_qp) << std::endl;
    os << "  L2-norm error (mm):    " << std::sqrt(s.L2_errsq_mm);
    return os;
}

template<typename Mesh>
solver_status<typename Mesh::coordinate_type>
run_diffusion_solver(Mesh& msh, const dg_config<typename Mesh::coordinate_type>& cfg)
{
    using mesh_type = Mesh;
    using T = typename mesh_type::coordinate_type;

    solver_status<T>    status;
    status.mesh_h = diameter(msh);

    msh.compute_connectivity();

    size_t degree = cfg.degree;
    T eta = 3*degree*degree*cfg.eta;


    /* PROBLEM ASSEMBLY */
    assembler<mesh_type> assm(msh, degree, cfg.use_preconditioner);

    for (auto& tcl : msh.cells)
    {
        auto tbasis = yaourt::bases::make_basis(msh, tcl, degree);

        blaze::DynamicMatrix<T> K(tbasis.size(), tbasis.size(), 0.0);
        blaze::DynamicVector<T> loc_rhs(tbasis.size(), 0.0);

        auto qps = yaourt::quadratures::integrate(msh, tcl, 2*degree);
        for (auto& qp : qps)
        {
            auto ep   = qp.point();
            auto phi  = tbasis.eval(ep);
            auto dphi = tbasis.eval_grads(ep);

            K += qp.weight() * dphi * trans(dphi);
            loc_rhs += qp.weight() * data::rhs(ep) * phi;
        }

        auto fcs = faces(msh, tcl);
        for (auto& fc : fcs)
        {
            blaze::DynamicMatrix<T> Att(tbasis.size(), tbasis.size(), 0.0);
            blaze::DynamicMatrix<T> Atn(tbasis.size(), tbasis.size(), 0.0);

            auto [ncl, has_neighbour] = neighbour_via(msh, tcl, fc);
            auto nbasis = yaourt::bases::make_basis(msh, ncl, degree);
            assert(tbasis.size() == nbasis.size());

            auto n     = normal(msh, tcl, fc);
            auto eta_l = eta / diameter(msh, fc);
            auto f_qps = yaourt::quadratures::integrate(msh, fc, 2*degree);

            for (auto& fqp : f_qps)
            {
                auto ep     = fqp.point();
                auto tphi   = tbasis.eval(ep);
                auto tdphi  = tbasis.eval_grads(ep);

                if (has_neighbour)
                {   /* NOT on a boundary */
                    Att += // [u][v]
                    Att += // {grad(u).n}[v]
                    Att += // [u]{grad(v).n}
                    
                    auto nphi   = nbasis.eval(ep);
                    auto ndphi  = nbasis.eval_grads(ep);

                    Atn += // [u][v]
                    Atn += // {grad(u).n}[v]
                    Atn += // [u]{grad(v).n}
                }
                else
                {   /* On a boundary*/
                    Att += // [u][v]
                    Att += // {grad(u).n}[v]
                    Att += // [u]{grad(v).n}

                    loc_rhs -= /* dirichlet conditions */;
                    loc_rhs += /* dirichlet conditions */;
                }
            }

            assm.assemble(msh, tcl, tcl, Att);
            if (has_neighbour)
                assm.assemble(msh, tcl, ncl, Atn);
        }

        assm.assemble(msh, tcl, K, loc_rhs);
    }

    assm.finalize();

    /* SOLUTION PART */
    blaze::DynamicVector<T> sol(assm.system_size());

    conjugated_gradient_params<T> cgp;
    cgp.verbose = true;
    cgp.rr_max = 10000;
    cgp.rr_tol = 1e-8;
    cgp.max_iter = 2*assm.system_size();

    if (cfg.use_preconditioner)
        conjugated_gradient(cgp, assm.lhs, assm.rhs, sol, assm.pc);
    else    
        conjugated_gradient(cgp, assm.lhs, assm.rhs, sol);

    /* POSTPROCESS PART */

    std::ofstream gnuplot_output("diffusion_solution.txt");

    status.L2_errsq_qp = 0.0;
    status.L2_errsq_mm = 0.0;
    for (auto& cl : msh.cells)
    {

        auto basis = yaourt::bases::make_basis(msh, cl, degree);
        auto basis_size = basis.size();
        auto ofs = offset(msh, cl);

        blaze::DynamicVector<T> loc_sol(basis_size);
        for (size_t i = 0; i < basis_size; i++)
            loc_sol[i] = sol[basis_size * ofs + i];

        auto tps = yaourt::make_test_points(msh, cl, 7);
        for (auto& tp : tps)
        {
            auto phi = basis.eval(tp);
            T sval = dot(loc_sol, phi);

            gnuplot_output << tp.x() << " " << tp.y() << " " << sval << std::endl;
        }

        blaze::DynamicMatrix<T> M(basis_size, basis_size, 0.0);
        blaze::DynamicVector<T> a(basis_size, 0.0);

        auto qps = yaourt::quadratures::integrate(msh, cl, 2*degree);
        for (auto& qp : qps)
        {
            auto ep   = qp.point();
            auto phi  = basis.eval(ep);

            auto sv = data::diffusion_ref_sol(ep);

            M += qp.weight() * phi * trans(phi);
            a += qp.weight() * sv * phi;

            T cv = dot(loc_sol, phi);
            status.L2_errsq_qp += qp.weight() * (sv - cv) * (sv - cv);
        }

        blaze::DynamicVector<T> proj = blaze::solve_LU(M, a);
        status.L2_errsq_mm += dot(proj-loc_sol, M*(proj-loc_sol));

    }

#ifdef WITH_SILO

    blaze::DynamicVector<T> var(msh.cells.size());
    auto bs = yaourt::bases::scalar_basis_size(degree, 2);
    for (size_t i = 0; i < msh.cells.size(); i++)
    {
        var[i] = sol[bs*i];
    }


    blaze::DynamicVector<T> dbg_epsilon( msh.points.size() );

    for (size_t i = 0; i < msh.points.size(); i++)
    {
        auto pt = msh.points.at(i);
        dbg_epsilon[i] = params::epsilon(pt);
    }

    yaourt::dataio::silo_database silo;
    silo.create("test_dg.silo");
    silo.add_mesh(msh, "test_mesh");
    silo.add_zonal_variable("test_mesh", "solution", var);
    silo.add_nodal_variable("test_mesh", "epsilon", dbg_epsilon);
    silo.close();

#endif

    return status;
}

template<typename Mesh>
void run_dg(const dg_config<typename Mesh::coordinate_type>& cfg)
{
    using mesh_type = Mesh;
    using T = typename Mesh::coordinate_type;

    mesh_type msh;

    auto mesher = yaourt::get_mesher(msh);
    mesher.create_mesh(msh, cfg.ref_levels);

    if (cfg.shatter)
        shatter_mesh(msh, 0.2);

    solver_status<T> status;

    std::cout << "Running dG diffusion solver" << std::endl;
    std::cout << "  degree: " << cfg.degree << ", eta: " << cfg.eta << std::endl;
    status = run_diffusion_solver(msh, cfg);
    std::cout << status << std::endl;
}

int main(int argc, char **argv)
{
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);

    using T = double;

    auto mt = yaourt::meshtype::TRIANGULAR;

    dg_config<T> cfg;

    int     ch;

    while ( (ch = getopt(argc, argv, "e:k:r:m:pSh")) != -1 )
    {
        switch(ch)
        {
            case 'e':
                cfg.eta = atof(optarg);
                break;

            case 'k':
                cfg.degree = atoi(optarg);
                break;

            case 'r':
                cfg.ref_levels = atoi(optarg);
                if (cfg.ref_levels < 0)
                    cfg.ref_levels = 1;
                break;

            case 'm':
                if ( strcmp(optarg, "tri") == 0 )
                    mt = yaourt::meshtype::TRIANGULAR;
                else if ( strcmp(optarg, "quad") == 0 )
                    mt = yaourt::meshtype::QUADRANGULAR;
                break;

            case 'p':
                cfg.use_preconditioner = true;
                break;

            case 'S':
                cfg.shatter = true;
                break;

            case 'h':
            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;


    switch (mt)
    {
        case yaourt::meshtype::TRIANGULAR:
            run_dg< yaourt::simplicial_mesh<T> >(cfg);
            break;

        case yaourt::meshtype::QUADRANGULAR:
            run_dg< yaourt::quad_mesh<T> >(cfg);
            break;

        default:
            std::cout << "Only triangular and quadrangular meshes for now" << std::endl;
            break;
    }


    return 0;
}

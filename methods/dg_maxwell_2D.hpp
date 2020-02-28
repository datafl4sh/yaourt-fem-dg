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

#pragma once

#include <chrono>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/quadratures.hpp"
#include "core/bases.hpp"
#include "core/blaze_sparse_init.hpp"
#include "core/dataio.hpp"


#define LOGLEVEL_INFO(x)        (x > 0)
#define LOGLEVEL_DETAIL(x)      (x > 1)

namespace yaourt::maxwell_2D {

enum class time_integrator_type {
    EXPLICIT_EULER, /* Don't use, only for experimental purposes */
    RUNGE_KUTTA_4,
};

template<typename T>
struct maxwell_config
{
    size_t                  degree;         /* Method degree */
    size_t                  mesh_levels;    /* Number of mesh refinement steps */
    size_t                  timesteps;      /* Total number of timesteps */
    size_t                  output_rate;    /* Number of timesteps between data dumps */

    T                       delta_t;        /* Timestep size */
    T                       eta;            /* dG penalization parameter */

    size_t                  verbosity;      /* Verbose output */
    bool                    upwind;         /* Use upwind fluxes */

    time_integrator_type    time_integrator;/* Time integrator type */

    char *                  error_fn;       /* Error dump filename */
    char *                  silo_basename;  /* Base name for silo export */

    maxwell_config() :
        degree(1), mesh_levels(4), timesteps(100), output_rate(10),
        delta_t(0.1), eta(1.0), verbosity(0), upwind(false),
        time_integrator(time_integrator_type::RUNGE_KUTTA_4),
        error_fn(nullptr), silo_basename(nullptr)
    {}
};


template<typename Mesh>
struct maxwell_context
{
    typedef Mesh                                    mesh_type;
    typedef typename mesh_type::coordinate_type     T;

    /* Configuration associated to this context */
    maxwell_config<T>           cfg;

    /* Mesh on which problem is solved */
    mesh_type                   msh;

    /* Global degrees of freedom */
    blaze::DynamicVector<T>     gDofs, gDofs_t_plus_one;

    /* Elemental mass and stiffness matrices (stiffness not needed, actually) */
    blaze::DynamicMatrix<T>     gM;//, gSx, gSy;

    /* Global operator diagonal and offdiagonal terms */
    blaze::DynamicMatrix<T>     gOp_ondiag, gOp_offdiag;

    /* Material parameters */
    T                           mu_0, eps_0;
    blaze::DynamicVector<T>     mu_r, eps_r;

    /* Basis size*/
    size_t                      basis_size;

    using no_pair_t = std::pair<size_t, bool>;
    std::vector<no_pair_t>      offdiag_neigh_offsets;

    constexpr int faces_per_elem()
    {
        if (std::is_same<Mesh, yaourt::simplicial_mesh<T>>::value)
            return 3;

        if (std::is_same<Mesh, yaourt::quad_mesh<T>>::value)
            return 4;

        return 0;
    }

    maxwell_context() = delete;

    maxwell_context(const maxwell_config<T>& p_cfg) :
        cfg(p_cfg), mu_0(4.*M_PI*1e-7), eps_0(8.8541878128e-12)
    {
        namespace yb = yaourt::bases;

        /* Create mesh */
        auto mesher = yaourt::get_mesher(msh, LOGLEVEL_INFO(cfg.verbosity));
        mesher.create_mesh(msh, cfg.mesh_levels);
        msh.compute_connectivity();

        /* Initialize data storage */
        basis_size = yb::scalar_basis_size(cfg.degree, 2);

        auto num_fDofs = basis_size * msh.cells.size();
        auto num_gDofs = 3 * num_fDofs;
        auto num_lDofs = 3 * basis_size;

        gDofs.resize(num_gDofs);
        gDofs_t_plus_one.resize(num_gDofs);

        gM.resize(num_fDofs, basis_size);
        //gSx.resize(num_fDofs, basis_size);
        //gSy.resize(num_fDofs, basis_size);

        gOp_ondiag.resize(num_gDofs, num_lDofs);
        gOp_offdiag.resize( faces_per_elem() * num_gDofs, num_lDofs );
        offdiag_neigh_offsets.resize( faces_per_elem() * num_gDofs );

        mu_r.resize(msh.cells.size());      mu_r = 1.0;
        eps_r.resize(msh.cells.size());     eps_r = 1.0;
    }
};

template<typename Mesh>
void
assemble(maxwell_context<Mesh>& ctx)
{
    using namespace blaze;
    namespace yb = yaourt::bases;
    namespace yq = yaourt::quadratures;

    using T = typename Mesh::coordinate_type;

    auto basis_size = yb::scalar_basis_size(ctx.cfg.degree, 2);

    if ( LOGLEVEL_INFO(ctx.cfg.verbosity) )
    {
        size_t ndofs = 3*basis_size*ctx.msh.cells.size();
        std::cout << "Assembling dG operator, " << ndofs << " DoFs." << std::endl;
    }

    auto asm_start_time = std::chrono::system_clock::now();

    size_t cell_i = 0;
    size_t offdiag_contrib_i = 0;
    for (auto& tcl : ctx.msh.cells)
    {
        auto tbasis = yb::make_basis(ctx.msh, tcl, ctx.cfg.degree);

        DynamicMatrix<T> M2d(basis_size, basis_size, 0.0);
        DynamicMatrix<T> Sx(basis_size, basis_size, 0.0);
        DynamicMatrix<T> Sy(basis_size, basis_size, 0.0);
        
        /* Make mass and stiffness matrices on the element */
        auto qps = yq::integrate(ctx.msh, tcl, 2*ctx.cfg.degree);
        for (auto& qp : qps)
        {
            auto ep   = qp.point();
            auto phi  = tbasis.eval(ep);
            auto dphi = tbasis.eval_grads(ep);

            auto dphi_x = blaze::column<0>(dphi);
            auto dphi_y = blaze::column<1>(dphi);

            /* Mass */
            M2d += qp.weight() * phi * trans( phi );
            /* Stiffness, direction x */
            Sx += qp.weight() * phi * trans( dphi_x );
            /* Stiffness, direction y */
            Sy += qp.weight() * phi * trans( dphi_y );
        }

        /* Save local mass matrix, will be needed in iteration */
        auto gM_offset = cell_i*basis_size;
        submatrix(ctx.gM, gM_offset, 0, basis_size, basis_size) = M2d;

        auto get_global_ondiag_block = [&](size_t i, size_t j) -> auto {
            auto offset_i = 3*basis_size*cell_i + i*basis_size;
            auto offset_j = j*basis_size;
            return submatrix(ctx.gOp_ondiag, offset_i, offset_j, basis_size, basis_size);
        };

        auto mu         = ctx.mu_r[cell_i];
        auto eps        = ctx.eps_r[cell_i];
        auto inv_mu     = 1./mu;
        auto inv_eps    = 1./eps;

        auto invM2d_Sx  = blaze::solve(M2d, Sx);
        auto invM2d_Sy  = blaze::solve(M2d, Sy);

        auto Z_this     = std::sqrt(mu/eps);
        auto Y_this     = 1./Z_this;
        
        get_global_ondiag_block(0, 2) = -inv_mu * invM2d_Sy;

        get_global_ondiag_block(1, 2) = +inv_mu * invM2d_Sx;

        get_global_ondiag_block(2, 0) = -inv_eps * invM2d_Sy;
        get_global_ondiag_block(2, 1) = +inv_eps * invM2d_Sx;


        /* Do numerical fluxes */
        auto fcs = faces(ctx.msh, tcl);
        for (auto& fc : fcs)
        {
            auto [ncl, has_neighbour] = neighbour_via(ctx.msh, tcl, fc);
            auto nbasis = yb::make_basis(ctx.msh, ncl, ctx.cfg.degree);
            assert(tbasis.size() == nbasis.size());

            auto n       = normal(ctx.msh, tcl, fc);
            auto nx      = n[0];
            auto ny      = n[1];

            DynamicMatrix<T> FC_diag(3*basis_size, 3*basis_size, 0.0);
            DynamicMatrix<T> FC_offdiag(3*basis_size, 3*basis_size, 0.0);

            auto get_block = [&](DynamicMatrix<T>& M, size_t i, size_t j) -> auto {
                auto offset_i = i*basis_size;
                auto offset_j = j*basis_size;
                return submatrix(M, offset_i, offset_j, basis_size, basis_size);
            };

            if (has_neighbour)
            {   /* NOT on a boundary */
                size_t neigh_ofs = offset(ctx.msh, ncl);
                ctx.offdiag_neigh_offsets[offdiag_contrib_i] = std::make_pair(neigh_ofs, true);

                /* Default use centered fluxes */
                T kappa_E = 0.5;
                T kappa_H = 0.5;
                T nu_E = 0.0;
                T nu_H = 0.0;

                if (ctx.cfg.upwind)
                {
                    auto mu_neigh = ctx.mu_r[neigh_ofs];
                    auto eps_neigh = ctx.eps_r[neigh_ofs];
                    auto Z_neigh = std::sqrt(mu_neigh/eps_neigh);
                    auto Y_neigh = 1./Z_neigh;

                    kappa_E = Y_neigh/(Y_this + Y_neigh);
                    kappa_E = Z_neigh/(Z_this + Z_neigh);
                    nu_E = ctx.cfg.eta/(Y_this + Y_neigh);
                    nu_H = ctx.cfg.eta/(Z_this + Z_neigh);
                }

                auto f_qps = yq::integrate(ctx.msh, fc, 2*ctx.cfg.degree);
                for (auto& fqp : f_qps)
                {
                    auto ep = fqp.point();

                    /* Basis evaluated on myself */
                    auto tphi  = tbasis.eval(ep);
                    auto tmass = fqp.weight() * tphi * trans(tphi);

                    /* Basis evaluated on the neighbour */
                    auto nphi  = nbasis.eval(ep);
                    auto nmass = fqp.weight() * tphi * trans(nphi);

                    /* Centered, myself */
                    get_block(FC_diag, 0, 2) += (+ny * kappa_E * inv_mu) * tmass;           // Hx equation, [E]
                    get_block(FC_diag, 1, 2) += (-nx * kappa_E * inv_mu) * tmass;           // Hy equation, [E]
                    get_block(FC_diag, 2, 0) += (+ny * kappa_H * inv_eps) * tmass;          // Ez equation, [H] (1)
                    get_block(FC_diag, 2, 1) += (-nx * kappa_H * inv_eps) * tmass;          // Ez equation, [H] (2)

                    /* Centered, neighbour */
                    get_block(FC_offdiag, 0, 2) -= +ny * kappa_E * inv_mu * nmass;          // Hx equation, [E]
                    get_block(FC_offdiag, 1, 2) -= -nx * kappa_E * inv_mu * nmass;          // Hy equation, [E]
                    get_block(FC_offdiag, 2, 0) -= +ny * kappa_H * inv_eps * nmass;         // Ez equation, [H] (1)
                    get_block(FC_offdiag, 2, 1) -= -nx * kappa_H * inv_eps * nmass;         // Ez equation, [H] (2)

                    if (ctx.cfg.upwind)
                    {   /* Upwind, myself */
                        get_block(FC_diag, 0, 0) += (-ny * ny * nu_H * inv_mu) * tmass;     // Hx equation, [H]
                        get_block(FC_diag, 0, 1) += (+ny * nx * nu_H * inv_mu) * tmass;     // Hx equation, [H]
                        get_block(FC_diag, 1, 0) += (+nx * ny * nu_H * inv_mu) * tmass;     // Hy equation, [H]
                        get_block(FC_diag, 1, 1) += (-nx * nx * nu_H * inv_mu) * tmass;     // Hy equation, [H]
                        get_block(FC_diag, 2, 2) += (-nu_E * inv_eps * tmass);              // Ez equation, [E]

                        /* Upwind, neighbour */
                        get_block(FC_offdiag, 0, 0) -= -ny * ny * nu_H * inv_mu * nmass;    // Hx equation, [H]
                        get_block(FC_offdiag, 0, 1) -= +ny * nx * nu_H * inv_mu * nmass;    // Hx equation, [H]
                        get_block(FC_offdiag, 1, 0) -= +nx * ny * nu_H * inv_mu * nmass;    // Hy equation, [H]
                        get_block(FC_offdiag, 1, 1) -= -nx * nx * nu_H * inv_mu * nmass;    // Hy equation, [H]
                        get_block(FC_offdiag, 2, 2) -= -nu_E * inv_eps * nmass;             // Ez equation, [E]
                    }
                } /* end for (auto& fqp : f_qps) */
            } /*end if (has_neighbour) */
            else
            {   /* On a boundary*/
                ctx.offdiag_neigh_offsets[offdiag_contrib_i] = std::make_pair(0, false);
                /* Default use centered fluxes */
                T kappa_E = 0.5;
                T kappa_H = 0.5;
                T nu_E = 0.0;
                T nu_H = 0.0;

                if (ctx.cfg.upwind)
                {
                    kappa_E = 0.5*Y_this;
                    kappa_E = 0.5*Z_this;
                    nu_E = ctx.cfg.eta/(2*Y_this);
                    nu_H = ctx.cfg.eta/(2*Z_this);
                }

                auto f_qps = yq::integrate(ctx.msh, fc, 2*ctx.cfg.degree);
                for (auto& fqp : f_qps)
                {
                    auto ep = fqp.point();

                    /* Basis evaluated on myself */
                    auto tphi  = tbasis.eval(ep);
                    auto tmass = fqp.weight() * tphi * trans(tphi);

                    /* Centered */
                    get_block(FC_diag, 0, 2) += +ny * 2 * kappa_E * inv_mu * tmass;     // Hx equation, [E]
                    get_block(FC_diag, 1, 2) += -nx * 2 * kappa_E * inv_mu * tmass;     // Hy equation, [E]

                    /* Upwind */
                    if (ctx.cfg.upwind)
                        get_block(FC_diag, 2, 2) += -2*nu_E * inv_eps * tmass;    // Ez equation, [E]
                }
            } /*end else (has_neighbour) */


            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    auto blk = get_block(FC_diag, i, j);
                    get_global_ondiag_block(i, j) += solve(M2d, blk);
                }
            }

            if (has_neighbour)
            {   /* Save offdiag */
                auto offdiag_base = 3*basis_size*offdiag_contrib_i;

                for (size_t i = 0; i < 3; i++)
                {
                    for (size_t j = 0; j < 3; j++)
                    {
                        auto offset_i = offdiag_base + basis_size*i;
                        auto offset_j = basis_size*j;
                        auto blk = get_block(FC_offdiag, i, j);
                        submatrix(ctx.gOp_offdiag, offset_i, offset_j, basis_size, basis_size) = solve(M2d, blk);
                    }
                }
            }

            /* LAST */
            offdiag_contrib_i++;
        } // for (auto& fc : fcs)

        /* LAST */
        cell_i++;
    } // for (auto& tcl : ctx.msh.cells)

    auto asm_end_time = std::chrono::system_clock::now();

    if ( LOGLEVEL_INFO(ctx.cfg.verbosity) )
    {
        std::chrono::duration<double> asmtime = asm_end_time - asm_start_time;
        std::cout << "Assembly time: " << asmtime.count() << " seconds" << std::endl;
    }
}

template<typename Mesh>
using ICF_type = std::function<typename Mesh::coordinate_type(const typename Mesh::point_type&, typename Mesh::coordinate_type)>;

template<typename Mesh>
void
apply_initial_condition(maxwell_context<Mesh>& ctx, const ICF_type<Mesh>& Hx_ic,
                        const ICF_type<Mesh>& Hy_ic, const ICF_type<Mesh>& Ez_ic)
{
    using namespace blaze;
    namespace yb = yaourt::bases;
    namespace yq = yaourt::quadratures;

    using T = typename Mesh::coordinate_type;
    auto basis_size = yb::scalar_basis_size(ctx.cfg.degree, 2);

    size_t cell_i = 0;
    for (auto& tcl : ctx.msh.cells)
    {
        auto tbasis = yb::make_basis(ctx.msh, tcl, ctx.cfg.degree);
        DynamicVector<T> loc_rhs_Hx(basis_size, 0.0);
        DynamicVector<T> loc_rhs_Hy(basis_size, 0.0);
        DynamicVector<T> loc_rhs_Ez(basis_size, 0.0);

        auto qps = yq::integrate(ctx.msh, tcl, 2*ctx.cfg.degree);
        for(auto& qp : qps)
        {
            auto phi = tbasis.eval(qp.point());
            loc_rhs_Hx += Hx_ic(qp.point(), 0.0) * qp.weight() * phi;
            loc_rhs_Hy += Hy_ic(qp.point(), 0.0) * qp.weight() * phi;
            loc_rhs_Ez += Ez_ic(qp.point(), 0.0) * qp.weight() * phi;
        }

        auto gM_offset = cell_i * basis_size;
        auto M2d = submatrix(ctx.gM, gM_offset, 0, basis_size, basis_size);

        auto gDofs_base = 3 * cell_i * basis_size;
        subvector(ctx.gDofs, gDofs_base, basis_size) = solve(M2d, loc_rhs_Hx);
        subvector(ctx.gDofs, gDofs_base + basis_size, basis_size) = solve(M2d, loc_rhs_Hy);
        subvector(ctx.gDofs, gDofs_base + 2*basis_size, basis_size) = solve(M2d, loc_rhs_Ez);

        /* LAST */
        cell_i++;
    }
}

template<typename Mesh>
void
do_timestep(maxwell_context<Mesh>& ctx)
{
    using namespace blaze;
    namespace yb = yaourt::bases;

    using T = typename Mesh::coordinate_type;
    auto basis_size = yb::scalar_basis_size(ctx.cfg.degree, 2);

    auto get_dofs = [&](DynamicVector<T>& v, size_t elem) -> auto {
        return subvector(v, 3*basis_size*elem, 3*basis_size);
    };

    auto get_ondiag = [&](size_t index) -> auto {
        auto size = 3*basis_size;
        auto offset = size*index;
        return submatrix(ctx.gOp_ondiag, offset, 0, size, size);
    };

    auto get_offdiag = [&](size_t index) -> auto {
        auto size = 3*basis_size;
        auto offset = size*index;
        return submatrix(ctx.gOp_offdiag, offset, 0, size, size);
    };

    // Total FLOPS: (2*(3*basis_size)^2)*(faces_per_elem+1)*ctx.msh.cells.size()
    auto apply_operator = [&](DynamicVector<T>& v, DynamicVector<T>& v_next) -> void {
        size_t cell_i = 0;
        size_t offdiag_contrib_i = 0;

        for (auto& tcl : ctx.msh.cells)
        {
            // 2*(3*basis_size)^2 FLOPS
            get_dofs(v_next, cell_i) = get_ondiag(cell_i) * get_dofs(v, cell_i);

//#define BE_NAIVE
#ifdef BE_NAIVE
            auto fcs = faces(ctx.msh, tcl);
            for (auto& fc : fcs)
            {
                auto nv = neighbour_via(ctx.msh, tcl, fc);
                auto ncl = nv.first;
                if (has_neighbour)
                {
                    auto neigh_ofs = offset(ctx.msh, ncl);
                    get_dofs(v_next, cell_i) += get_offdiag(offdiag_contrib_i) * get_dofs(v, neigh_ofs);
                
                }
                /* LAST */
                offdiag_contrib_i++;
            }
#else
            // (2*(3*basis_size)^2)*faces_per_elem FLOPS
            for (size_t fi = 0; fi < ctx.faces_per_elem(); fi++)
            {
                auto no = ctx.offdiag_neigh_offsets[offdiag_contrib_i];
                if (no.second)
                    get_dofs(v_next, cell_i) += get_offdiag(offdiag_contrib_i) * get_dofs(v, no.first);

                /* LAST */
                offdiag_contrib_i++;
            }
#endif
            /* LAST */
            cell_i++;
        }
    };

    auto ts_start_time = std::chrono::system_clock::now();

    /* Here ctx.gDofs_t_plus_one is used as scratch space */
    if (ctx.cfg.time_integrator == time_integrator_type::EXPLICIT_EULER)
    {
        apply_operator(ctx.gDofs, ctx.gDofs_t_plus_one);
        ctx.gDofs_t_plus_one = ctx.gDofs + ctx.cfg.delta_t*ctx.gDofs_t_plus_one;
    }

    if (ctx.cfg.time_integrator == time_integrator_type::RUNGE_KUTTA_4)
    {
        auto size = ctx.gDofs.size();
        blaze::DynamicVector<T> k1(size);
        blaze::DynamicVector<T> k2(size);
        blaze::DynamicVector<T> k3(size);
        blaze::DynamicVector<T> k4(size);

        apply_operator(ctx.gDofs, k1);

        ctx.gDofs_t_plus_one = ctx.gDofs + (ctx.cfg.delta_t/2.)*k1; //2*(3*basis_size*msh.cells.size()) FLOPS
        apply_operator(ctx.gDofs_t_plus_one, k2);
        
        ctx.gDofs_t_plus_one = ctx.gDofs + (ctx.cfg.delta_t/2.)*k2; //2*(3*basis_size*msh.cells.size()) FLOPS
        apply_operator(ctx.gDofs_t_plus_one, k3);
        
        ctx.gDofs_t_plus_one = ctx.gDofs + (ctx.cfg.delta_t)*k3;    //2*(3*basis_size*msh.cells.size()) FLOPS
        apply_operator(ctx.gDofs_t_plus_one, k4);

        // 6 * (3*basis_size*msh.cells.size()) FLOPS
        ctx.gDofs_t_plus_one = ctx.gDofs + ((1./6.) * ctx.cfg.delta_t) * (k1 + 2*(k2 + k3) + k4);
    }

    auto ts_end_time = std::chrono::system_clock::now();

    if ( LOGLEVEL_DETAIL(ctx.cfg.verbosity) )
    {
        std::chrono::duration<double> tstime = ts_end_time - ts_start_time;
        double time = tstime.count();
        std::cout << "Timestep time: " << time << " seconds. ";

        size_t totflops;
        totflops  = 4*(2*(3*basis_size)*(3*basis_size)*(ctx.faces_per_elem()+1))*ctx.msh.cells.size(); //operator evaluation
        totflops += 12*(3*basis_size*ctx.msh.cells.size()); // RK4 vector-scalar multiplications

        std::cout << "Estimated performance: " << double(totflops)/time << std::endl;
    }

    swap(ctx.gDofs_t_plus_one, ctx.gDofs);
}

} // namespace yaourt::maxwell_2D


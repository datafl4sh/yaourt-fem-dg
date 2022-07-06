/*
 * Yaourt-FEM-DG - Yet AnOther Useful Resource for Teaching FEM and DG.
 *
 * Matteo Cicuttin (C) 2019, 2020, 2021, 2022
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

#include <vector>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/quadratures.hpp"
#include "core/bases.hpp"

#include "blaze/Math.h"

#include "core/blaze_sparse_init.hpp"

template<bool mixed>
struct hho_flavour {
    size_t degree;
    hho_flavour(size_t d) : degree(d) {}
};

using equal_order = hho_flavour<false>;
using mixed_order = hho_flavour<true>;

class hho_degree_info
{
    size_t      fd, cd, rd;
    bool        mixedord;
    
public:
    hho_degree_info()
        : fd(0), cd(0), rd(1), mixedord(false)
    {}
    
    hho_degree_info(size_t d)
        : fd(d), cd(d), rd(d+1), mixedord(false)
    {}
    
    hho_degree_info(const equal_order& eo)
        : fd(eo.degree), cd(eo.degree), rd(eo.degree+1), mixedord(false)
    {}
    
    hho_degree_info(const mixed_order& mo)
        : fd(mo.degree), cd(mo.degree+1), rd(mo.degree+1), mixedord(true)
    {}
    
    size_t  face_degree() const { return fd; }
    size_t  cell_degree() const { return cd; }
    size_t  reconstruction_degree() const { return rd; }
    bool    mixed_order() const { return mixedord; }
};

template<typename Mesh, typename Element, typename Basis>
blaze::DynamicMatrix<typename Mesh::coordinate_type>
make_stiffness_matrix(const Mesh& msh,
                      const Element& elem,
                      const Basis& basis)
{
    using T = typename Mesh::coordinate_type;
    
    auto bd = basis.degree();
    auto bs = basis.size();
    
    blaze::DynamicMatrix<T> K(bs, bs, 0.0);
    
    auto qps = yaourt::quadratures::integrate(msh, elem, 2*bd);
    for (auto& qp : qps)
    {
        auto dphi = basis.eval_grads(qp.point());
        K += qp.weight() * dphi * trans(dphi);
    }
    
    return K;
}

template<typename Mesh, typename Element, typename Basis, typename Function>
blaze::DynamicVector<typename Mesh::coordinate_type>
project(const Mesh& msh,
        const Element& elem,
        const Basis& basis,
        const Function& f)
{
    using T = typename Mesh::coordinate_type;
    
    auto bd = basis.degree();
    auto bs = basis.size();
    
    blaze::DynamicMatrix<T> M(bs, bs, 0.0);
    blaze::DynamicVector<T> rhs(bs, 0.0);
    
    auto qps = yaourt::quadratures::integrate(msh, elem, 2*bd+1);
    for (auto& qp : qps)
    {
        auto phi = basis.eval(qp.point());
        M += qp.weight() * phi * trans(phi);
        rhs += qp.weight() * f(qp.point()) * phi;
    }
    
    blaze::DynamicVector<T> proj = blaze::solve_LU(M, rhs);
    
    return proj;
}

template<typename Mesh, typename Function>
blaze::DynamicVector<typename Mesh::coordinate_type>
project_hho(const Mesh& msh,
            const typename Mesh::cell_type& cl,
            const hho_degree_info& hdi,
            const Function& f)
{
    using T = typename Mesh::coordinate_type;
    
    auto fd = hdi.face_degree();
    auto cd = hdi.cell_degree();
    
    auto fbs = yaourt::bases::scalar_basis_size(fd, 1);
    auto cbs = yaourt::bases::scalar_basis_size(cd, 2);
    
    auto fcs = faces(msh, cl);
    auto nf = fcs.size();
    
    blaze::DynamicVector<T> hho_proj(cbs + nf*fbs, 0.0);

    auto cb = yaourt::bases::make_basis(msh, cl, cd);
    subvector(hho_proj, 0, cbs) = project(msh, cl, cb, f);
    
    for (size_t fc_i = 0; fc_i < nf; fc_i++)
    {
        size_t ofs = cbs + fc_i*fbs;
        const auto& fc = fcs[fc_i];
        auto fb = yaourt::bases::make_basis(msh, fc, fd);
        
        subvector(hho_proj, ofs, fbs) = project(msh, fc, fb, f);
    }
    
    return hho_proj;
}

template<typename Mesh>
blaze::DynamicMatrix<typename Mesh::coordinate_type>
make_hho_gradient_reconstruction(const Mesh& msh,
                                 const typename Mesh::cell_type& cl,
                                 const hho_degree_info& hdi)
{
    using T = typename Mesh::coordinate_type;
    
    /* Degrees for each space */
    auto cd = hdi.cell_degree();
    auto fd = hdi.face_degree();
    auto rd = hdi.reconstruction_degree();
    
    /* Basis sizes */
    auto fbs = yaourt::bases::scalar_basis_size(fd, 1);
    auto cbs = yaourt::bases::scalar_basis_size(cd, 2);
    auto rbs = yaourt::bases::scalar_basis_size(rd, 2);
    
    /* Cell and reconstruction space bases */
    auto cbasis = yaourt::bases::make_basis(msh, cl, cd);
    auto rbasis = yaourt::bases::make_basis(msh, cl, rd);
    
    auto fcs = faces(msh, cl);
    auto nf = fcs.size();
    
    blaze::DynamicMatrix<T> K(rbs, rbs, 0.0);
    blaze::DynamicMatrix<T> oper_lhs(rbs-1, rbs-1, 0.0);
    blaze::DynamicMatrix<T> oper_rhs(rbs-1, cbs + nf*fbs, 0.0);
    
    auto qps = yaourt::quadratures::integrate(msh, cl, 2*(rd-1));
    for (auto& qp : qps)
    {
        auto ep   = qp.point();
        auto r_dphi = rbasis.eval_grads(ep);
        K += qp.weight() * r_dphi * trans(r_dphi);
    }
    
    submatrix(oper_lhs, 0, 0, rbs-1, rbs-1) = submatrix(K, 1, 1, rbs-1, rbs-1);
    submatrix(oper_rhs, 0, 0, rbs-1, cbs) = submatrix(K, 1, 0, rbs-1, cbs);
    
    for (size_t fc_i = 0; fc_i < nf; fc_i++)
    {
        size_t ofs = cbs + fc_i*fbs;
        const auto& fc = fcs[fc_i];
        auto fbasis = yaourt::bases::make_basis(msh, fc, fd);
        auto n = normal(msh, cl, fc);

        auto fqps = yaourt::quadratures::integrate(msh, fc, (rd-1)+std::max(cd,fd));
        for (auto& fqp : fqps)
        {
            auto ep         = fqp.point();
            auto f_phi      = fbasis.eval(ep);
            auto c_phi      = cbasis.eval(ep);
            auto r_dphi     = rbasis.eval_grads(ep);
            auto r_dphi_n2  = r_dphi * n;
            auto r_dphi_n   = subvector(r_dphi_n2, 1, rbs-1);
            
            submatrix(oper_rhs, 0, 0, rbs-1, cbs) -=
                fqp.weight() * r_dphi_n * trans(c_phi);
            
            submatrix(oper_rhs, 0, ofs, rbs-1, fbs) +=
                fqp.weight() * r_dphi_n * trans(f_phi);
        }
    }
    
    return blaze::solve_LU(oper_lhs, oper_rhs);
}

template<typename Mesh>
blaze::DynamicMatrix<typename Mesh::coordinate_type>
make_hho_stabilization(const Mesh& msh,
                       const typename Mesh::cell_type& cl,
                       const hho_degree_info& hdi)
{
    using T = typename Mesh::coordinate_type;
    
    /* Degrees for each space */
    auto cd = hdi.cell_degree();
    auto fd = hdi.face_degree();
    
    /* Basis sizes */
    auto fbs = yaourt::bases::scalar_basis_size(fd, 1);
    auto cbs = yaourt::bases::scalar_basis_size(cd, 2);
    
    /* Cell and reconstruction space bases */
    auto cbasis = yaourt::bases::make_basis(msh, cl, cd);
    
    auto fcs = faces(msh, cl);
    auto nf = fcs.size();
    
    blaze::DynamicMatrix<T> oper(cbs + nf*fbs, cbs + nf*fbs, 0.0);
    auto ht = diameter(msh, cl);
    
    for (size_t fc_i = 0; fc_i < nf; fc_i++)
    {
        size_t ofs = cbs + fc_i*fbs;
        const auto& fc = fcs[fc_i];
        
        blaze::DynamicMatrix<T> oper_rhs(fbs, cbs + nf*fbs, 0.0);
        blaze::DynamicMatrix<T> mass(fbs, fbs, 0.0);
        blaze::DynamicMatrix<T> trace(fbs, cbs, 0.0);
    
        auto I = blaze::IdentityMatrix<T>(fbs);
        submatrix(oper_rhs, 0, ofs, fbs, fbs) = -I;
        
        auto fbasis = yaourt::bases::make_basis(msh, fc, fd);

        auto fqps = yaourt::quadratures::integrate(msh, fc, cd+fd);
        for (auto& fqp : fqps)
        {
            auto ep     = fqp.point();
            auto f_phi  = fbasis.eval(ep);
            auto c_phi  = cbasis.eval(ep);
            
            mass    += fqp.weight() * f_phi * trans(f_phi);
            trace   += fqp.weight() * f_phi * trans(c_phi);
        }
        
        submatrix(oper_rhs, 0, 0, fbs, cbs) = blaze::solve_LU(mass, trace);
        
        oper += trans(oper_rhs) * mass * oper_rhs / ht;
    }
    
    return oper;
}

template<typename Mesh>
blaze::DynamicMatrix<typename Mesh::coordinate_type>
make_hho_stabilization_xxx(const Mesh& msh,
                       const typename Mesh::cell_type& cl,
                       blaze::DynamicMatrix<typename Mesh::coordinate_type>& R,
                       const hho_degree_info& hdi)
{
    using T = typename Mesh::coordinate_type;
    
    /* Degrees for each space */
    auto cd = hdi.cell_degree();
    auto fd = hdi.face_degree();
    auto rd = hdi.reconstruction_degree();
    
    /* Basis sizes */
    auto fbs = yaourt::bases::scalar_basis_size(fd, 1);
    auto cbs = yaourt::bases::scalar_basis_size(cd, 2);
    auto rbs = yaourt::bases::scalar_basis_size(rd, 2);
    
    /* Cell and reconstruction space bases */
    auto cbasis = yaourt::bases::make_basis(msh, cl, cd);
    auto rbasis = yaourt::bases::make_basis(msh, cl, rd);
    
    auto fcs = faces(msh, cl);
    auto nf = fcs.size();
    
    blaze::DynamicMatrix<T> oper(cbs + nf*fbs, cbs + nf*fbs, 0.0);
    auto ht = diameter(msh, cl);
    
    blaze::DynamicMatrix<T> CT(cbs, rbs, 0.0);
    auto qps = yaourt::quadratures::integrate(msh, cl, cd+rd);
    for (auto& qp : qps)
    {
        auto r_phi = rbasis.eval(qp.point());
        auto c_phi = subvector(r_phi, 0, cbs);
        CT += qp.weight() * c_phi * trans(r_phi);
    }
    
    blaze::DynamicMatrix<T> evalRC = submatrix(CT, 0, 1, cbs, rbs-1);
    blaze::DynamicMatrix<T> Cmass  = submatrix(CT, 0, 0, cbs, cbs);
    
    blaze::DynamicMatrix<T> evR = evalRC * R;
    blaze::DynamicMatrix<T> projRT = - blaze::solve_LU(Cmass, evR);
    submatrix(projRT, 0, 0, cbs, cbs) += blaze::IdentityMatrix<T>(cbs);
    
    for (size_t fc_i = 0; fc_i < nf; fc_i++)
    {
        size_t ofs = cbs + fc_i*fbs;
        const auto& fc = fcs[fc_i];
        
        blaze::DynamicMatrix<T> oper_rhs(fbs, cbs + nf*fbs, 0.0);
        blaze::DynamicMatrix<T> Fmass(fbs, fbs, 0.0);
        blaze::DynamicMatrix<T> Ftrace(fbs, rbs, 0.0);
        
        auto fbasis = yaourt::bases::make_basis(msh, fc, fd);

        auto fqps = yaourt::quadratures::integrate(msh, fc, rd+fd);
        for (auto& fqp : fqps)
        {
            auto ep     = fqp.point();
            auto f_phi  = fbasis.eval(ep);
            auto r_phi  = rbasis.eval(ep);
            
            Fmass   += fqp.weight() * f_phi * trans(f_phi);
            Ftrace  += fqp.weight() * f_phi * trans(r_phi);
        }
        
        blaze::DynamicMatrix<T> F = submatrix(Ftrace, 0, 1, fbs, rbs-1)*R;
        F += submatrix(Ftrace, 0, 0, fbs, cbs)*projRT;
        
        oper_rhs = blaze::solve_LU(Fmass, F);
        
        auto I = blaze::IdentityMatrix<T>(fbs);
        submatrix(oper_rhs, 0, ofs, fbs, fbs) -= I;
        
        oper += trans(oper_rhs) * Fmass * oper_rhs / ht;
    }
    
    return oper;
}

template<typename Mesh>
blaze::DynamicMatrix<typename Mesh::coordinate_type>
make_hho_stabilization(const Mesh& msh,
                       const typename Mesh::cell_type& cl,
                       blaze::DynamicMatrix<typename Mesh::coordinate_type>& R,
                       const hho_degree_info& hdi)
{
    using T = typename Mesh::coordinate_type;
    
    /* Degrees for each space */
    auto cd = hdi.cell_degree();
    auto fd = hdi.face_degree();
    auto rd = hdi.reconstruction_degree();
    
    /* Basis sizes */
    auto fbs = yaourt::bases::scalar_basis_size(fd, 1);
    auto cbs = yaourt::bases::scalar_basis_size(cd, 2);
    auto rbs = yaourt::bases::scalar_basis_size(rd, 2);
    
    /* Cell and reconstruction space bases */
    auto cbasis = yaourt::bases::make_basis(msh, cl, cd);
    auto rbasis = yaourt::bases::make_basis(msh, cl, rd);
    
    auto fcs = faces(msh, cl);
    auto nf = fcs.size();
    
    blaze::DynamicMatrix<T> oper(cbs + nf*fbs, cbs + nf*fbs, 0.0);
    auto ht = diameter(msh, cl);
    
    blaze::DynamicMatrix<T> CT(cbs, rbs, 0.0);
    auto qps = yaourt::quadratures::integrate(msh, cl, cd+rd);
    for (auto& qp : qps)
    {
        auto r_phi = rbasis.eval(qp.point());
        auto c_phi = subvector(r_phi, 0, cbs);
        CT += qp.weight() * c_phi * trans(r_phi);
    }
    
    blaze::DynamicMatrix<T> evalRC = submatrix(CT, 0, 1, cbs, rbs-1);
    blaze::DynamicMatrix<T> Cmass  = submatrix(CT, 0, 0, cbs, cbs);
    
    blaze::DynamicMatrix<T> evR = -evalRC * R;
    blaze::DynamicMatrix<T> projRT = blaze::solve_LU(Cmass, evR);
    submatrix(projRT, 0, 0, cbs, cbs) += blaze::IdentityMatrix<T>(cbs);
    
    for (size_t fc_i = 0; fc_i < nf; fc_i++)
    {
        size_t ofs = cbs + fc_i*fbs;
        const auto& fc = fcs[fc_i];
        blaze::DynamicMatrix<T> oper_rhs(fbs, cbs + nf*fbs, 0.0);
        blaze::DynamicMatrix<T> Fmass(fbs, fbs, 0.0);
        blaze::DynamicMatrix<T> Ftrace(fbs, rbs, 0.0);
        
        auto fbasis = yaourt::bases::make_basis(msh, fc, fd);

        auto fqps = yaourt::quadratures::integrate(msh, fc, rd+fd);
        for (auto& fqp : fqps)
        {
            auto ep     = fqp.point();
            auto f_phi  = fbasis.eval(ep);
            auto r_phi  = rbasis.eval(ep);
            
            Fmass   += fqp.weight() * f_phi * trans(f_phi);
            Ftrace  += fqp.weight() * f_phi * trans(r_phi);
        }
        
        blaze::DynamicMatrix<T> F = submatrix(Ftrace, 0, 1, fbs, rbs-1)*R;
        F += submatrix(Ftrace, 0, 0, fbs, cbs)*projRT;
        
        oper_rhs = blaze::solve_LU(Fmass, F);
        
        auto I = blaze::IdentityMatrix<T>(fbs);
        submatrix(oper_rhs, 0, ofs, fbs, fbs) -= I;
        
        oper += trans(oper_rhs) * Fmass * oper_rhs / ht;
    }
    
    return oper;
}
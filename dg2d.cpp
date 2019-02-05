#include <iostream>
#include <fstream>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/quadratures.hpp"
#include "core/bases.hpp"
#include "core/solvers.hpp"
#include "core/blaze_sparse_init.hpp"
#include "core/dataio.hpp"


template<typename Mesh>
class assembler
{
    using T = typename Mesh::coordinate_type;
    using triplet_type = blaze::triplet<T>;

    std::vector<triplet_type>       triplets;

    size_t                          sys_size, basis_size;

public:
    blaze::CompressedMatrix<T>      lhs;
    blaze::DynamicVector<T>         rhs;

    assembler()
    {}

    assembler(const Mesh& msh, size_t degree)
    {
        basis_size = dg2d::bases::scalar_basis_size(degree,2);
        sys_size = basis_size * msh.cells.size();

        lhs.resize( sys_size, sys_size );
        rhs.resize( sys_size );
    }

    bool assemble(const Mesh& msh,
                  const typename Mesh::cell_type& cl_a,
                  const typename Mesh::cell_type& cl_b,
                  const blaze::DynamicMatrix<T>& local_rhs)
    {
        auto cl_a_ofs = offset(msh, cl_a) * basis_size;
        auto cl_b_ofs = offset(msh, cl_b) * basis_size;

        for (size_t i = 0; i < basis_size; i++)
        {
            auto ci = cl_a_ofs + i;

            for (size_t j = 0; j < basis_size; j++)
            {
                auto cj = cl_b_ofs + j;
                triplets.push_back( {ci, cj, local_rhs(i,j)} );
            }
        }
        
        return true;
    }

    bool assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const blaze::DynamicMatrix<T>& local_rhs,
                  const blaze::DynamicVector<T>& local_lhs)
    {
        auto cl_ofs = offset(msh, cl) * basis_size;

        for (size_t i = 0; i < basis_size; i++)
        {
            auto ci = cl_ofs + i;

            for (size_t j = 0; j < basis_size; j++)
            {
                auto cj = cl_ofs + j;
                triplets.push_back( {ci, cj, local_rhs(i,j)} );
            }

            rhs[ci] = local_lhs[i];
        }
        
        return true;
    }

    void finalize()
    {
        blaze::init_from_triplets(lhs, triplets.begin(), triplets.end());
        triplets.clear();
    }

    size_t system_size() const { return sys_size; }
};

int main(void)
{
    using T = double;
    using mesh_type = dg2d::simplicial_mesh<T>;

    mesh_type msh;
    auto mesher = dg2d::get_mesher(msh);

    mesher.create_mesh(msh,4);

    msh.compute_connectivity();

    size_t degree = 2;
    T eta = 10*degree*degree;

    auto rhs_fun = [](const typename mesh_type::point_type& pt) -> auto {
        auto sx = std::sin(M_PI*pt.x());
        auto sy = std::sin(M_PI*pt.y());

        return 2.0 * M_PI * M_PI * sx * sy;
    };


    assembler<mesh_type> assm(msh, degree);


    std::ofstream ofs("basis.dat");

    for (auto& tcl : msh.cells)
    {
        auto qps = dg2d::quadratures::integrate(msh, tcl, 2*degree+2);
        auto tbasis = dg2d::bases::make_basis(msh, tcl, degree);

        blaze::DynamicMatrix<T> K(tbasis.size(), tbasis.size(), 0.0);
        blaze::DynamicVector<T> loc_rhs(tbasis.size(), 0.0);

        for (auto& qp : qps)
        {
            auto ep   = qp.point();
            auto phi  = tbasis.eval(ep);
            auto dphi = tbasis.eval_grads(ep);

            K += qp.weight() * dphi * trans(dphi);

            loc_rhs += qp.weight() * rhs_fun(ep) * phi;
        }

        assm.assemble(msh, tcl, K, loc_rhs);

        auto fcs = faces(msh, tcl);
        for (auto& fc : fcs)
        {
            blaze::DynamicMatrix<T> Att(tbasis.size(), tbasis.size(), 0.0);
            blaze::DynamicMatrix<T> Atn(tbasis.size(), tbasis.size(), 0.0);
            blaze::DynamicMatrix<T> Ant(tbasis.size(), tbasis.size(), 0.0);
            blaze::DynamicMatrix<T> Ann(tbasis.size(), tbasis.size(), 0.0);

            auto nv = neighbour_via(msh, tcl, fc);
            auto ncl = nv.first;
            auto nbasis = dg2d::bases::make_basis(msh, ncl, degree);
            assert(tbasis.size() == nbasis.size());

            auto n       = normal(msh, tcl, fc);
            auto eta_l  = eta / diameter(msh, fc);
            auto f_qps = dg2d::quadratures::integrate(msh, fc, 2*degree+2);
            
            for (auto& fqp : f_qps)
            {
                auto ep     = fqp.point();
                auto tphi   = tbasis.eval(ep);
                auto tdphi  = tbasis.eval_grads(ep);

                if (nv.second)
                {
                    Att  = + fqp.weight() * eta_l * tphi * trans(tphi);
                    Att += - fqp.weight() * 0.5 * tphi * trans(tdphi*n);
                    Att += - fqp.weight() * 0.5 * (tdphi*n) * trans(tphi);
                }
                else
                {
                    Att  = + fqp.weight() * eta_l * tphi * trans(tphi);
                    Att += - fqp.weight() * tphi * trans(tdphi*n);
                    Att += - fqp.weight() * (tdphi*n) * trans(tphi);
                    continue;
                }

                auto nphi   = nbasis.eval(ep);
                auto ndphi  = nbasis.eval_grads(ep);

                Atn  = - fqp.weight() * eta_l * tphi * trans(nphi);
                Atn += - fqp.weight() * 0.5 * tphi * trans(ndphi*n);
                Atn += + fqp.weight() * 0.5 * (tdphi*n) * trans(nphi); 

                //Ant  = - fqp.weight() * eta_l * nphi * trans(tphi);
                //Ant += + fqp.weight() * 0.5 * (ndphi*n) * trans(tphi);
                //Ant += - fqp.weight() * 0.5 * nphi * trans(tdphi*n);

                //Ann  = + fqp.weight() * eta_l * nphi * trans(nphi);
                //Ann += + fqp.weight() * 0.5 * (ndphi*n) * trans(nphi);
                //Ann += + fqp.weight() * 0.5 * nphi * trans(ndphi*n);
            }

            assm.assemble(msh, tcl, tcl, Att);
            if (nv.second)
            {
                assm.assemble(msh, tcl, ncl, Atn);
                //assm.assemble(msh, ncl, tcl, Ant);
                //assm.assemble(msh, ncl, ncl, Ann);
            }
        }
    }

    assm.finalize();

    blaze::DynamicVector<T> sol(assm.system_size());

    conjugated_gradient_params<T> cgp;
    cgp.verbose = true;
    cgp.rr_max = 10000;
    cgp.max_iter = 2*assm.system_size();

    conjugated_gradient(cgp, assm.lhs, assm.rhs, sol);
    //qmr(assm.lhs, assm.rhs, sol);

    //blaze::DynamicMatrix<T> A(assm.lhs);
    //sol = inv(A)*assm.rhs;

    blaze::DynamicVector<T> var(msh.cells.size());
    auto bs = dg2d::bases::scalar_basis_size(degree, 2);
    for (size_t i = 0; i < msh.cells.size(); i++)
    {
        var[i] = sol[bs*i];
    }

#ifdef WITH_SILO
    dg2d::silo_database silo;
    silo.create("test_dg.silo");

    silo.add_mesh(msh, "test_mesh");

    silo.add_zonal_variable("test_mesh", "solution", var);
#endif


    return 0;
}

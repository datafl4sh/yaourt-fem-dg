#include <iostream>
#include <fstream>

#include <unistd.h>

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
    std::vector<triplet_type>       pc_triplets;

    size_t                          sys_size, basis_size;    
    bool                            build_pc;

public:
    blaze::CompressedMatrix<T>      lhs;
    blaze::DynamicVector<T>         rhs;
    blaze::CompressedMatrix<T>      pc;
    blaze::DynamicVector<T>         pc_temp;

    assembler()
        : build_pc(false)
    {}

    assembler(const Mesh& msh, size_t degree, bool bpc = false)
        : build_pc(bpc)
    {
        basis_size = dg2d::bases::scalar_basis_size(degree,2);
        sys_size = basis_size * msh.cells.size();

        lhs.resize( sys_size, sys_size );
        rhs.resize( sys_size );
        pc.resize( sys_size, sys_size );
        pc_temp = blaze::DynamicVector<T>(sys_size, 0.0);
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

                if (build_pc && ci == cj)
                    pc_temp[ci] += local_rhs(i,j);

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

                if (build_pc && ci == cj)
                    pc_temp[ci] += local_rhs(i,j);
            }

            rhs[ci] = local_lhs[i];
        }
        
        return true;
    }

    void finalize()
    {

        blaze::init_from_triplets(lhs, triplets.begin(), triplets.end());
        triplets.clear();

        if (build_pc)
        {
            for(size_t i = 0; i < pc_temp.size(); i++)
            {
                assert( std::abs(pc_temp[i]) > 1e-2 );
                pc_triplets.push_back({i,i,1.0/pc_temp[i]});
            }

            blaze::init_from_triplets(pc, pc_triplets.begin(), pc_triplets.end());
            pc_triplets.clear();
        }
    }

    size_t system_size() const { return sys_size; }
};



template<typename T>
struct dg_config
{
    T       eta;
    int     degree;
    int     ref_levels;
    bool    use_preconditioner;
    bool    shatter;

    dg_config()
        : eta(0.0), degree(1), ref_levels(4), use_preconditioner(false)
    {}
};



template<typename Mesh>
void run_dg(Mesh& msh, const dg_config<typename Mesh::coordinate_type>& cfg)
{
#define DEBUG
    using mesh_type = Mesh;
    using T = typename mesh_type::coordinate_type;


    msh.compute_connectivity();

    size_t degree = cfg.degree;
    T eta = 70*degree*degree;

    auto rhs_fun = [](const typename mesh_type::point_type& pt) -> auto {
        auto sx = std::sin(M_PI*pt.x());
        auto sy = std::sin(M_PI*pt.y());

        return 2.0 * M_PI * M_PI * sx * sy;
    };

#ifdef DEBUG
    auto sol_fun = [](const typename mesh_type::point_type& pt) -> auto {
        auto sx = std::sin(M_PI*pt.x());
        auto sy = std::sin(M_PI*pt.y());

        return pt.x();//sx * sy;
    };
#endif /* DEBUG */

    assembler<mesh_type> assm(msh, degree, cfg.use_preconditioner);


    std::ofstream ofs("basis.dat");

    T c_err = 0.0;

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

        //blaze::DynamicMatrix<T> diag = K;
        auto fcs = faces(msh, tcl);
        for (auto& fc : fcs)
        {
            blaze::DynamicMatrix<T> Att(tbasis.size(), tbasis.size(), 0.0);
            blaze::DynamicMatrix<T> Atn(tbasis.size(), tbasis.size(), 0.0);
#ifdef DEBUG
            blaze::DynamicMatrix<T> Ant(tbasis.size(), tbasis.size(), 0.0);
            blaze::DynamicMatrix<T> Ann(tbasis.size(), tbasis.size(), 0.0);
#endif /* DEBUG */
            auto nv = neighbour_via(msh, tcl, fc);
            auto ncl = nv.first;
            auto nbasis = dg2d::bases::make_basis(msh, ncl, degree);
            assert(tbasis.size() == nbasis.size());

            auto n     = normal(msh, tcl, fc);
            auto eta_l = eta / diameter(msh, fc);
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
#ifdef DEBUG
                Ant  = - fqp.weight() * eta_l * nphi * trans(tphi);
                Ant += + fqp.weight() * 0.5 * (ndphi*n) * trans(tphi);
                Ant += - fqp.weight() * 0.5 * nphi * trans(tdphi*n);

                Ann  = + fqp.weight() * eta_l * nphi * trans(nphi);
                Ann += + fqp.weight() * 0.5 * (ndphi*n) * trans(nphi);
                Ann += + fqp.weight() * 0.5 * nphi * trans(ndphi*n);
#endif /* DEBUG */
            }

            assm.assemble(msh, tcl, tcl, Att);
            if (nv.second)
            {
                assm.assemble(msh, tcl, ncl, Atn);
                //assm.assemble(msh, ncl, tcl, Ant);
                //assm.assemble(msh, ncl, ncl, Ann);
            
#ifdef DEBUG
                blaze::DynamicMatrix<T> Mt(tbasis.size(), tbasis.size(), 0.0);
                blaze::DynamicMatrix<T> Mn(tbasis.size(), tbasis.size(), 0.0);
                blaze::DynamicVector<T> loc_rhs_t(tbasis.size(), 0.0);
                blaze::DynamicVector<T> loc_rhs_n(tbasis.size(), 0.0);

                for (auto& qp : qps)
                {
                    auto ep   = qp.point();
                    auto tphi  = tbasis.eval(ep);
                    auto nphi  = nbasis.eval(ep);

                    Mt += qp.weight() * tphi * trans(tphi);
                    Mn += qp.weight() * nphi * trans(tphi);

                    loc_rhs_t += qp.weight() * sol_fun(ep) * tphi;
                    loc_rhs_n += qp.weight() * sol_fun(ep) * nphi;
                }

                blaze::DynamicVector<T> t_proj = inv(Mt)*loc_rhs_t;
                blaze::DynamicVector<T> n_proj = inv(Mn)*loc_rhs_n;

                blaze::DynamicVector<T> res_t = Att*t_proj + Atn*n_proj;
                blaze::DynamicVector<T> res_n = Ant*t_proj + Ann*n_proj;

                std::cout << fc << ": " << dot(t_proj, res_t) + dot(n_proj,res_n) << std::endl;

#endif /* DEBUG */
            }
        }
    }

    assm.finalize();

#ifdef DEBUG
    std::cout << c_err << std::endl;
#endif /* DEBUG */


    blaze::DynamicVector<T> sol(assm.system_size());

    conjugated_gradient_params<T> cgp;
    cgp.verbose = true;
    cgp.rr_max = 10000;
    cgp.rr_tol = 1e-12;
    cgp.max_iter = 2*assm.system_size();

    if (cfg.use_preconditioner)
        conjugated_gradient(cgp, assm.lhs, assm.rhs, sol, assm.pc);
    else
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
#ifndef DEBUG
    auto sol_fun = [](const typename mesh_type::point_type& pt) -> auto {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };
#endif

    auto sol_grad = [](const typename mesh_type::point_type& pt) -> auto {
        blaze::StaticVector<T,2> grad(0.0);
        grad[0] = M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        grad[1] = M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());
        return grad;
    };

    T L2_errsq = 0.0;
    for (auto& cl : msh.cells)
    {
        auto basis = dg2d::bases::make_basis(msh, cl, degree);
        auto basis_size = basis.size();
        auto ofs = offset(msh, cl);

        blaze::DynamicVector<T> loc_sol(basis_size);
        for (size_t i = 0; i < basis_size; i++)
            loc_sol[i] = sol[basis_size * ofs + i];

        blaze::DynamicMatrix<T> M(basis_size, basis_size, 0.0);
        blaze::DynamicVector<T> a(basis_size, 0.0);

        auto qps = dg2d::quadratures::integrate(msh, cl, 2*degree);
        for (auto& qp : qps)
        {
            auto ep   = qp.point();
            auto phi  = basis.eval(ep);

            T val = dot(loc_sol, phi);
            L2_errsq += qp.weight() * (sol_fun(ep) - val) * (sol_fun(ep) - val);
        }
    }

    std::cout << "L2-norm error: " << std::sqrt(L2_errsq) << std::endl;

#ifdef WITH_SILO
    dg2d::silo_database silo;
    silo.create("test_dg.silo");

    silo.add_mesh(msh, "test_mesh");

    silo.add_zonal_variable("test_mesh", "solution", var);
#endif
}

template<typename T>
void run_triangle_dg(const dg_config<T>& cfg)
{
    using mesh_type = dg2d::simplicial_mesh<T>;

    mesh_type msh;
    auto mesher = dg2d::get_mesher(msh);
    mesher.create_mesh(msh, cfg.ref_levels);

    if (cfg.shatter)
        shatter_mesh(msh, 0.2);

    run_dg(msh, cfg);
}

template<typename T>
void run_quadrangle_dg(const dg_config<T>& cfg)
{
    using mesh_type = dg2d::quad_mesh<T>;

    mesh_type msh;
    auto mesher = dg2d::get_mesher(msh);
    mesher.create_mesh(msh, cfg.ref_levels);

    if (cfg.shatter)
        shatter_mesh(msh, 0.2);

    run_dg(msh, cfg);
}

enum class meshtype  {
    TRIANGULAR,
    QUADRANGULAR,
    TETRAHEDRAL,
    HEXAHEDRAL
};

int main(int argc, char **argv)
{
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);

    using T = double;

    meshtype mt = meshtype::TRIANGULAR;

    dg_config<T> cfg;

    int     ch;

    cfg.shatter = false;

    while ( (ch = getopt(argc, argv, "e:k:r:hqsvpS")) != -1 )
    {
        switch(ch)
        {
            case 'e':
                cfg.eta = atof(optarg);
                break;

            case 'k':
                cfg.degree = atoi(optarg);
                if (cfg.degree < 1)
                {
                    std::cout << "Degree must be positive. Falling back to 1." << std::endl;
                    cfg.degree = 1;
                }
                break;

            case 'r':
                cfg.ref_levels = atoi(optarg);
                if (cfg.ref_levels < 0)
                {
                    std::cout << "Degree must be positive. Falling back to 1." << std::endl;
                    cfg.ref_levels = 1;
                }
                break;

            case 'v':
                break;

            case 'q':
                mt = meshtype::QUADRANGULAR;
                break;

            case 's':
                mt = meshtype::TRIANGULAR;
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
        case meshtype::TRIANGULAR:
            run_triangle_dg(cfg);
            break;

        case meshtype::QUADRANGULAR:
            run_quadrangle_dg(cfg);
            break;

        case meshtype::TETRAHEDRAL:
        case meshtype::HEXAHEDRAL:
            break;
    }


    return 0;
}


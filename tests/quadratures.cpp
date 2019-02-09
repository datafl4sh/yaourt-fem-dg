#include <iostream>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/bases.hpp"
#include "core/quadratures.hpp"

template<typename Mesh>
std::pair<typename Mesh::coordinate_type, typename Mesh::coordinate_type>
compute_cell_errors(const Mesh& msh, size_t degree)
{
    using mesh_type = Mesh;
    using T = typename mesh_type::coordinate_type;

    auto test_fun = [](const typename mesh_type::point_type& pt) -> auto {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };

    auto test_grad = [](const typename mesh_type::point_type& pt) -> auto {
        blaze::StaticVector<T,2> grad(0.0);
        grad[0] = M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        grad[1] = M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());
        return grad;
    };

    T fun_err = 0.0;
    T grad_err = 0.0;
    for (auto& cl : msh.cells)
    {
        auto basis = dg2d::bases::make_basis(msh, cl, degree);
        auto basis_size = basis.size();

        blaze::DynamicMatrix<T> M(basis_size, basis_size, 0.0);
        blaze::DynamicMatrix<T> S(basis_size, basis_size, 0.0);
        blaze::DynamicVector<T> rhs(basis_size, 0.0);

        auto qps = dg2d::quadratures::integrate(msh, cl, 2*degree+2);
        for (auto& qp : qps)
        {
            auto ep   = qp.point();
            auto phi  = basis.eval(ep);

            auto temp = qp.weight() * phi;
            M   += temp * trans(phi);
            rhs += test_fun(ep) * temp;
        }

        blaze::DynamicVector<T> proj = inv(M)*rhs;

        for (auto& qp : qps)
        {
            auto ep   = qp.point();
            auto phi  = basis.eval(ep);
            auto dphi = basis.eval_grads(ep);

            auto val = dot(proj, phi);
            fun_err += qp.weight() * (test_fun(ep) - val) * (test_fun(ep) - val);

            blaze::StaticVector<T,2> grad(0.0);
            for (size_t i = 0; i < basis.size(); i++)
            {
                grad[0] += proj[i]*dphi(i,0);
                grad[1] += proj[i]*dphi(i,1);
            }
            blaze::StaticVector<T,2> gdiff = test_grad(ep) - grad;
            grad_err += qp.weight() * dot(gdiff,gdiff);
        }
    }

    return std::make_pair(std::sqrt(fun_err), std::sqrt(grad_err));
}

template<typename Mesh>
typename Mesh::coordinate_type
compute_face_error(const Mesh& msh, size_t degree)
{
    using mesh_type = Mesh;
    using T = typename mesh_type::coordinate_type;

    auto test_fun = [](const typename mesh_type::point_type& pt) -> auto {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };

    T fun_err = 0.0;
    for (auto& fc : msh.faces)
    {
        auto basis = dg2d::bases::make_basis(msh, fc, degree);
        auto basis_size = basis.size();

        blaze::DynamicMatrix<T> M(basis_size, basis_size, 0.0);
        blaze::DynamicVector<T> rhs(basis_size, 0.0);

        auto qps = dg2d::quadratures::integrate(msh, fc, 2*degree+2);
        for (auto& qp : qps)
        {
            auto ep   = qp.point();
            auto phi  = basis.eval(ep);

            auto temp = qp.weight() * phi;
            M   += temp * trans(phi);
            rhs += test_fun(ep) * temp;
        }

        blaze::DynamicVector<T> proj = inv(M)*rhs;

        for (auto& qp : qps)
        {
            auto ep   = qp.point();
            auto phi  = basis.eval(ep);

            auto val = dot(proj, phi);
            fun_err += qp.weight() * (test_fun(ep) - val) * (test_fun(ep) - val);
        }
    }

    return std::sqrt(fun_err);
}

template<typename Mesh>
void test_quadratures(Mesh& msh)
{
    using T = typename Mesh::coordinate_type;

    auto mesher = dg2d::get_mesher(msh);
    mesher.create_mesh(msh, 0);

    T               prev_ferr;
    std::pair<T,T>  prev_cerrs;

    size_t degree = 1;

    for (size_t rl = 0; rl < 5; rl++)
    {
        auto cerrs = compute_cell_errors(msh, degree);
        auto ferr = compute_face_error(msh, degree);
        if (rl > 0)
        {
            auto ce1 = std::log(prev_cerrs.first / cerrs.first)/std::log(2);
            auto ce2 = std::log(prev_cerrs.second / cerrs.second)/std::log(2);
            auto fe  = std::log(prev_ferr / ferr)/std::log(2);
            std::cout << ce1 << " " << ce2 << " " << fe << std::endl;
        }
        prev_cerrs = cerrs;
        prev_ferr = ferr;
        mesher.refine_mesh(msh,1);
    }
}

int main(void)
{
    using T = double;

    //dg2d::simplicial_mesh<T> msh_s;
    //shatter_mesh(msh_s, 0.15);
    //test_quadratures(msh_s);

    dg2d::quad_mesh<T> msh_q;
    //shatter_mesh(msh_q, 0.15);
    test_quadratures(msh_q);

    return 0;
}
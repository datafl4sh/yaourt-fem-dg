#include <iostream>
#include <sstream>

#include "core/mesh.hpp"
#include "core/meshers.hpp"

#include "methods/hho.hpp"
#include "core/dataio.hpp"

template<typename Mesh>
typename Mesh::coordinate_type
test_hho_reconstruction_operator(const Mesh& msh, size_t degree)
{
    using T = typename Mesh::coordinate_type;
    
    auto f = [](const point<T,2>& pt) -> T {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };
    
    hho_degree_info hdi( equal_order{degree} );
    auto rd = hdi.reconstruction_degree();
    auto rbs = yaourt::bases::scalar_basis_size(rd, 2);
    blaze::DynamicVector<T> R(rbs, 0.0);
    
    T err = 0.0;
    for (auto& cl : msh.cells)
    {
        auto hpf = project_hho(msh, cl, hdi, f);
        auto gr = make_hho_gradient_reconstruction(msh, cl, hdi);
        subvector(R, 1, rbs-1) = gr*hpf;
        
        auto rb = yaourt::bases::make_basis(msh, cl, rd);
        auto pf = project(msh, cl, rb, f);
        auto K = make_stiffness_matrix(msh, cl, rb);
        auto diff = R-pf;
        
        err += dot(diff, K*diff);
    }
    
    return std::sqrt(err);
}

template<typename Mesh>
typename Mesh::coordinate_type
test_hho_stabilization(const Mesh& msh, size_t degree)
{
    using T = typename Mesh::coordinate_type;
    
    auto f = [](const point<T,2>& pt) -> T {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };
    
    hho_degree_info hdi( equal_order{degree} );
    
    T err = 0.0;
    for (auto& cl : msh.cells)
    {
        auto hpf = project_hho(msh, cl, hdi, f);
        auto gr = make_hho_gradient_reconstruction(msh, cl, hdi);
        auto Z = make_hho_stabilization(msh, cl, gr, hdi);
        
        err += std::abs( dot(hpf, Z*hpf) );
    }
    
    return std::sqrt(err);
}

enum class convergence_status {
    OK,
    TOO_LOW,
    TOO_HIGH,
};

std::ostream&
operator<<(std::ostream& os, const convergence_status& cs)
{
    switch (cs)
    {
        case convergence_status::OK:
            os << "\x1b[32m";
            break;
        
        case convergence_status::TOO_LOW:
            os << "\x1b[31m";
            break;
            
        case convergence_status::TOO_HIGH:
            os << "\x1b[33m";
            break;
    }
    
    return os;
}
            
convergence_status
rate_color(double actual, double expected)
{
    const double low = 0.9;
    const double high = 1.1;
            
    if ( actual < low*expected )
        return convergence_status::TOO_LOW;
            
    if ( actual > high*expected )
        return convergence_status::TOO_HIGH;
            
    return convergence_status::OK;
}

template<typename Mesh, typename Function>
int test(const Function& tf)
{
    using mesh_type = Mesh;
    using T = typename mesh_type::coordinate_type;

    for (size_t degree = 0; degree < 4; degree++)
    {
        std::cout << "Testing order " << degree << ", expected rate is ";
        std::cout << degree+1 << std::endl;

        mesh_type msh;
        auto mesher = yaourt::get_mesher(msh);
        mesher.create_mesh(msh, 2);
        
        T err_prev = tf(msh, degree);
        
        for (size_t r = 0; r < 3; r++)
        {
            mesher.refine_mesh(msh, 1);
            
            T err_curr = tf(msh, degree);
            
            auto rate = std::log(err_prev/err_curr)/std::log(2);
            
            auto h = diameter(msh);
            
            std::cout << std::setw(10) << std::setprecision(4) << h << "  ";
            std::cout << std::setw(10) << std::setprecision(4) << err_curr << " ";
            std::cout << std::setw(10) << std::setprecision(4);
            std::cout << rate_color(rate, degree+1) << rate << "\x1b[0m" << std::endl;
            
            err_prev = err_curr;
            
            std::stringstream ss;
            ss << "hho_reconstruction_mesh_" << r << ".silo";
            yaourt::dataio::silo_database silo;
            silo.create( ss.str() );
            silo.add_mesh(msh, "mesh");
        }
    }
    return 0;
}

template<typename Mesh>
int test()
{
    auto testfun = [](Mesh& msh, size_t degree) {
        return test_hho_stabilization(msh, degree);
    };
    
    test<Mesh>(testfun);
    
    return 0;
}

int main(int argc, char **argv)
{
    using T = double;
    std::cout << "Simplicial meshes" << std::endl;
    test<yaourt::simplicial_mesh<T>>();
            
    std::cout << "Quad meshes" << std::endl;
    test<yaourt::quad_mesh<T>>();

    return 0;
}

#include "methods/hho.hpp"

template<typename Mesh>
static void
run_hho_solver(Mesh& msh)
{
    using matr = blaze::DynamicMatrix<typename Mesh::coordinate_type>;
    using vect = blaze::DynamicVector<typename Mesh::coordinate_type>;
    using point_type = typename Mesh::point_type;

    hho_degree_info hdi( equal_order{1} );

    auto rhs_fun = [](const point_type& pt) -> auto {
        return 0.0;
    };

    for (auto& cl : msh.cells)
    {
        auto [GR, A] = make_hho_gradient_reconstruction(msh, cl, hdi);
        auto S = make_hho_stabilization(msh, cl, GR, hdi);
        matr L = A + S;
        auto b = hho_rhs(msh, cl, hdi, rhs_fun);

        auto [LC, bC] = static_condensation(hdi, L, b);
    }
}

int main(int argc, char **argv)
{
	using T = double;
	using mesh_type = yaourt::simplicial_mesh<T>;

	mesh_type msh;

    auto mesher = yaourt::get_mesher(msh);
    mesher.create_mesh(msh, 6);

	run_hho_solver(msh);

	return 0;
}

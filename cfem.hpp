
#include "mesh.hpp"
#include "blaze/Math.h"


namespace dg2d { namespace cfem {

template<typename T>
blaze::StaticVector<T, 3>
eval_basis(const simplicial_mesh<T>& msh,
           const typename simplicial_mesh<T>::cell_type& cl,
           const typename simplicial_mesh<T>::point_type& pt)
{
    blaze::StaticVector<T, 3> ret;

    auto pts = points(msh, cl);
    auto x0 = pts[0].x(); auto y0 = pts[0].y();
    auto x1 = pts[1].x(); auto y1 = pts[1].y();
    auto x2 = pts[2].x(); auto y2 = pts[2].y();

    auto m = (x1*y2 - y1*x2 - x0*(y2 - y1) + y0*(x2 - x1));

    ret[0] = (x1*y2 - y1*x2 - pt.x() * (y2 - y1) + pt.y() * (x2 - x1)) / m;
    ret[1] = (x2*y0 - y2*x0 + pt.x() * (y2 - y0) - pt.y() * (x2 - x0)) / m;
    ret[2] = (x0*y1 - y0*x1 - pt.x() * (y1 - y0) + pt.y() * (x1 - x0)) / m;

    return ret;
}

template<typename T>
blaze::StaticMatrix<T, 3, 2>
eval_basis_grad(const simplicial_mesh<T>& msh,
                const typename simplicial_mesh<T>::cell_type& cl)
{
    blaze::StaticMatrix<T, 3, 2> ret;

    auto pts = points(msh, cl);
    auto x0 = pts[0].x(); auto y0 = pts[0].y();
    auto x1 = pts[1].x(); auto y1 = pts[1].y();
    auto x2 = pts[2].x(); auto y2 = pts[2].y();

    auto m = (x1*y2 - y1*x2 - x0*(y2 - y1) + y0*(x2 - x1));

    ret(0,0) = (y1 - y2) / m;
    ret(1,0) = (y2 - y0) / m;
    ret(2,0) = (y0 - y1) / m;
    ret(0,1) = (x2 - x1) / m;
    ret(1,1) = (x0 - x2) / m;
    ret(2,1) = (x1 - x0) / m;

    return ret;
}

enum class bc_mode
{
    STRONG,
    WEAK,
    NITSCHE
};

template<typename T, typename Idx = int>
class cmat_triplet
{
    Idx     i, j;
    T       val;

public:
    cmat_triplet() : i(0), j(0), val(0)
    {}

    cmat_triplet(Idx p_i, Idx p_j, T p_val)
        : i(p_i), j(p_j), val(p_val)
    {}

    bool operator<(const cmat_triplet& other) const {
        return (i == other.i and j < other.j) or (i < other.i);
    }

    auto row() const { return i; }
    auto col() const { return j; }
    auto val() const { return val; }
};

template<typename Mesh>
class assembler
{
    blaze::CompressedMatrix<T>      lhs;
    blaze::DynamicVector<T>         rhs;

    std::vector<cmat_triplet>       triplets;
    std::vector<bool>               dirichlet_nodes;

    std::vector<int>                compress_map;
    std::vector<int>                expand_map;

public:
    assembler()
    {}

    assembler(const Mesh& msh, const BoundaryConditions& bc,
              const bc_mode& bcmode)
    {

    }

    bool assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const blaze::StaticMatrix<3,3>& local_rhs,
                  const blaze::StaticVector<3>& local_lhs)
    {
        auto l2g = cl.point_ids();

        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                triplets.push_back( l2g(i), l2g(j), local_rhs(i,j) );
            }

            rhs[ l2g(i) ] = local_lhs[i];
        }
    }
};

} //namespace cfem
} //namespace dg2d
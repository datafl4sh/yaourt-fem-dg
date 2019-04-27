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

#pragma once

#include <blaze/Math.h>
#include <cassert>
#include "../core/mesh.hpp"
#include "../core/blaze_sparse_init.hpp"


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

template<typename T>
blaze::StaticMatrix<T, 3, 3>
stiffness_matrix(const simplicial_mesh<T>& msh,
                 const typename simplicial_mesh<T>::cell_type& cl)
{
    blaze::StaticMatrix<T, 3, 3> ret;

    auto meas = measure(msh, cl);
    auto dphi = eval_basis_grad(msh, cl);
    auto stiff = meas * dphi * trans(dphi);

    return stiff;
}

enum class bc_mode
{
    STRONG,
    WEAK,
    NITSCHE
};

template<typename Mesh>
class assembler
{
    using T = typename Mesh::coordinate_type;
    using triplet_type = blaze::triplet<T>;

    std::vector<triplet_type>       triplets;
    std::vector<bool>               dirichlet_nodes;

    std::vector<size_t>             compress_map;
    std::vector<size_t>             expand_map;
    
    size_t                          sys_size;

public:
    blaze::CompressedMatrix<T>      lhs;
    blaze::DynamicVector<T>         rhs;

    assembler()
    {}

    assembler(const Mesh& msh/*, const BoundaryConditions& bc,
              const bc_mode& bcmode*/)
    {
        dirichlet_nodes.resize( msh.points.size() );
        for (auto& f : msh.faces)
        {
            if ( is_boundary(msh, f) )
            {
                auto pts = f.point_ids();
                assert(pts.size() == 2);
                dirichlet_nodes.at( pts[0] ) = true;
                dirichlet_nodes.at( pts[1] ) = true;
            }
        }
        
        compress_map.resize( msh.points.size() );
        sys_size = std::count_if(dirichlet_nodes.begin(),
                                 dirichlet_nodes.end(), 
                                 [](bool d) -> bool {return !d;});
        expand_map.resize( sys_size );
        
        auto nnum = 0;
        for (size_t i = 0; i < msh.points.size(); i++)
        {
            if ( dirichlet_nodes.at(i) )
                continue;

            expand_map.at(nnum) = i;
            compress_map.at(i) = nnum++;
        }
        
        lhs.resize( sys_size, sys_size );
        rhs.resize( sys_size );
    }

    bool assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const blaze::StaticMatrix<T,3,3>& local_rhs,
                  const blaze::StaticVector<T,3>& local_lhs)
    {
        auto l2g = cl.point_ids();
        assert(l2g.size() == 3);
        
        for (size_t i = 0; i < 3; i++)
        {
            if ( dirichlet_nodes.at( l2g[i] ) )
                continue;
            
            for (size_t j = 0; j < 3; j++)
            {
                if ( dirichlet_nodes.at( l2g[j] ) )
                    continue;
                auto ci = compress_map.at(l2g[i]);
                auto cj = compress_map.at(l2g[j]);
                
                triplets.push_back( {ci, cj, local_rhs(i,j)} );
            }

            rhs[ compress_map.at(l2g[i]) ] += local_lhs[i];
        }
        
        return true;
    }
    
    void finalize()
    {
        blaze::init_from_triplets(lhs, triplets.begin(), triplets.end());
        triplets.clear();
    }

    size_t system_size() const { return sys_size; }

    void expand(const blaze::DynamicVector<T>& sol,
                blaze::DynamicVector<T>& expanded_sol)
    {
        expanded_sol.resize( compress_map.size() );
        blaze::reset(expanded_sol);
        for (size_t i = 0; i < sol.size(); i++)
            expanded_sol[ expand_map.at(i) ] = sol[i];
    }
};

template<typename Mesh>
auto get_assembler(const Mesh& msh, size_t degree)
{
    return assembler<Mesh>(msh);
}

} //namespace cfem
} //namespace dg2d

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

#include <stdexcept>
#include <vector>

#include "blaze/Math.h"


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
        : sys_size(0), basis_size(0)
    {}

    assembler(const Mesh& msh, size_t degree, bool bpc = false)
    {
        initialize(msh, degree, bpc);
    }

    void initialize(const Mesh& msh, size_t degree, bool bpc = false)
    {
        basis_size = yaourt::bases::scalar_basis_size(degree,2);
        sys_size = basis_size * msh.cells.size();

        lhs.resize( sys_size, sys_size );
        rhs.resize( sys_size );
        pc.resize( sys_size, sys_size );
        pc_temp = blaze::DynamicVector<T>(sys_size, 0.0);
        build_pc = bpc;
    }

    bool assemble(const Mesh& msh,
                  const typename Mesh::cell_type& cl_a,
                  const typename Mesh::cell_type& cl_b,
                  const blaze::DynamicMatrix<T>& local_rhs)
    {
        if ( sys_size == 0 or basis_size == 0 )
            throw std::invalid_argument("Assembler in invalid state");

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
        if ( sys_size == 0 or basis_size == 0 )
            throw std::invalid_argument("Assembler in invalid state");

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
        if ( sys_size == 0 or basis_size == 0 )
            throw std::invalid_argument("Assembler in invalid state");

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

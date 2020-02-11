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

#ifdef WITH_SILO

#include <silo.h>
#include <blaze/Math.h>

#include "mesh.hpp"

namespace yaourt {
namespace dataio {

class silo_database
{
    DBfile          *m_siloDb;

public:
    silo_database()
        : m_siloDb(nullptr)
    {}

    bool create(const std::string& db_name)
    {
        m_siloDb = DBCreate(db_name.c_str(), DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
        if (m_siloDb)
            return true;

        std::cout << "Error creating database" << std::endl;
        return false;
    }

    bool open(const std::string& db_name)
    {
        m_siloDb = DBOpen(db_name.c_str(), DB_PDB, DB_APPEND);
        if (m_siloDb)
            return true;

        std::cout << "Error opening database" << std::endl;
        return false;
    }

    bool close()
    {
        if (m_siloDb)
            DBClose(m_siloDb);
        m_siloDb = NULL;
        return true;
    }

    ~silo_database()
    {
        if (m_siloDb)
            DBClose(m_siloDb);
    }

    template<typename T>
    bool
    add_mesh(const simplicial_mesh<T>& msh, const std::string& name)
    {
        static_assert(std::is_same<T,double>::value || std::is_same<T,float>::value, "Wrong type");

        std::vector<T> x_coords, y_coords;
        x_coords.reserve(msh.points.size());
        y_coords.reserve(msh.points.size());

        for (auto& pt : msh.points)
        {
            x_coords.push_back( pt.x() );
            y_coords.push_back( pt.y() );
        }

        T *coords[] = {x_coords.data(), y_coords.data()};

        std::vector<int> nodelist;
        nodelist.reserve( 3*msh.cells.size() );

        for (auto& cl : msh.cells)
        {
            auto ptids = cl.point_ids();
            assert(ptids.size() == 3);

            for (auto& ptid : ptids)
                nodelist.push_back( ptid + 1 ); /* SILO uses 1-based indices */
        }

        int lnodelist = nodelist.size();

        int shapesize[] = {3};
        int shapecounts[] = { static_cast<int>(msh.cells.size()) };
        int nshapetypes = 1;
        int nnodes = msh.points.size();
        int nzones = msh.cells.size();
        int ndims = 2;

        std::stringstream zlname;
        zlname << "zonelist_" << name;
        std::string zonelist_name = zlname.str();

        DBPutZonelist(m_siloDb, zonelist_name.c_str(), nzones, ndims,
            nodelist.data(), lnodelist, 1, shapesize, shapecounts, nshapetypes);

        if ( std::is_same<T, float>::value )
        {
            DBPutUcdmesh(m_siloDb, name.c_str(), ndims, NULL, coords, nnodes, nzones,
                zonelist_name.c_str(), NULL, DB_FLOAT, NULL);
        }

        if ( std::is_same<T, double>::value )
        {
            DBPutUcdmesh(m_siloDb, name.c_str(), ndims, NULL, coords, nnodes, nzones,
                zonelist_name.c_str(), NULL, DB_DOUBLE, NULL);
        }


        return true;
    }

    template<typename T>
    bool
    add_mesh(const quad_mesh<T>& msh, const std::string& name)
    {
        static_assert(std::is_same<T,double>::value || std::is_same<T,float>::value, "Wrong type");

        std::vector<T> x_coords, y_coords;
        x_coords.reserve(msh.points.size());
        y_coords.reserve(msh.points.size());

        for (auto& pt : msh.points)
        {
            x_coords.push_back( pt.x() );
            y_coords.push_back( pt.y() );
        }

        T *coords[] = {x_coords.data(), y_coords.data()};

        std::vector<int> nodelist;
        nodelist.reserve( 4*msh.cells.size() );

        for (auto& cl : msh.cells)
        {
            auto ptids = cl.point_ids();
            assert(ptids.size() == 4);

            for (auto& ptid : ptids)
                nodelist.push_back( ptid + 1 ); /* SILO uses 1-based indices */
        }

        int lnodelist = nodelist.size();

        int shapesize[] = {4};
        int shapecounts[] = { static_cast<int>(msh.cells.size()) };
        int nshapetypes = 1;
        int nnodes = msh.points.size();
        int nzones = msh.cells.size();
        int ndims = 2;

        std::stringstream zlname;
        zlname << "zonelist_" << name;
        std::string zonelist_name = zlname.str();

        DBPutZonelist(m_siloDb, zonelist_name.c_str(), nzones, ndims,
            nodelist.data(), lnodelist, 1, shapesize, shapecounts, nshapetypes);

        if ( std::is_same<T, float>::value )
        {
            DBPutUcdmesh(m_siloDb, name.c_str(), ndims, NULL, coords, nnodes, nzones,
                zonelist_name.c_str(), NULL, DB_FLOAT, NULL);
        }

        if ( std::is_same<T, double>::value )
        {
            DBPutUcdmesh(m_siloDb, name.c_str(), ndims, NULL, coords, nnodes, nzones,
                zonelist_name.c_str(), NULL, DB_DOUBLE, NULL);
        }


        return true;
    }

    template<typename T>
    bool add_nodal_variable(const std::string& mesh_name,
                            const std::string& var_name,
                            blaze::DynamicVector<T>& var)
    {
        static_assert(std::is_same<T, double>::value, "Sorry, only double for now");

        if (!m_siloDb)
        {
            std::cout << "Silo database not opened" << std::endl;
            return false;
        }

        DBPutUcdvar1(m_siloDb, var_name.c_str(), mesh_name.c_str(),
                     var.data(),
                     var.size(), NULL, 0, DB_DOUBLE,
                     DB_NODECENT, NULL);

        return true;
    }

    template<typename T>
    bool add_zonal_variable(const std::string& mesh_name,
                            const std::string& var_name,
                            blaze::DynamicVector<T>& var)
    {
        static_assert(std::is_same<T, double>::value, "Sorry, only double for now");

        if (!m_siloDb)
        {
            std::cout << "Silo database not opened" << std::endl;
            return false;
        }

        DBPutUcdvar1(m_siloDb, var_name.c_str(), mesh_name.c_str(),
                     var.data(),
                     var.size(), NULL, 0, DB_DOUBLE,
                     DB_ZONECENT, NULL);

        return true;
    }
};

} // namespace dataio
} // namespace yaourt

#endif

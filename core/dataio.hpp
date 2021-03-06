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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#include <blaze/Math.h>
#pragma clang diagnostic pop

#include "mesh.hpp"

namespace yaourt {
namespace dataio {

class silo_database
{
    DBfile          *m_siloDb;
    DBoptlist       *m_optlist;

    int             cycle;
    double          time;

public:
    silo_database()
        : m_siloDb(nullptr), m_optlist(nullptr)
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
        {
            DBClose(m_siloDb);
            m_siloDb = NULL;
        }

        if (m_optlist)
        {
            DBFreeOptlist(m_optlist);
            m_optlist = nullptr;
        }

        return true;
    }

    ~silo_database()
    {
        if (m_siloDb)
            DBClose(m_siloDb);

        if (m_optlist)
            DBFreeOptlist(m_optlist);
    }

    template<typename T>
    bool
    add_time(int p_cycle, T p_time)
    {
        static_assert(std::is_same<T,double>::value, "Wrong type");

        cycle = p_cycle;
        time = p_time;

        /* cycle and time must live at least until DBPutUcdMesh() */
        m_optlist = DBMakeOptlist(2);
        DBAddOption(m_optlist, DBOPT_CYCLE, &cycle);
        DBAddOption(m_optlist, DBOPT_DTIME, &time);

        return true;
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

        int lnodelist = int(nodelist.size());

        int shapetype[] = { DB_ZONETYPE_TRIANGLE };
        int shapesize[] = {3};
        int shapecounts[] = { static_cast<int>(msh.cells.size()) };
        int nshapetypes = 1;
        int nnodes = msh.points.size();
        int nzones = msh.cells.size();
        int ndims = 2;

        std::stringstream zlname;
        zlname << "zonelist_" << name;
        std::string zonelist_name = zlname.str();

        DBPutZonelist2(m_siloDb, zonelist_name.c_str(), nzones, ndims,
            nodelist.data(), lnodelist, 1, 0, 0, shapetype, shapesize,
            shapecounts, nshapetypes, NULL);

        if ( std::is_same<T, float>::value )
        {
            DBPutUcdmesh(m_siloDb, name.c_str(), ndims, NULL, coords, nnodes, nzones,
                zonelist_name.c_str(), NULL, DB_FLOAT, m_optlist);
        }

        if ( std::is_same<T, double>::value )
        {
            DBPutUcdmesh(m_siloDb, name.c_str(), ndims, NULL, coords, nnodes, nzones,
                zonelist_name.c_str(), NULL, DB_DOUBLE, m_optlist);
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

        int lnodelist = int(nodelist.size());

        int shapetype[] = { DB_ZONETYPE_QUAD };
        int shapesize[] = {4};
        int shapecounts[] = { static_cast<int>(msh.cells.size()) };
        int nshapetypes = 1;
        int nnodes = msh.points.size();
        int nzones = msh.cells.size();
        int ndims = 2;

        std::stringstream zlname;
        zlname << "zonelist_" << name;
        std::string zonelist_name = zlname.str();

        DBPutZonelist2(m_siloDb, zonelist_name.c_str(), nzones, ndims,
            nodelist.data(), lnodelist, 1, 0, 0, shapetype, shapesize,
            shapecounts, nshapetypes, NULL);

        if ( std::is_same<T, float>::value )
        {
            DBPutUcdmesh(m_siloDb, name.c_str(), ndims, NULL, coords, nnodes, nzones,
                zonelist_name.c_str(), NULL, DB_FLOAT, m_optlist);
        }

        if ( std::is_same<T, double>::value )
        {
            DBPutUcdmesh(m_siloDb, name.c_str(), ndims, NULL, coords, nnodes, nzones,
                zonelist_name.c_str(), NULL, DB_DOUBLE, m_optlist);
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

    bool add_expression(const std::string& expr_name,
                        const std::string& expr_definition,
                        int expr_type)
    {
        std::stringstream ss;
        ss << "def_" << expr_name;
        const char *name[] = { expr_name.c_str() };
        const char *def[] = { expr_definition.c_str() };
        DBPutDefvars(m_siloDb, ss.str().c_str(), 1, name, &expr_type, def, NULL);
        return true;
    }
};

} // namespace dataio
} // namespace yaourt

#endif

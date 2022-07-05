/*
 * Yaourt-FEM-DG - Yet AnOther Useful Resource for Teaching FEM and DG.
 *
 * Matteo Cicuttin (C) 2019-2022
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

#include <iostream>
#include <fstream>
#include <sstream>

#include <cstdio>
#include <cstring>
#include <cmath>

#include <unistd.h>

#include "core/mesh.hpp"
#include "core/meshers.hpp"
#include "core/quadratures.hpp"
#include "core/bases.hpp"
#include "core/solvers.hpp"
#include "core/blaze_sparse_init.hpp"
#include "core/dataio.hpp"

#include "methods/dg.hpp"

#define VX	0
#define VY	1
#define P	2

template<typename T>
struct field_energies
{
	T	Wvx;
	T	Wvy;
	T	Wp;
};

class gnuplot
{
    FILE *gp_fh;

public:
    gnuplot()
    {
        gp_fh = popen("/usr/bin/gnuplot", "w");
        if (!gp_fh)
        {
            std::cout << "Can't popen() gnuplot" << std::endl;
            return;
        }
    }

    void plot(const std::vector<double>& data)
    {
		if (!gp_fh)
			return;

        fprintf(gp_fh, "plot '-' using 1 w lp\n");
		for (auto& d : data)
			fprintf(gp_fh, "%g\n", d);
		fprintf(gp_fh, "e\n");
		fflush(gp_fh);
    }

	template<typename T>
	void plot(const std::vector<field_energies<T>>& data)
    {
		if (!gp_fh)
			return;

        fprintf(gp_fh, "plot '-' w l ti 'Wvx', '-' w l ti 'Wvy', '-' w l ti 'Wp'\n");
		
		for (auto& d : data)
			fprintf(gp_fh, "%g\n", d.Wvx);
		fprintf(gp_fh, "e\n");

		for (auto& d : data)
			fprintf(gp_fh, "%g\n", d.Wvy);
		fprintf(gp_fh, "e\n");
		
		for (auto& d : data)
			fprintf(gp_fh, "%g\n", d.Wp);
		fprintf(gp_fh, "e\n");
		
		fflush(gp_fh);
    }

    ~gnuplot()
    {
        if (gp_fh)
            pclose(gp_fh);
    }
};



template<typename Mesh>
void
apply_acoustics_operator(Mesh& msh,
	const blaze::DynamicMatrix<typename Mesh::coordinate_type>& in,
	blaze::DynamicMatrix<typename Mesh::coordinate_type>& out)
{
	using mesh_type = Mesh;
    using T = typename mesh_type::coordinate_type;

	assert( msh.cells.size() == in.rows() );
	assert( msh.cells.size() == out.rows() );
	assert( in.columns() == 3 );
	assert( out.columns() == 3 );

    msh.compute_connectivity();

	for (auto& tcl : msh.cells)
    {
		auto ofs_mine = offset(msh, tcl);
		auto ht = measure(msh, tcl);
		auto alpha = 1.0;
		
		T flux_p  = 0.0;
		T flux_vx = 0.0;
		T flux_vy = 0.0;

		auto fcs = faces(msh, tcl);
        for (auto& fc : fcs)
        {
			auto hf = measure(msh, fc);
            auto [ncl, has_neighbour] = neighbour_via(msh, tcl, fc);
            auto n = normal(msh, tcl, fc);
			auto nx = n[0];
			auto ny = n[1];

			if (not has_neighbour)
			{
				assert(fc.is_boundary == true);
				auto svx = in(ofs_mine, VX);
				auto svy = in(ofs_mine, VY);
				auto sp  = in(ofs_mine, P);
				
				auto jvx = in(ofs_mine, VX);
				auto jvy = in(ofs_mine, VY);
				auto jp = in(ofs_mine, P);

				//if (fc.boundary_id != 1)
				//{
				//	flux_vx += (hf/ht) * ( nx*sp + alpha*jvx );
				//	flux_vy += (hf/ht) * ( ny*sp + alpha*jvy );
				//}
				
				flux_p  += (hf/ht) * ( nx*svx + ny*svy + alpha*jp );
			}
			else
			{
				auto ofs_neigh = offset(msh, ncl);

				auto svx = in(ofs_mine, VX) + in(ofs_neigh, VX);
				auto svy = in(ofs_mine, VY) + in(ofs_neigh, VY);
				auto sp  = in(ofs_mine, P)  + in(ofs_neigh, P);

				auto jvx = in(ofs_mine, VX) - in(ofs_neigh, VX);
				auto jvy = in(ofs_mine, VY) - in(ofs_neigh, VY);
				auto jp = in(ofs_mine, P) - in(ofs_neigh, P);

				flux_vx += (0.5*hf/ht) * ( nx*sp + alpha*jvx );
				flux_vy += (0.5*hf/ht) * ( ny*sp + alpha*jvy );
				flux_p  += (0.5*hf/ht) * ( nx*svx + ny*svy + alpha*jp );
			}
		}
		
		out(ofs_mine, VX) = -flux_vx;
		out(ofs_mine, VY) = -flux_vy;
		out(ofs_mine, P) = -flux_p;
	}
}



template<typename Mesh>
field_energies<typename Mesh::coordinate_type>
compute_energies(Mesh& msh,
	const blaze::DynamicMatrix<typename Mesh::coordinate_type>& field)
{
	using mesh_type = Mesh;
    using T = typename mesh_type::coordinate_type;

	field_energies<T> fe;
	fe.Wvx = 0.0;
	fe.Wvy = 0.0;
	fe.Wp = 0.0;

	size_t cell_i = 0;
	for (auto& tcl : msh.cells)
	{
		auto ht = measure(msh, tcl);

		T vx = field(cell_i, VX);
		T vy = field(cell_i, VY);
		T p = field(cell_i, P);

		fe.Wvx += 0.5*vx*vx*ht;
		fe.Wvy += 0.5*vy*vy*ht;
		fe.Wp += 0.5*p*p*ht;

		cell_i++;
	}

	return fe;
}

#ifdef WITH_SILO
template<typename Mesh>
static void
export_solution(Mesh& msh, size_t ts,
	blaze::DynamicMatrix<typename Mesh::coordinate_type>& data)
{
	using T = typename Mesh::coordinate_type;
	std::stringstream ss;
	ss << "fvol_acoustics_" << ts << ".silo";
	yaourt::dataio::silo_database silo;
    silo.create( ss.str() );
    silo.add_mesh(msh, "mesh");

	blaze::DynamicVector<T> vx = column(data, VX);
	silo.add_zonal_variable("mesh", "vx", vx);

	blaze::DynamicVector<T> vy = column(data, VY);
	silo.add_zonal_variable("mesh", "vy", vy);

	blaze::DynamicVector<T> p = column(data, P);
	silo.add_zonal_variable("mesh", "p", p);

    silo.close();
}
#endif

template<typename Mesh>
static void
run_acoustics_solver(Mesh& msh)
{
	auto num_cells = msh.cells.size();

	using T = typename Mesh::coordinate_type;
	blaze::DynamicMatrix<T> curr(num_cells, 3);
	blaze::DynamicMatrix<T> next(num_cells, 3);
	blaze::DynamicMatrix<T> k1(num_cells, 3);
	blaze::DynamicMatrix<T> k2(num_cells, 3);
	blaze::DynamicMatrix<T> k3(num_cells, 3);
	blaze::DynamicMatrix<T> k4(num_cells, 3);
	blaze::DynamicMatrix<T> tmp(num_cells, 3);

	T dt = 0.0001;

	for (auto& tcl : msh.cells)
    {
		auto bar = barycenter(msh, tcl);
		auto ax = bar.x() - 0.5;
		auto ay = bar.y() - 0.5;
		//T e = -100*(ax*ax + ay*ay);
		//T val = std::exp(e);
		T val = std::sin(M_PI*bar.x())*std::sin(M_PI*bar.y());
		auto ofs = offset(msh, tcl);
		curr(ofs, P) = val;
	}

	std::vector<field_energies<T>> nrg;

	gnuplot gp;

	for (size_t i = 0; i < 20000; i++)
	{
		std::cout << "Timestep " << i << "\r" << std::flush;
		apply_acoustics_operator(msh, curr, k1);

		tmp = curr + 0.5*dt*k1;
		apply_acoustics_operator(msh, tmp, k2);

		tmp = curr + 0.5*dt*k2;
		apply_acoustics_operator(msh, tmp, k3);

		tmp = curr + dt*k3;
		apply_acoustics_operator(msh, tmp, k4);

		next = curr + dt*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;

		if (i%100 == 0)
		{
#ifdef WITH_SILO
			export_solution(msh, i+1, next);
#endif
			nrg.push_back( compute_energies(msh, next) );
			gp.plot(nrg);
		}

		curr = next;
	}
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	using T = double;
	using mesh_type = yaourt::quad_mesh<T>;

	mesh_type msh;

    auto mesher = yaourt::get_mesher(msh);
    mesher.create_mesh(msh, 6);

	run_acoustics_solver(msh);

	return 0;
}

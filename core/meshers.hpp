#pragma once

#include "mesh.hpp"

namespace dg2d {

template<typename Mesh>
class mesher;

template<typename T>
class mesher<simplicial_mesh<T>>
{
	using coordinate_type = T;
	using mesh_type = simplicial_mesh<T>;
	using cell_type = typename mesh_type::cell_type;
	using face_type = typename mesh_type::face_type;
	using point_type = typename mesh_type::point_type;

	bool 					verbose;
	std::vector<cell_type> 	new_cells;

	void
	refine_mesh(mesh_type& msh)
	{
		new_cells.clear();

		 /* break all the faces */
        for (auto& e : msh.faces)
        {
        	assert(e.p0 < msh.points.size());
        	assert(e.p1 < msh.points.size());
            assert(!e.is_broken);
            auto bar = (msh.points[e.p0] + msh.points[e.p1])/T(2);
            e.pb = msh.points.size();
            msh.points.push_back(bar);
            e.is_broken = true;
        }

        /* refine the triangles */
        size_t triangles_to_process = msh.cells.size();
        msh.faces.reserve( msh.faces.size() + triangles_to_process*12 );
        auto faces_end = msh.faces.end();
        for (size_t i = 0; i < triangles_to_process; i++)
        {
            triangle t = msh.cells[i];
            auto pts = t.point_ids();

            auto p0 = pts[0];
            auto p1 = pts[1];
            auto p2 = pts[2];

            /* find the faces of the triangle */
            auto t_e0 = *std::lower_bound(msh.faces.begin(), faces_end, face_type(p0, p1));
            assert(t_e0.is_broken);

            auto t_e1 = *std::lower_bound(msh.faces.begin(), faces_end, face_type(p1, p2));
            assert(t_e1.is_broken);

            auto t_e2 = *std::lower_bound(msh.faces.begin(), faces_end, face_type(p0, p2));
            assert(t_e2.is_broken);

            auto p0b = t_e0.pb;
            auto p1b = t_e1.pb;
            auto p2b = t_e2.pb;

            /* compute the faces of the new four triangles */
            /* first triangle */
            msh.faces.push_back( face_type(p0, p0b, t_e0.boundary_id, t_e0.is_boundary) );
            msh.faces.push_back( face_type(p0b, p2b, false) );
            msh.faces.push_back( face_type(p0, p2b, t_e2.boundary_id, t_e2.is_boundary) );
            triangle t0(p0, p0b, p2b);
            new_cells.push_back(t0);

            /* second triangle */
            msh.faces.push_back( face_type(p0b, p1, t_e0.boundary_id, t_e0.is_boundary) );
            msh.faces.push_back( face_type(p0b, p1b, false) );
            msh.faces.push_back( face_type(p1, p1b, t_e1.boundary_id, t_e1.is_boundary) );
            triangle t1(p0b, p1, p1b);
            new_cells.push_back(t1);

            /* third triangle */
            msh.faces.push_back( face_type(p1b, p2, t_e1.boundary_id, t_e1.is_boundary) );
            msh.faces.push_back( face_type(p2b, p1b, false) );
            msh.faces.push_back( face_type(p2, p2b, t_e2.boundary_id, t_e2.is_boundary) );
            triangle t2(p1b, p2, p2b);
            new_cells.push_back(t2);

            /* fourth triangle */
            triangle t3(p0b, p1b, p2b);
            new_cells.push_back(t3);
        }

        /* we don't need the broken faces anymore, discard them */
        auto face_is_broken = [](const face_type& f) -> bool { return f.is_broken; };
        std::remove_if(msh.faces.begin(), msh.faces.end(), face_is_broken);

        /* sort the faces to allow fast lookups */
        priv::sort_uniq(msh.faces);
        std::swap(msh.cells, new_cells);
	}

public:
	mesher()
		: verbose(false)
	{}

	mesher(bool vrb)
		: verbose(vrb)
	{}

	void
	create_mesh(mesh_type& msh, size_t refinement_iterations)
	{
		msh.points.push_back( point_type(0.0, 0.0) );
		msh.points.push_back( point_type(1.0, 0.0) );
		msh.points.push_back( point_type(1.0, 1.0) );
		msh.points.push_back( point_type(0.0, 1.0) );
		msh.points.push_back( point_type(0.5, 0.5) );

		msh.faces.push_back( face_type(0,1,0,true) );
		msh.faces.push_back( face_type(0,3,3,true) );
		msh.faces.push_back( face_type(0,4) );
		msh.faces.push_back( face_type(1,2,1,true) );
		msh.faces.push_back( face_type(1,4) );
		msh.faces.push_back( face_type(2,3,2,true) );
		msh.faces.push_back( face_type(2,4) );
		msh.faces.push_back( face_type(3,4) );

		msh.cells.push_back( triangle(0,1,4) );
		msh.cells.push_back( triangle(0,4,3) );
		msh.cells.push_back( triangle(1,2,4) );
		msh.cells.push_back( triangle(2,3,4) );

		if (verbose)
		{
			std::cout << "[Simplicial mesher] Base mesh created: ";
			std::cout << msh.cells.size() << " elements";
			std::cout << std::endl;
		}

		refine_mesh(msh, refinement_iterations);
	}

	void
	refine_mesh(mesh_type& msh, size_t refinement_iterations)
	{
		for (size_t i = 0; i < refinement_iterations; i++)
		{
			refine_mesh(msh);

			if (verbose)
			{
				std::cout << "[Simplicial mesher] Refinement step ";
				std::cout << i << ": " << msh.cells.size() << " elements";
				std::cout << std::endl;
			}
		}

		std::sort(msh.cells.begin(), msh.cells.end());
	}


};

template<typename Mesh>
auto
get_mesher(const Mesh&)
{
	return mesher<Mesh>(true);
}

}
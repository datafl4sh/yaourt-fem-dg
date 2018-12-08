#pragma once

#include "point.hpp"

namespace dg2d {

namespace priv {

template<typename T>
void sort_uniq(std::vector<T>& vec)
{
	std::sort(vec.begin(), vec.end());
	vec.erase( std::unique(vec.begin(), vec.end()), vec.end() );
}

} //namespace priv

struct edge
{
    edge(){}

    edge(size_t ap0, size_t ap1, bool bnd = false)
    {
        assert (ap0 != ap1);
        p0                  = ap0;
        p1                  = ap1;
        if (p0 > p1)
            std::swap(p0, p1);

        is_boundary         = bnd;
        is_broken           = false;
    }

    edge(size_t ap0, size_t ap1, size_t bid, bool bnd)
    {
        assert (ap0 != ap1);
        p0                  = ap0;
        p1                  = ap1;
        if (p0 > p1)
            std::swap(p0, p1);

        is_boundary         = bnd;
        is_broken           = false;
        boundary_id         = bid;
    }

    friend bool operator<(const edge& a, const edge& b) {
        assert(a.p0 < a.p1);
        assert(b.p0 < b.p1);
        return ( a.p0 < b.p0 or (a.p0 == b.p0 and a.p1 < b.p1) );
    }

    friend bool operator==(const edge& a, const edge& b) {
        return ( a.p0 == b.p0 and a.p1 == b.p1 );
    }

    friend std::ostream& operator<<(std::ostream& os, const edge& e) {
        os << "Edge: " << e.p0 << " " << e.p1;
        if (e.is_broken) os << ", broken";
        if (e.is_boundary) os << ", boundary";
        return os;
    }

    size_t  p0, p1, pb;
    bool    is_boundary, is_broken;
    size_t  boundary_id;
};


struct triangle
{
    triangle() {}

    triangle(size_t ap0, size_t ap1, size_t ap2)
    	: p{ap0, ap1, ap2}
    {
        std::sort(p.begin(), p.end());
    }

    std::array<size_t, 3> p;

    friend bool operator<(const triangle& a, const triangle& b) {
        return std::lexicographical_compare( a.p.begin(), a.p.end(),
                                             b.p.begin(), b.p.end() );
    }

    friend std::ostream& operator<<(std::ostream& os, const triangle& t) {
        os << "Triangle: " << t.p[0] << " " << t.p[1] << " " << t.p[2];
        return os;
    }
};

template<typename Mesh>
size_t
offset(const Mesh& msh, const typename Mesh::edge_type& edge)
{
	auto itor = std::lower_bound(msh.edges.begin(), msh.edges.end(), edge);
	if (itor == msh.edges.end())
		throw std::invalid_argument("edge not found");

	return std::distance(msh.edges.begin(), itor);
}

template<typename T>
std::array<edge, 3>
edges(const simplicial_mesh<T>& msh, const simplicial_mesh<T>::cell_type& cell)
{
	std::array<edge, 3> ret;
	ret[0] = msh.edges.at( offset(msh, edge(cell.p[0], cell.p[1])) );
	ret[1] = msh.edges.at( offset(msh, edge(cell.p[1], cell.p[2])) );
	ret[2] = msh.edges.at( offset(msh, edge(cell.p[0], cell.p[2])) );
	return ret;
}

template<typename T>
std::array<size_t, 3>
edge_ids(const simplicial_mesh<T>& msh, const simplicial_mesh<T>::cell_type& cell)
{
	std::array<size_t, 3> ret;
	ret[0] = offset(msh, edge(cell.p[0], cell.p[1]));
	ret[1] = offset(msh, edge(cell.p[1], cell.p[2]));
	ret[2] = offset(msh, edge(cell.p[0], cell.p[2]));
	return ret;
}

template<typename T, typename CellT>
class mesh
{
public:
	using coordinate_type 	= T;
	using edge_type 		= edge;
	using cell_type			= CellT;
	using point_type 		= point<T,2>;

	std::vector<point_type>		points;
	std::vector<edge_type>		edges;
	std::vector<cell_type>		cells;

	std::vector<std::array<size_t,2>> edge_owners;

	mesh()
	{}
};

template<typename T>
using simplicial_mesh = mesh<T, triangle>;

template<typename Mesh>
class mesher;

template<typename T>
class mesher<simplicial_mesh<T>>
{
	using coordinate_type = T;
	using mesh_type = simplicial_mesh<T>;
	using cell_type = typename mesh_type::cell_type;
	using point_type = typename mesh_type::point_type;

	bool 					verbose;
	std::vector<cell_type> 	new_cells;

	void
	refine_mesh(mesh_type& msh)
	{
		new_cells.clear();

		 /* break all the edges */
        for (auto& e : msh.edges)
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
        msh.edges.reserve( msh.edges.size() + triangles_to_process*12 );
        auto edges_end = msh.edges.end();
        for (size_t i = 0; i < triangles_to_process; i++)
        {
            /* remove the original triangle */
            triangle t = msh.cells[i];

            /* find the edges of the triangle */
            auto t_e0 = *std::lower_bound(msh.edges.begin(), edges_end, edge(t.p[0], t.p[1]));
            assert(t_e0.is_broken);

            auto t_e1 = *std::lower_bound(msh.edges.begin(), edges_end, edge(t.p[1], t.p[2]));
            assert(t_e1.is_broken);

            auto t_e2 = *std::lower_bound(msh.edges.begin(), edges_end, edge(t.p[0], t.p[2]));
            assert(t_e2.is_broken);

            assert(t_e0.p1 == t_e1.p0);
            assert(t_e1.p1 == t_e2.p1);
            assert(t_e0.p0 == t_e2.p0);

            /* compute the edges of the new four triangles */
            /* first triangle */
            msh.edges.push_back( edge(t_e0.p0, t_e0.pb, t_e0.boundary_id, t_e0.is_boundary) );
            msh.edges.push_back( edge(t_e0.pb, t_e2.pb, false) );
            msh.edges.push_back( edge(t_e0.p0, t_e2.pb, t_e2.boundary_id, t_e2.is_boundary) );
            triangle t0(t_e0.p0, t_e0.pb, t_e2.pb);
            new_cells.push_back(t0);

            /* second triangle */
            msh.edges.push_back( edge(t_e0.p1, t_e0.pb, t_e0.boundary_id, t_e0.is_boundary) );
            msh.edges.push_back( edge(t_e0.pb, t_e1.pb, false) );
            msh.edges.push_back( edge(t_e1.p0, t_e1.pb, t_e1.boundary_id, t_e1.is_boundary) );
            triangle t1(t_e0.p1, t_e0.pb, t_e1.pb);
            new_cells.push_back(t1);

            /* third triangle */
            msh.edges.push_back( edge(t_e1.p1, t_e1.pb, t_e1.boundary_id, t_e1.is_boundary) );
            msh.edges.push_back( edge(t_e1.pb, t_e2.pb, false) );
            msh.edges.push_back( edge(t_e2.p1, t_e2.pb, t_e2.boundary_id, t_e2.is_boundary) );
            triangle t2(t_e1.p1, t_e1.pb, t_e2.pb);
            new_cells.push_back(t2);

            /* fourth triangle */
            triangle t3(t_e0.pb, t_e1.pb, t_e2.pb);
            new_cells.push_back(t3);
        }

        /* we don't need the broken edges anymore, discard them */
        auto edge_is_broken = [](const edge& e) -> bool { return e.is_broken; };
        std::remove_if(msh.edges.begin(), msh.edges.end(), edge_is_broken);

        /* sort the edges to allow fast lookups */
        priv::sort_uniq(msh.edges);
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
		msh.points.push_back( point_type(0.0, 1.0) );
		msh.points.push_back( point_type(1.0, 1.0) );
		msh.points.push_back( point_type(0.5, 0.5) );

		msh.edges.push_back( edge(0,1,0,true) );
		msh.edges.push_back( edge(0,3,3,true) );
		msh.edges.push_back( edge(0,4) );
		msh.edges.push_back( edge(1,2,1,true) );
		msh.edges.push_back( edge(1,4) );
		msh.edges.push_back( edge(2,3,2,true) );
		msh.edges.push_back( edge(2,4) );
		msh.edges.push_back( edge(3,4) );

		msh.cells.push_back( triangle(0,1,4) );
		msh.cells.push_back( triangle(0,3,4) );
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

} //namespace dg2d
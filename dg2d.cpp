/*
 * Toy implememtation of Discontinuous Galerkin for teaching purposes
 *
 * Matteo Cicuttin (c) 2018
 */

#include <iostream>
#include <vector>
#include <list>
#include <blaze/Math.h>

#include "mesh.hpp"



int main(int argc, char **argv)
{
	
	using T = double;

	dg2d::simplicial_mesh<T> mesh;
	auto mesher = dg2d::get_mesher(mesh);

	mesher.create_mesh(mesh, 4);

	return 0;
}




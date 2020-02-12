#pragma once

template<typename Mesh>
solver_status<typename Mesh::coordinate_type>
run_maxwell_TE_solver(Mesh& msh, const dg_config<typename Mesh::coordinate_type>& cfg)
{
    using mesh_type = Mesh;
    using T = typename mesh_type::coordinate_type;

    solver_status<T>    status;
    status.mesh_h = diameter(msh);

    msh.compute_connectivity();

    size_t degree = cfg.degree;

    size_t eta = 5;

    size_t faces_per_elem = 0;
    if (std::is_same<Mesh, yaourt::simplicial_mesh<T>>::value)
        faces_per_elem = 3;
    if (std::is_same<Mesh, yaourt::quad_mesh<T>>::value)
        faces_per_elem = 4;

    std::vector<T> mu( msh.cells.size() );
    std::vector<T> eps( msh.cells.size() );

    size_t timesteps = 200;
    T delta_t = 0.01;

    for (size_t i = 0; i < msh.cells.size(); i++)
    {
        mu[i] = 1;
        eps[i] = 1;
    }

    auto basis_size = yaourt::bases::scalar_basis_size(degree, 2);

    blaze::DynamicVector<T> global_dofs_T(3*basis_size*msh.cells.size(), 0.0);
    blaze::DynamicVector<T> global_dofs_T_plus_one(3*basis_size*msh.cells.size(), 0.0);

    blaze::DynamicMatrix<T> global_M2d(basis_size*msh.cells.size(), basis_size, 0.0);
    blaze::DynamicMatrix<T> global_M(3*basis_size*msh.cells.size(), 3*basis_size, 0.0);
    blaze::DynamicMatrix<T> global_offdiag_contribs(faces_per_elem*3*basis_size*msh.cells.size(), 3*basis_size, 0.0);
    
    size_t cell_i = 0;
    size_t offdiag_contrib_i = 0;
    for (auto& tcl : msh.cells)
    {
        auto tbasis = yaourt::bases::make_basis(msh, tcl, degree);

        blaze::DynamicMatrix<T> M2d(tbasis.size(), tbasis.size(), 0.0);
        blaze::DynamicMatrix<T> Sx(tbasis.size(), tbasis.size(), 0.0);
        blaze::DynamicMatrix<T> Sy(tbasis.size(), tbasis.size(), 0.0);
        

        auto qps = yaourt::quadratures::integrate(msh, tcl, 2*degree);
        for (auto& qp : qps)
        {
            auto ep   = qp.point();
            auto phi  = tbasis.eval(ep);
            auto dphi = tbasis.eval_grads(ep);

            auto dphi_x = blaze::column<0>(dphi);
            auto dphi_y = blaze::column<1>(dphi);

            /* Mass */
            M2d += qp.weight() * phi * trans(phi);
            /* Stiffness, direction x */
            Sx += qp.weight() * phi * trans( dphi_x );
            /* Stiffness, direction y */
            Sy += qp.weight() * phi * trans( dphi_y );

            //loc_rhs += ic(qp.point()) * qp.weight() * phi;
        }

        submatrix(global_M2d, cell_i*basis_size, 0, basis_size, basis_size) = M2d;

        auto get_global_block = [&](size_t i, size_t j) -> auto {
            auto offset_i = 3*basis_size*cell_i + i*basis_size;
            auto offset_j = j*basis_size;
            return submatrix(global_M, offset_i, offset_j, basis_size, basis_size);
        };

        auto get_global_rowblock = [&](size_t i) -> auto {
            auto offset_i = 3*basis_size*cell_i + i*basis_size;
            auto offset_j = 0;
            return submatrix(global_M, offset_i, offset_j, basis_size, 3*basis_size);
        };

        auto invM2d_Sx = blaze::solve(M2d, Sx);
        auto invM2d_Sy = blaze::solve(M2d, Sy);

        get_global_block(0, 0) = blaze::IdentityMatrix<T>(basis_size);
        get_global_block(0, 2) = (delta_t/eps[cell_i])*invM2d_Sy;

        get_global_block(1, 1) = blaze::IdentityMatrix<T>(basis_size);
        get_global_block(1, 2) = -(delta_t/eps[cell_i])*invM2d_Sx;
        
        get_global_block(2, 0) = (delta_t/mu[cell_i])*invM2d_Sy;
        get_global_block(2, 1) = -(delta_t/mu[cell_i])*invM2d_Sx;
        get_global_block(2, 2) = blaze::IdentityMatrix<T>(basis_size);

        auto fcs = faces(msh, tcl);
        for (auto& fc : fcs)
        {
            auto nv = neighbour_via(msh, tcl, fc);
            auto ncl = nv.first;
            auto nbasis = yaourt::bases::make_basis(msh, ncl, degree);
            assert(tbasis.size() == nbasis.size());

            auto n       = normal(msh, tcl, fc);
            auto nx      = n[0];
            auto ny      = n[1];

            blaze::DynamicMatrix<T> FC_diag(3*tbasis.size(), 3*tbasis.size(), 0.0);
            blaze::DynamicMatrix<T> FC_offdiag(3*tbasis.size(), 3*tbasis.size(), 0.0);

            auto get_block = [&](blaze::DynamicMatrix<T>& M, size_t i, size_t j) -> auto {
                auto offset_i = i*basis_size;
                auto offset_j = j*basis_size;
                return submatrix(M, offset_i, offset_j, basis_size, basis_size);
            };

            auto f_qps = yaourt::quadratures::integrate(msh, fc, 2*degree);
            for (auto& fqp : f_qps)
            {
                auto ep = fqp.point();

                if (nv.second)
                {   /* NOT on a boundary */
                    size_t neigh_ofs = offset(msh, ncl);

                    auto Z_minus = sqrt(mu[cell_i]/eps[cell_i]);
                    auto Z_plus  = sqrt(mu[neigh_ofs]/eps[neigh_ofs]);
                    auto Z_avg   = (Z_plus + Z_minus)/2.;

                    auto Y_minus = 1./Z_minus;
                    auto Y_plus  = 1./Z_plus;
                    auto Y_avg   = (Y_plus + Y_minus)/2.;

                    auto tphi  = tbasis.eval(ep);
                    auto tmass = fqp.weight() * tphi * trans(tphi);

                    get_block(FC_diag, 0, 2) -= 0.5*(delta_t/eps[cell_i]) * tmass;
                    get_block(FC_diag, 1, 2) += 0.5*(delta_t/eps[cell_i]) * tmass;
                    get_block(FC_diag, 2, 0) += 0.5*(delta_t/mu[cell_i]) * (-ny) * tmass;
                    get_block(FC_diag, 2, 1) += 0.5*(delta_t/mu[cell_i]) * (+nx) * tmass;


                    /*
                    get_block(FC_diag, 0, 0) += (delta_t/eps[cell_i]) * (ny/Z_avg) * (-ny) * tmass;
                    get_block(FC_diag, 0, 1) += (delta_t/eps[cell_i]) * (ny/Z_avg) * (nx) * tmass;
                    get_block(FC_diag, 0, 2) += (delta_t/eps[cell_i]) * (ny/Z_avg) * (-Z_plus) * tmass;

                    get_block(FC_diag, 1, 0) += (delta_t/eps[cell_i]) * (nx/Z_avg) * (ny) * tmass;
                    get_block(FC_diag, 1, 1) += (delta_t/eps[cell_i]) * (nx/Z_avg) * (-nx) * tmass;
                    get_block(FC_diag, 1, 2) += (delta_t/eps[cell_i]) * (nx/Z_avg) * (Z_plus) * tmass;

                    get_block(FC_diag, 2, 0) += (delta_t/mu[cell_i]) * (1./Y_avg) * (-Y_plus*ny) * tmass;
                    get_block(FC_diag, 2, 1) += (delta_t/mu[cell_i]) * (1./Y_avg) * (Y_plus*nx) * tmass;
                    get_block(FC_diag, 2, 2) += (delta_t/mu[cell_i]) * (1./Y_avg) * (1.) * tmass;
                    */

                    auto nphi  = nbasis.eval(ep);
                    auto nmass = fqp.weight() * tphi * trans(nphi);

                    get_block(FC_diag, 0, 2) += 0.5*(delta_t/eps[cell_i]) * nmass;
                    get_block(FC_diag, 1, 2) -= 0.5*(delta_t/eps[cell_i]) * nmass;
                    get_block(FC_diag, 2, 0) -= 0.5*(delta_t/mu[cell_i]) * (-ny) * nmass;
                    get_block(FC_diag, 2, 1) -= 0.5*(delta_t/mu[cell_i]) * (+nx) * nmass;
                    /*
                    get_block(FC_offdiag, 0, 0) += (delta_t/eps[cell_i]) * (ny/Z_avg) * (-ny) * nmass;
                    get_block(FC_offdiag, 0, 1) += (delta_t/eps[cell_i]) * (ny/Z_avg) * (nx) * nmass;
                    get_block(FC_offdiag, 0, 2) += (delta_t/eps[cell_i]) * (ny/Z_avg) * (-Z_plus) * nmass;

                    get_block(FC_offdiag, 1, 0) += (delta_t/eps[cell_i]) * (nx/Z_avg) * (ny) * nmass;
                    get_block(FC_offdiag, 1, 1) += (delta_t/eps[cell_i]) * (nx/Z_avg) * (-nx) * nmass;
                    get_block(FC_offdiag, 1, 2) += (delta_t/eps[cell_i]) * (nx/Z_avg) * (Z_plus) * nmass;

                    get_block(FC_offdiag, 2, 0) += (delta_t/mu[cell_i]) * (1./Y_avg) * (-Y_plus*ny) * nmass;
                    get_block(FC_offdiag, 2, 1) += (delta_t/mu[cell_i]) * (1./Y_avg) * (Y_plus*nx) * nmass;
                    get_block(FC_offdiag, 2, 2) += (delta_t/mu[cell_i]) * (1./Y_avg) * (1.) * nmass;
                    */
                }
                else
                {   /* On a boundary*/
                    auto Z = sqrt(mu[cell_i]/eps[cell_i]);
                    auto Y = 1./Z;

                    auto tphi  = tbasis.eval(ep);
                    auto tmass = fqp.weight() * tphi * trans(tphi);

                    //get_block(FC_diag, 0, 2) -= 2*(delta_t/eps[cell_i]) * tmass;
                    //get_block(FC_diag, 1, 2) += 2*(delta_t/eps[cell_i]) * tmass;
                    //get_block(FC_diag, 2, 0) += 2*(delta_t/mu[cell_i]) * (+ny) * tmass;
                    //get_block(FC_diag, 2, 1) += 2*(delta_t/mu[cell_i]) * (+nx) * tmass;
                    
                    /*
                    get_block(FC_diag, 0, 0) += 2. * (delta_t/eps[cell_i]) * (ny/Z) * (-ny) * tmass;
                    get_block(FC_diag, 0, 1) += 2. * (delta_t/eps[cell_i]) * (ny/Z) * (nx) * tmass;
                    get_block(FC_diag, 0, 2) += 2. * (delta_t/eps[cell_i]) * (ny) * (-1.) * tmass;
                    get_block(FC_diag, 1, 0) += 2. * (delta_t/eps[cell_i]) * (nx/Z) * (ny) * tmass;
                    get_block(FC_diag, 1, 1) += 2. * (delta_t/eps[cell_i]) * (nx/Z) * (-nx) * tmass;
                    get_block(FC_diag, 1, 2) += 2. * (delta_t/eps[cell_i]) * (nx) * (1.) * tmass;
                    //get_block(FC_diag, 2, 0) += (delta_t/mu[cell_i]) * (1.) * (-1.*ny) * tmass;
                    //get_block(FC_diag, 2, 1) += (delta_t/mu[cell_i]) * (1.) * (nx) * tmass;
                    //get_block(FC_diag, 2, 2) += (delta_t/mu[cell_i]) * (1./Y) * (1.) * tmass;
                    */
                }

                
            }

            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    blaze::DynamicMatrix<T> blk = get_block(FC_diag, i, j);
                    get_global_block(i, j) += solve(M2d, blk);
                }
            }

            if (nv.second)
            {   /* Save offdiag */
                auto offdiag_base = 3*basis_size*offdiag_contrib_i;

                for (size_t i = 0; i < 3; i++)
                {
                    for (size_t j = 0; j < 3; j++)
                    {
                        auto offset_i = offdiag_base + basis_size*i;
                        auto offset_j = basis_size*j;
                        blaze::DynamicMatrix<T> blk = get_block(FC_offdiag, i, j);
                        submatrix(global_offdiag_contribs, offset_i, offset_j, basis_size, basis_size) = solve(M2d, blk);
                    }
                }
            }

            /* LAST */
            offdiag_contrib_i++;
        }        

        /* LAST */
        cell_i++;
    }

    for (size_t t = 0; t < timesteps; t++)
    {
        std::cout << "Time: " << t*delta_t << std::endl;


#ifdef WITH_SILO
        blaze::DynamicVector<T> Ex(msh.cells.size(), 0.0);
        blaze::DynamicVector<T> Ey(msh.cells.size(), 0.0);
        blaze::DynamicVector<T> Hz(msh.cells.size(), 0.0);
#endif

        auto source = [&](const typename Mesh::point_type& pt, double t) -> auto {
            auto f = 1;
            return std::sin(2.0*M_PI*pt.x()) * std::sin(2*M_PI*f*t);
        };

        cell_i = 0;
        offdiag_contrib_i = 0;
        for (auto& tcl : msh.cells)
        {
            auto tbasis = yaourt::bases::make_basis(msh, tcl, degree);
            blaze::DynamicVector<T> loc_rhs(tbasis.size(), 0.0);
            auto qps = yaourt::quadratures::integrate(msh, tcl, 2*degree);
            for (auto& qp : qps)
            {
                auto ep   = qp.point();
                auto phi  = tbasis.eval(ep);
                loc_rhs += source(qp.point(), t*delta_t) * qp.weight() * phi;
            }

            blaze::DynamicMatrix<T> M2d = submatrix(global_M2d, cell_i*basis_size, 0, basis_size, basis_size);

            subvector(global_dofs_T, 3*cell_i*basis_size, basis_size) = 0.;
            subvector(global_dofs_T, 3*cell_i*basis_size+basis_size, basis_size) = solve(M2d, loc_rhs);

            auto get_dofs = [&](blaze::DynamicVector<T>& v, size_t elem) -> auto {
                return subvector(v, 3*basis_size*elem, 3*basis_size);
            };

            auto get_diag = [&]() -> auto {
                return submatrix(global_M, 3*basis_size*cell_i, 0, 3*basis_size, 3*basis_size);
            };

            auto get_offdiag = [&]() -> auto {
                return submatrix(global_offdiag_contribs, 3*basis_size*offdiag_contrib_i, 0, 3*basis_size, 3*basis_size);
            };

            get_dofs(global_dofs_T_plus_one, cell_i) = get_diag() * get_dofs(global_dofs_T, cell_i);

            auto fcs = faces(msh, tcl);
            for (auto& fc : fcs)
            {
                auto nv = neighbour_via(msh, tcl, fc);
                auto ncl = nv.first;
                if (nv.second)
                {
                    auto neigh_ofs = offset(msh, ncl);
                    get_dofs(global_dofs_T_plus_one, cell_i) += get_offdiag() * get_dofs(global_dofs_T, neigh_ofs);
                }
                /* LAST */
                offdiag_contrib_i++;
            }

#ifdef WITH_SILO
            Ex[cell_i] = global_dofs_T[cell_i*3*basis_size];
            Ey[cell_i] = global_dofs_T[cell_i*3*basis_size + basis_size];
            Hz[cell_i] = global_dofs_T[cell_i*3*basis_size + 2*basis_size];
#endif

            /* LAST */
            cell_i++;
        }

#ifdef WITH_SILO
        std::stringstream ss;
        ss << "maxwell_TE_" << t << ".silo";

        yaourt::dataio::silo_database silo;
        silo.create( ss.str() );
        silo.add_mesh(msh, "mesh_TE");

        silo.add_zonal_variable("mesh_TE", "Ex", Ex);
        silo.add_zonal_variable("mesh_TE", "Ey", Ey);
        silo.add_zonal_variable("mesh_TE", "Hz", Hz);
#endif
        global_dofs_T = global_dofs_T_plus_one;
    }   

    return status;
}















/*******************************************************************************************
 * MAXWELL TM SOLVER
 *******************************************************************************************/

template<typename Mesh>
solver_status<typename Mesh::coordinate_type>
run_maxwell_TM_solver(Mesh& msh, const dg_config<typename Mesh::coordinate_type>& cfg)
{
    using mesh_type = Mesh;
    using T = typename mesh_type::coordinate_type;

    solver_status<T>    status;
    status.mesh_h = diameter(msh);

    msh.compute_connectivity();

    size_t degree = cfg.degree;

    size_t faces_per_elem = 0;
    if (std::is_same<Mesh, yaourt::simplicial_mesh<T>>::value)
        faces_per_elem = 3;
    if (std::is_same<Mesh, yaourt::quad_mesh<T>>::value)
        faces_per_elem = 4;

    /* Material parameters */
    std::vector<T> mu( msh.cells.size() );
    std::vector<T> eps( msh.cells.size() );

    for (size_t i = 0; i < msh.cells.size(); i++)
    {
        mu[i] = 1;
        eps[i] = 1;
    }

    /* Time parameters */
    size_t timesteps = cfg.timesteps;
    T delta_t = cfg.delta_t;

    /* Global data vectors */
    auto basis_size = yaourt::bases::scalar_basis_size(degree, 2);

    blaze::DynamicVector<T> global_dofs_T(3*basis_size*msh.cells.size(), 0.0);
    blaze::DynamicVector<T> global_dofs_T_plus_one(3*basis_size*msh.cells.size(), 0.0);

    blaze::DynamicMatrix<T> global_M2d(basis_size*msh.cells.size(), basis_size, 0.0);
    blaze::DynamicMatrix<T> global_M(3*basis_size*msh.cells.size(), 3*basis_size, 0.0);
    blaze::DynamicMatrix<T> global_offdiag_contribs(faces_per_elem*3*basis_size*msh.cells.size(), 3*basis_size, 0.0);

    // TEST: stiffness matrix
    //blaze::DynamicMatrix<T> global_invM2d_Sx(basis_size*msh.cells.size(), basis_size, 0.0);
    //blaze::DynamicMatrix<T> global_invM2d_Sy(basis_size*msh.cells.size(), basis_size, 0.0);
    
    size_t cell_i = 0;
    size_t offdiag_contrib_i = 0;
    for (auto& tcl : msh.cells)
    {
        auto tbasis = yaourt::bases::make_basis(msh, tcl, degree);

        blaze::DynamicMatrix<T> M2d(tbasis.size(), tbasis.size(), 0.0);
        blaze::DynamicMatrix<T> Sx(tbasis.size(), tbasis.size(), 0.0);
        blaze::DynamicMatrix<T> Sy(tbasis.size(), tbasis.size(), 0.0);
        
        /* Make mass and stiffness matrices on the element */
        auto qps = yaourt::quadratures::integrate(msh, tcl, 2*degree);
        for (auto& qp : qps)
        {
            auto ep   = qp.point();
            auto phi  = tbasis.eval(ep);
            auto dphi = tbasis.eval_grads(ep);

            auto dphi_x = blaze::column<0>(dphi);
            auto dphi_y = blaze::column<1>(dphi);

            /* Mass */
            M2d += qp.weight() * phi * trans( phi );
            /* Stiffness, direction x */
            Sx += qp.weight() * phi * trans( dphi_x );
            /* Stiffness, direction y */
            Sy += qp.weight() * phi * trans( dphi_y );
        }

        /* Save local mass matrix, will be needed in iteration */
        submatrix(global_M2d, cell_i*basis_size, 0, basis_size, basis_size) = M2d;

        auto get_global_block = [&](size_t i, size_t j) -> auto {
            auto offset_i = 3*basis_size*cell_i + i*basis_size;
            auto offset_j = j*basis_size;
            return submatrix(global_M, offset_i, offset_j, basis_size, basis_size);
        };

        auto invM2d_Sx  = blaze::solve(M2d, Sx);
        auto invM2d_Sy  = blaze::solve(M2d, Sy);
        auto dt_eps     = delta_t/eps[cell_i];
        auto dt_mu      = delta_t/mu[cell_i];

        // TEST: stiffness matrix
        //submatrix(global_invM2d_Sx, cell_i*basis_size, 0, basis_size, basis_size) = invM2d_Sx;
        //submatrix(global_invM2d_Sy, cell_i*basis_size, 0, basis_size, basis_size) = invM2d_Sy;

        
        get_global_block(0, 0) = blaze::IdentityMatrix<T>(basis_size);
        get_global_block(0, 2) = -dt_mu * invM2d_Sy;

        get_global_block(1, 1) = blaze::IdentityMatrix<T>(basis_size);
        get_global_block(1, 2) = +dt_mu * invM2d_Sx;

        get_global_block(2, 0) = -dt_eps * invM2d_Sy;
        get_global_block(2, 1) = +dt_eps * invM2d_Sx;
        get_global_block(2, 2) = blaze::IdentityMatrix<T>(basis_size);

        /* Do numerical fluxes */
        auto fcs = faces(msh, tcl);
        for (auto& fc : fcs)
        {
            auto nv = neighbour_via(msh, tcl, fc);
            auto ncl = nv.first;
            auto nbasis = yaourt::bases::make_basis(msh, ncl, degree);
            assert(tbasis.size() == nbasis.size());

            auto n       = normal(msh, tcl, fc);
            auto nx      = n[0];
            auto ny      = n[1];

            blaze::DynamicMatrix<T> FC_diag(3*tbasis.size(), 3*tbasis.size(), 0.0);
            blaze::DynamicMatrix<T> FC_offdiag(3*tbasis.size(), 3*tbasis.size(), 0.0);

            auto get_block = [&](blaze::DynamicMatrix<T>& M, size_t i, size_t j) -> auto {
                auto offset_i = i*basis_size;
                auto offset_j = j*basis_size;
                return submatrix(M, offset_i, offset_j, basis_size, basis_size);
            };

            auto f_qps = yaourt::quadratures::integrate(msh, fc, 2*degree);
            for (auto& fqp : f_qps)
            {
                auto ep = fqp.point();

                if (nv.second)
                {   /* NOT on a boundary */
                    size_t neigh_ofs = offset(msh, ncl);

                    auto tphi  = tbasis.eval(ep);
                    auto tmass = fqp.weight() * tphi * trans(tphi);

                    get_block(FC_diag, 0, 2) += +ny * 0.5 * dt_mu * tmass;
                    get_block(FC_diag, 1, 2) += -nx * 0.5 * dt_mu * tmass;
                    get_block(FC_diag, 2, 0) += +ny * 0.5 * dt_eps * tmass;
                    get_block(FC_diag, 2, 1) += -nx * 0.5 * dt_eps * tmass;


                    auto nphi  = nbasis.eval(ep);
                    auto nmass = fqp.weight() * tphi * trans(nphi);

                    get_block(FC_offdiag, 0, 2) -= +ny * 0.5 * dt_mu * nmass;
                    get_block(FC_offdiag, 1, 2) -= -nx * 0.5 * dt_mu * nmass;
                    get_block(FC_offdiag, 2, 0) -= +ny * 0.5 * dt_eps * nmass;
                    get_block(FC_offdiag, 2, 1) -= -nx * 0.5 * dt_eps * nmass;

                }
                else
                {   /* On a boundary*/
                    auto Z = sqrt(mu[cell_i]/eps[cell_i]);
                    auto Y = 1./Z;

                    auto tphi  = tbasis.eval(ep);
                    auto tmass = fqp.weight() * tphi * trans(tphi);
                    get_block(FC_diag, 0, 0) += nx * dt_mu * tmass;
                    get_block(FC_diag, 1, 0) += nx * dt_mu * tmass;
                    get_block(FC_diag, 0, 1) += ny * dt_mu * tmass;
                    get_block(FC_diag, 1, 1) += ny * dt_mu * tmass;
                    get_block(FC_diag, 2, 2) += dt_eps * tmass;
                }
            }

            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    blaze::DynamicMatrix<T> blk = get_block(FC_diag, i, j);
                    get_global_block(i, j) += solve(M2d, blk);
                }
            }

            if (nv.second)
            {   /* Save offdiag */
                auto offdiag_base = 3*basis_size*offdiag_contrib_i;

                for (size_t i = 0; i < 3; i++)
                {
                    for (size_t j = 0; j < 3; j++)
                    {
                        auto offset_i = offdiag_base + basis_size*i;
                        auto offset_j = basis_size*j;
                        blaze::DynamicMatrix<T> blk = get_block(FC_offdiag, i, j);
                        submatrix(global_offdiag_contribs, offset_i, offset_j, basis_size, basis_size) = solve(M2d, blk);
                    }
                }
            }

            /* LAST */
            offdiag_contrib_i++;
        } //for each face

        /* LAST */
        cell_i++;
    } //for each cell


    auto test_src = [&](const typename Mesh::point_type& pt, T t) -> auto {
        auto f = 1;
        return std::sin(M_PI*pt.y()) * std::cos(2*M_PI*f*t);
    };

    auto test_stiff = [&](const typename Mesh::point_type& pt) -> auto {
        return pt.x() * pt.y();
    };

    auto gOfs_Hx = [&](size_t cell_i) -> auto {
        size_t loc_ofs = 0;
        return 3*cell_i*basis_size + loc_ofs;
    };

    auto gOfs_Hy = [&](size_t cell_i) -> auto {
        size_t loc_ofs = basis_size;
        return 3*cell_i*basis_size + loc_ofs;
    };

    auto gOfs_Ez = [&](size_t cell_i) -> auto {
        size_t loc_ofs = 2*basis_size;
        return 3*cell_i*basis_size + loc_ofs;
    };

    auto m = 1;
    auto n = 1;
    auto omega = M_PI*std::sqrt(m*m + n*n);

    /* Initial conditions */
    auto ic = [&](const typename Mesh::point_type& pt) -> auto {
        return std::sin(m*M_PI*pt.x()) * std::sin(n*M_PI*pt.y());
    };

    /* Reference solution */
    auto Hx_ref = [&](const typename Mesh::point_type& pt, T t) -> auto {
        return -(M_PI*n/omega) * std::sin(m*M_PI*pt.x()) * std::cos(n*M_PI*pt.y()) * std::sin(omega*t);
    };

    auto Hy_ref = [&](const typename Mesh::point_type& pt, T t) -> auto {
        return (M_PI*m/omega) * std::cos(m*M_PI*pt.x()) * std::sin(n*M_PI*pt.y()) * std::sin(omega*t);
    };

    auto Ez_ref = [&](const typename Mesh::point_type& pt, T t) -> auto {
        return std::sin(m*M_PI*pt.x()) * std::sin(n*M_PI*pt.y()) * std::cos(omega*t);
    };


    cell_i = 0;
    for (auto& tcl : msh.cells)
    {
        auto tbasis = yaourt::bases::make_basis(msh, tcl, degree);
        blaze::DynamicVector<T> loc_rhs(tbasis.size(), 0.0);
        auto qps = yaourt::quadratures::integrate(msh, tcl, 2*degree);
        for(auto& qp : qps)
        {
            auto phi = tbasis.eval(qp.point());
            loc_rhs += ic(qp.point()) * qp.weight() * phi;
        }

        blaze::DynamicMatrix<T> M2d = submatrix(global_M2d, cell_i*basis_size, 0, basis_size, basis_size);
        subvector(global_dofs_T, gOfs_Ez(cell_i), basis_size) = solve(M2d, loc_rhs);

        /* LAST */
        cell_i++;
    }

    for (size_t t = 0; t < timesteps; t++)
    {
        std::cout << "Time: " << t*delta_t << std::endl;


#ifdef WITH_SILO
        blaze::DynamicVector<T> Hx(msh.cells.size(), 0.0);
        blaze::DynamicVector<T> Hy(msh.cells.size(), 0.0);
        blaze::DynamicVector<T> Ez(msh.cells.size(), 0.0);

        blaze::DynamicVector<T> Hx_refsol(msh.cells.size(), 0.0);
        blaze::DynamicVector<T> Hy_refsol(msh.cells.size(), 0.0);
        blaze::DynamicVector<T> Ez_refsol(msh.cells.size(), 0.0);
#endif
#if 0
        cell_i = 0;
        for (auto& tcl : msh.cells)
        {
            auto tbasis = yaourt::bases::make_basis(msh, tcl, degree);
            blaze::DynamicVector<T> loc_rhs(tbasis.size(), 0.0);
            auto qps = yaourt::quadratures::integrate(msh, tcl, 2*degree);
            for(auto& qp : qps)
            {
                auto phi = tbasis.eval(qp.point());
                loc_rhs += test_stiff(qp.point()) * qp.weight() * phi;
            }

            // TEST MASS MATRIX
            /*
            auto old_dofs = subvector(global_dofs_T, gOfs_Ez(cell_i), basis_size);
            blaze::DynamicMatrix<T> M2d = submatrix(global_M2d, cell_i*basis_size, 0, basis_size, basis_size);
            auto new_dofs = old_dofs + delta_t * solve(M2d, loc_rhs);
            subvector(global_dofs_T_plus_one, gOfs_Ez(cell_i), basis_size) = new_dofs;
            */

            // TEST STIFFNESS MATRIX
            /*
            blaze::DynamicMatrix<T> M2d = submatrix(global_M2d, cell_i*basis_size, 0, basis_size, basis_size);
            blaze::DynamicMatrix<T> invM2d_Sx = submatrix(global_invM2d_Sx, cell_i*basis_size, 0, basis_size, basis_size);
            blaze::DynamicMatrix<T> invM2d_Sy = submatrix(global_invM2d_Sy, cell_i*basis_size, 0, basis_size, basis_size);
            blaze::DynamicVector<T> dofs_dx = invM2d_Sx * solve(M2d, loc_rhs);
            blaze::DynamicVector<T> dofs_dy = invM2d_Sy * solve(M2d, loc_rhs);
            subvector(global_dofs_T, gOfs_Hx(cell_i), basis_size) = dofs_dx;
            subvector(global_dofs_T, gOfs_Hy(cell_i), basis_size) = dofs_dy;
            */

            /* LAST */
            cell_i++;
        }


        auto source = [&](const typename Mesh::point_type& pt, double t) -> auto {
            auto f = 2;
            return std::sin(M_PI*pt.y()) * std::cos(2*M_PI*f*t);
        };
#endif


        cell_i = 0;
        offdiag_contrib_i = 0;
        for (auto& tcl : msh.cells)
        {
            auto get_dofs = [&](blaze::DynamicVector<T>& v, size_t elem) -> auto {
                return subvector(v, 3*basis_size*elem, 3*basis_size);
            };

            auto get_diag = [&]() -> auto {
                return submatrix(global_M, 3*basis_size*cell_i, 0, 3*basis_size, 3*basis_size);
            };

            auto get_offdiag = [&]() -> auto {
                return submatrix(global_offdiag_contribs, 3*basis_size*offdiag_contrib_i, 0, 3*basis_size, 3*basis_size);
            };

            get_dofs(global_dofs_T_plus_one, cell_i) = get_diag() * get_dofs(global_dofs_T, cell_i);

            auto fcs = faces(msh, tcl);
            for (auto& fc : fcs)
            {
                auto nv = neighbour_via(msh, tcl, fc);
                auto ncl = nv.first;
                if (nv.second)
                {
                    auto neigh_ofs = offset(msh, ncl);
                    get_dofs(global_dofs_T_plus_one, cell_i) += get_offdiag() * get_dofs(global_dofs_T, neigh_ofs);
                    
                }
                /* LAST */
                offdiag_contrib_i++;
            }           

#ifdef WITH_SILO
            Hx[cell_i] = global_dofs_T[cell_i*3*basis_size];
            Hy[cell_i] = global_dofs_T[cell_i*3*basis_size + basis_size];
            Ez[cell_i] = global_dofs_T[cell_i*3*basis_size + 2*basis_size];

            auto bar = barycenter(msh, tcl);

            Hx_refsol[cell_i] = Hx_ref(bar, t*delta_t);
            Hy_refsol[cell_i] = Hy_ref(bar, t*delta_t);
            Ez_refsol[cell_i] = Ez_ref(bar, t*delta_t);
#endif

            /* LAST */
            cell_i++;
        }

#ifdef WITH_SILO
        std::stringstream ss;
        ss << "maxwell_TM_" << t << ".silo";

        yaourt::dataio::silo_database silo;
        silo.create( ss.str() );
        silo.add_mesh(msh, "mesh_TM");

        silo.add_zonal_variable("mesh_TM", "Hx", Hx);
        silo.add_zonal_variable("mesh_TM", "Hy", Hy);
        silo.add_zonal_variable("mesh_TM", "Ez", Ez);

        silo.add_zonal_variable("mesh_TM", "Hx_refsol", Hx_refsol);
        silo.add_zonal_variable("mesh_TM", "Hy_refsol", Hy_refsol);
        silo.add_zonal_variable("mesh_TM", "Ez_refsol", Ez_refsol);
#endif
        global_dofs_T = global_dofs_T_plus_one;
    }   

    return status; 
}


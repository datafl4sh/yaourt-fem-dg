/*
 * Yaourt-FEM-DG - Yet AnOther Useful Resource for Teaching FEM and DG.
 *
 * Matteo Cicuttin (C) 2019-2020
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
#include <getopt.h>

#include "methods/dg_maxwell_2D.hpp"

namespace ymax = yaourt::maxwell_2D;

#ifdef WITH_SILO
template<typename Mesh>
bool
export_data_to_silo(ymax::maxwell_context<Mesh>& ctx, size_t cycle,
                    const ymax::ICF_type<Mesh>& Hx_ref,
                    const ymax::ICF_type<Mesh>& Hy_ref,
                    const ymax::ICF_type<Mesh>& Ez_ref)
{
    if (!ctx.cfg.silo_basename)
        return false;

    namespace yb = yaourt::bases;
    auto basis_size = yb::scalar_basis_size(ctx.cfg.degree, 2);

    using T = typename Mesh::coordinate_type;
    using namespace blaze;
    DynamicVector<T> Hx(ctx.msh.cells.size(), 0.0);
    DynamicVector<T> Hy(ctx.msh.cells.size(), 0.0);
    DynamicVector<T> Ez(ctx.msh.cells.size(), 0.0);

    DynamicVector<T> Hx_refsol(ctx.msh.cells.size(), 0.0);
    DynamicVector<T> Hy_refsol(ctx.msh.cells.size(), 0.0);
    DynamicVector<T> Ez_refsol(ctx.msh.cells.size(), 0.0);

    size_t cell_i = 0;
    for (auto& tcl : ctx.msh.cells)
    {          
        Hx[cell_i] = ctx.gDofs[cell_i*3*basis_size];
        Hy[cell_i] = ctx.gDofs[cell_i*3*basis_size + basis_size];
        Ez[cell_i] = ctx.gDofs[cell_i*3*basis_size + 2*basis_size];

        auto bar = barycenter(ctx.msh, tcl);
        Hx_refsol[cell_i] = Hx_ref(bar, cycle*ctx.cfg.delta_t);
        Hy_refsol[cell_i] = Hy_ref(bar, cycle*ctx.cfg.delta_t);
        Ez_refsol[cell_i] = Ez_ref(bar, cycle*ctx.cfg.delta_t);


        /* LAST */
        cell_i++;
    }


    //std::cout << "Time: " << t*delta_t << ", step " << t+1 << " of " << timesteps << std::endl;

    std::stringstream ss;
    ss << ctx.cfg.silo_basename << "_" << cycle << ".silo";

    yaourt::dataio::silo_database silo;

    silo.create( ss.str() );

    silo.add_time(cycle, cycle*ctx.cfg.delta_t);

    const char *mesh_name = "mesh_maxwell2D";

    silo.add_mesh(ctx.msh, mesh_name);

    silo.add_zonal_variable(mesh_name, "Hx", Hx);
    silo.add_zonal_variable(mesh_name, "Hy", Hy);
    silo.add_zonal_variable(mesh_name, "Ez", Ez);
    silo.add_expression("H", "{Hx, Hy}", DB_VARTYPE_VECTOR);

    silo.add_zonal_variable(mesh_name, "Hx_refsol", Hx_refsol);
    silo.add_zonal_variable(mesh_name, "Hy_refsol", Hy_refsol);
    silo.add_zonal_variable(mesh_name, "Ez_refsol", Ez_refsol);
    silo.add_expression("H_refsol", "{Hx_refsol, Hy_refsol}", DB_VARTYPE_VECTOR);

    silo.add_expression("Hx_diff", "Hx-Hx_refsol", DB_VARTYPE_SCALAR);
    silo.add_expression("Hy_diff", "Hy-Hy_refsol", DB_VARTYPE_SCALAR);
    silo.add_expression("Ez_diff", "Ez-Ez_refsol", DB_VARTYPE_SCALAR);

    silo.add_expression("H_diff", "{Hx_diff, Hy_diff}", DB_VARTYPE_VECTOR);

    return true;
}
#endif

template<typename T>
struct error_info
{
    size_t  cycle;
    T       time;
    T       Hx_err;
    T       Hy_err;
    T       Ez_err;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const error_info<T>& ei)
{
    os << ei.cycle << " ";
    os << ei.time << " ";
    os << ei.Hx_err << " ";
    os << ei.Hy_err << " ";
    os << ei.Ez_err;
    return os;
}

template<typename Mesh>
error_info<typename Mesh::coordinate_type>
report_errors(ymax::maxwell_context<Mesh>& ctx, size_t cycle,
              const ymax::ICF_type<Mesh>& Hx_ref,
              const ymax::ICF_type<Mesh>& Hy_ref,
              const ymax::ICF_type<Mesh>& Ez_ref)
{
    namespace yb = yaourt::bases;
    namespace yq = yaourt::quadratures;
    auto basis_size = yb::scalar_basis_size(ctx.cfg.degree, 2);

    using T = typename Mesh::coordinate_type;
    using namespace blaze;

    T Hx_err = 0.0;
    T Hy_err = 0.0;
    T Ez_err = 0.0;

    size_t cell_i = 0;
    for (auto& tcl : ctx.msh.cells)
    {
        auto dofs_ofs = 3*basis_size*cell_i;

        auto local_dofs = subvector(ctx.gDofs, dofs_ofs, 3*basis_size);
        auto local_Hx_dofs = subvector(local_dofs, 0, basis_size);
        auto local_Hy_dofs = subvector(local_dofs, basis_size, basis_size);
        auto local_Ez_dofs = subvector(local_dofs, 2*basis_size, basis_size);

        auto tbasis = yb::make_basis(ctx.msh, tcl, ctx.cfg.degree);
        auto qps = yq::integrate(ctx.msh, tcl, 2*ctx.cfg.degree);
        for (auto& qp : qps)
        {
            auto phi = tbasis.eval(qp.point());
            
            auto Hx_num = dot(local_Hx_dofs, phi);
            auto Hx_ana = Hx_ref(qp.point(), cycle*ctx.cfg.delta_t);
            Hx_err  += qp.weight() * (Hx_num-Hx_ana)*(Hx_num-Hx_ana);

            auto Hy_num = dot(local_Hy_dofs, phi);
            auto Hy_ana = Hy_ref(qp.point(), cycle*ctx.cfg.delta_t);
            Hy_err += qp.weight() * (Hy_num-Hy_ana)*(Hy_num-Hy_ana);

            auto Ez_num = dot(local_Ez_dofs, phi);
            auto Ez_ana = Ez_ref(qp.point(), cycle*ctx.cfg.delta_t);
            Ez_err += qp.weight() * (Ez_num-Ez_ana)*(Ez_num-Ez_ana);
        }

        /* LAST */
        cell_i++;
    }

    error_info<T> ei;
    ei.cycle    = cycle;
    ei.time     = cycle*ctx.cfg.delta_t;
    ei.Hx_err   = std::sqrt(Hx_err);
    ei.Hy_err   = std::sqrt(Hy_err);
    ei.Ez_err   = std::sqrt(Ez_err);

    return ei;
}

template<typename Mesh>
void
run_maxwell_solver(const ymax::maxwell_config<typename Mesh::coordinate_type>& cfg)
{
    using point_type = typename Mesh::point_type;
    using T = typename Mesh::coordinate_type;
    namespace yb = yaourt::bases;

    /* Reference solution */
    size_t m = 1;
    size_t n = 1;
    auto omega = M_PI*std::sqrt(m*m + n*n);
    
    auto Hx_ref = [&](const point_type& pt, T t) -> auto {
        return -(M_PI*n/omega) * std::sin(m*M_PI*pt.x()) * std::cos(n*M_PI*pt.y()) * std::sin(omega*t);
    };

    auto Hy_ref = [&](const point_type& pt, T t) -> auto {
        return (M_PI*m/omega) * std::cos(m*M_PI*pt.x()) * std::sin(n*M_PI*pt.y()) * std::sin(omega*t);
    };

    auto Ez_ref = [&](const point_type& pt, T t) -> auto {
        return std::sin(m*M_PI*pt.x()) * std::sin(n*M_PI*pt.y()) * std::cos(omega*t) * pt.x();
    };

    /* Init solver context */
    ymax::maxwell_context<Mesh> ctx(cfg);

    auto basis_size = yb::scalar_basis_size(ctx.cfg.degree, 2);
    std::ofstream err_ofs;
    if (ctx.cfg.error_fn)
    {
        err_ofs.open(ctx.cfg.error_fn);
        err_ofs << "# -> Maxwell 2D solver <- " << std::endl;
        err_ofs << "# mesh levels:      " << ctx.cfg.mesh_levels << std::endl;
        err_ofs << "# DOFs:             " << 3*ctx.msh.cells.size()*basis_size << std::endl;
        err_ofs << "# degree:           " << ctx.cfg.degree << std::endl;
        err_ofs << "# delta_t:          " << ctx.cfg.delta_t << std::endl;
        err_ofs << "# total steps:      " << ctx.cfg.timesteps << std::endl;
        err_ofs << "# upwind:           " << (ctx.cfg.upwind ? "yes" : "no") << std::endl;

        if (ctx.cfg.time_integrator == ymax::time_integrator_type::EXPLICIT_EULER)
            err_ofs << "# time integrator:  " << "Explicit Euler (you're doing the wrong thing)" << std::endl;

        if (ctx.cfg.time_integrator == ymax::time_integrator_type::RUNGE_KUTTA_4)
            err_ofs << "# time integrator:  " << "4th order Runge-Kutta" << std::endl;
    }

    assemble(ctx);
    apply_initial_condition(ctx, Hx_ref, Hy_ref, Ez_ref);

    auto last_output_time = std::chrono::system_clock::now();
    for (size_t cycle = 0; cycle < ctx.cfg.timesteps; cycle++)
    {
        do_timestep(ctx);

        bool do_output = (cycle % ctx.cfg.output_rate == 0);

#ifdef WITH_SILO
        if (ctx.cfg.silo_basename and do_output)
            export_data_to_silo(ctx, cycle, Hx_ref, Hy_ref, Ez_ref);
#endif
        if (ctx.cfg.verbosity > 0 and do_output)
        {
            std::cout << "Cycle " << cycle << ", t=" << cycle*ctx.cfg.delta_t << ". ";

            auto now = std::chrono::system_clock::now();
            std::chrono::duration<double> time = now - last_output_time;
            std::cout << "Time since last output: " << time.count() << " seconds" << std::endl;
            last_output_time = now;
        }

        if (ctx.cfg.error_fn)
        {
            auto ei = report_errors(ctx, cycle, Hx_ref, Hy_ref, Ez_ref);
            err_ofs << ei << std::endl;
        }
    }
}

enum class mesh_type  {
    TRIANGULAR,
    QUADRANGULAR,
    TETRAHEDRAL,
    HEXAHEDRAL
};

int main(int argc, char **argv)
{
    using T = double;

    mesh_type mt = mesh_type::TRIANGULAR;
    ymax::maxwell_config<T> cfg;
    int ch;

    static struct option longopts[] = {
        { "mesh-type",              required_argument,  NULL, 'm' },
        { "mesh-refinement",        required_argument,  NULL, 'r' },
        { "degree",                 required_argument,  NULL, 'k' },
        { "time-integrator",        required_argument,  NULL, 'i' },
        { "deltat",                 required_argument,  NULL, 'd' },
        { "timesteps",              required_argument,  NULL, 't' },
        { "max-time",               required_argument,  NULL, 'T' },
        { "error-log",              required_argument,  NULL, 'E' },
        { "silo-basename",          required_argument,  NULL, 's' },
        { "use-upwind",             no_argument,        NULL, 'u' },
        { "verbose",                no_argument,        NULL, 'v' },
        { NULL,                     0,                  NULL,  0  }
    };

    while ((ch = getopt_long(argc, argv, "m:r:k:i:d:t:T:E:s:uv", longopts, NULL)) != -1)
    {
        switch (ch)
        {
            case 'm':
                if ( strcmp(optarg, "quad") == 0 )
                    mt = mesh_type::QUADRANGULAR;
                else if ( strcmp(optarg, "tri") == 0 )
                    mt = mesh_type::TRIANGULAR;
                break;

            case 'r':
                cfg.mesh_levels = atoi(optarg);
                break;

            case 'k':
                cfg.degree = atoi(optarg);
                break;

            case 'i':
                if ( strcmp(optarg, "euler") == 0 )
                    cfg.time_integrator = ymax::time_integrator_type::EXPLICIT_EULER;
                break;
            
            case 'd':
                cfg.delta_t = atof(optarg);
                break;

            case 't':
                cfg.timesteps = atoi(optarg);
                break;

            case 'T': {
                float maxtime = atof(optarg);
                cfg.timesteps = (size_t) std::ceil(maxtime/cfg.delta_t);
                }
                break;

            case 'E':
                cfg.error_fn = optarg;
                break;

            case 's':
                cfg.silo_basename = optarg;
                break;

            case 'u':
                cfg.upwind = true;
                break;

            case 'v':
                cfg.verbosity++;
                break;

            case 0:
                break;
        
            default:
                //usage();
                break;
        }
    }

    argc -= optind;
    argv += optind;

    if (mt == mesh_type::TRIANGULAR)
        run_maxwell_solver<yaourt::simplicial_mesh<T>>(cfg);
    else if (mt == mesh_type::QUADRANGULAR)
        run_maxwell_solver<yaourt::quad_mesh<T>>(cfg);

    return 0;
}


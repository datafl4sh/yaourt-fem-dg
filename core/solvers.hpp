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

#include <blaze/Math.h>
#include <fstream>
#include <string>

template<typename T>
struct conjugated_gradient_params
{
    T               rr_tol;
    T               rr_max;
    size_t          max_iter;
    bool            verbose;
    bool            save_iteration_history;
    bool            use_initial_guess;
    bool            use_normal_eqns;
    std::string     history_filename;

    conjugated_gradient_params() : rr_tol(1e-8),
                                   rr_max(20),
                                   max_iter(100),
                                   verbose(false),
                                   save_iteration_history(false),
                                   use_initial_guess(false),
                                   use_normal_eqns(false) {}
};

// TODO: return false and some kind of error in case of non convergence.
template<typename T>
bool
conjugated_gradient(const conjugated_gradient_params<T>& cgp,
                    const blaze::CompressedMatrix<T>& A,
                    const blaze::DynamicVector<T>& b,
                    blaze::DynamicVector<T>& x)
{
    if ( A.rows() != A.columns() )
    {
        if (cgp.verbose)
            std::cout << "[CG solver] A square matrix is required" << std::endl;

        return false;
    }

    size_t N = A.columns();

    if (b.size() != N)
    {
        if (cgp.verbose)
            std::cout << "[CG solver] Wrong size of RHS vector" << std::endl;

        return false;
    }

    if (x.size() != N)
    {
        if (cgp.verbose)
            std::cout << "[CG solver] Wrong size of solution vector" << std::endl;

        return false;
    }

    if (!cgp.use_initial_guess)
        x = blaze::DynamicVector<T>(N, 0.0);

    size_t                      iter = 0;
    T                           nr, nr0;
    T                           alpha, beta, rho;

    blaze::DynamicVector<T> d(N), r(N), r0(N), y(N);

    if (cgp.use_normal_eqns)
        r0 = d = r = trans(A)*(b - A*x);
    else
        r0 = d = r = b - A*x;

    nr = nr0 = norm(r);

    std::ofstream iter_hist_ofs;
    if (cgp.save_iteration_history)
        iter_hist_ofs.open(cgp.history_filename);

    while ( nr/nr0 > cgp.rr_tol && iter < cgp.max_iter && nr/nr0 < cgp.rr_max )
    {
        if (cgp.verbose )
        {
            std::cout << "                                                 \r";
            std::cout << " -> Iteration " << iter << ", rr = ";
            std::cout << nr/nr0 << "\b\r";
            std::cout.flush();
        }

        if (cgp.save_iteration_history)
            iter_hist_ofs << nr/nr0 << std::endl;

        if (cgp.use_normal_eqns)
            y = trans(A)*A*d;
        else
            y = A*d;

        rho = dot(r,r);
        alpha = rho/dot(d,y);
        x = x + alpha * d;
        r = r - alpha * y;
        beta = dot(r,r)/rho;
        d = r + beta * d;

        nr = norm(r);
        iter++;
    }

    if (cgp.save_iteration_history)
    {
        iter_hist_ofs << nr/nr0 << std::endl;
        iter_hist_ofs.close();
    }

    if (cgp.verbose)
        std::cout << " -> Iteration " << iter << ", rr = " << nr/nr0 << std::endl;

    return true;
}

// TODO: return false and some kind of error in case of non convergence.
template<typename T>
bool
conjugated_gradient(const conjugated_gradient_params<T>& cgp,
                    const blaze::CompressedMatrix<T>& A,
                    const blaze::DynamicVector<T>& b,
                    blaze::DynamicVector<T>& x,
                    const blaze::CompressedMatrix<T>& iM)
{
    if ( A.rows() != A.columns() )
    {
        if (cgp.verbose)
            std::cout << "[CG solver] A square matrix is required" << std::endl;

        return false;
    }

    size_t N = A.columns();

    if (b.size() != N)
    {
        if (cgp.verbose)
            std::cout << "[CG solver] Wrong size of RHS vector" << std::endl;

        return false;
    }

    if (x.size() != N)
    {
        if (cgp.verbose)
            std::cout << "[CG solver] Wrong size of solution vector" << std::endl;

        return false;
    }

    if (!cgp.use_initial_guess)
        x = blaze::DynamicVector<T>(N, 0.0);

    size_t                      iter = 0;
    T                           nr, nr0;
    T                           alpha, beta, rho;

    blaze::DynamicVector<T> d(N), r(N), r0(N), y(N);
    blaze::CompressedMatrix<T> real_iM;

    if (cgp.use_normal_eqns)
    {
        r0 = r = trans(A)*(b - A*x);
        real_iM = iM;
    }
    else
    {
        r0 = r = b - A*x;
        real_iM = iM;
    }

    d = real_iM * r0;

    nr = nr0 = norm(r);

    std::ofstream iter_hist_ofs;
    if (cgp.save_iteration_history)
        iter_hist_ofs.open(cgp.history_filename);

    while ( nr/nr0 > cgp.rr_tol && iter < cgp.max_iter && nr/nr0 < cgp.rr_max )
    {
        if (cgp.verbose )
        {
            std::cout << "                                                 \r";
            std::cout << " -> Iteration " << iter << ", rr = ";
            std::cout << nr/nr0 << "\b\r";
            std::cout.flush();
        }

        if (cgp.save_iteration_history)
            iter_hist_ofs << nr/nr0 << std::endl;

        if (cgp.use_normal_eqns)
            y = trans(A)*A*d;
        else
            y = A*d;

        rho = dot(r,real_iM*r);
        alpha = rho/dot(d,y);
        x = x + alpha * d;
        r = r - alpha * y;
        beta = dot(r,real_iM*r)/rho;
        d = real_iM*r + beta * d;

        nr = norm(r);
        iter++;
    }

    if (cgp.save_iteration_history)
    {
        iter_hist_ofs << nr/nr0 << std::endl;
        iter_hist_ofs.close();
    }

    if (cgp.verbose)
        std::cout << " -> Iteration " << iter << ", rr = " << nr/nr0 << std::endl;

    return true;
}

// TODO: return false and some kind of error in case of non convergence.
template<typename T>
bool
bicgstab(const conjugated_gradient_params<T>& cgp,
         const blaze::CompressedMatrix<T>& A,
         const blaze::DynamicVector<T>& b,
         blaze::DynamicVector<T>& x)
{
    if ( A.rows() != A.columns() )
    {
        if (cgp.verbose)
            std::cout << "[CG solver] A square matrix is required" << std::endl;

        return false;
    }

    size_t N = A.columns();

    if (b.size() != N)
    {
        if (cgp.verbose)
            std::cout << "[CG solver] Wrong size of RHS vector" << std::endl;

        return false;
    }

    if (x.size() != N)
    {
        if (cgp.verbose)
            std::cout << "[CG solver] Wrong size of solution vector" << std::endl;

        return false;
    }

    if (!cgp.use_initial_guess)
        x = blaze::DynamicVector<T>(N, 0.0);

    size_t  iter = 0;
    T       nr, nr0;
    T       alpha = 1.0, omega = 1.0, rho = 1.0;

    blaze::DynamicVector<T> r(N), r0(N), p(N, 0.0), v(N, 0.0);
    blaze::DynamicVector<T> s(N), t(N);

    if (cgp.use_normal_eqns)
        r0 = r = trans(A)*(b - A*x);
    else
        r0 = r = b - A*x;

    nr = nr0 = norm(r);

    std::ofstream iter_hist_ofs;
    if (cgp.save_iteration_history)
        iter_hist_ofs.open(cgp.history_filename);

    while ( nr/nr0 > cgp.rr_tol && iter < cgp.max_iter && nr/nr0 < cgp.rr_max )
    {
        if ( cgp.verbose )
        {
            std::cout << "                                                 \r";
            std::cout << " -> Iteration " << iter << ", rr = ";
            std::cout << nr/nr0 << "\b\r";
            std::cout.flush();
        }

        if (cgp.save_iteration_history)
            iter_hist_ofs << nr/nr0 << std::endl;

        T rho_old = rho;

        rho = dot(r,r0);
        if ( std::abs(rho) < 1e-9 )
        {
            if (cgp.use_normal_eqns)
                r = b - trans(A)*A*x;
            else
                r = b - A*x;
            r0 = r;
            rho = dot(r,r0);
        }

        T beta = (rho/rho_old)*(alpha/omega);
        p = r + beta * (p - omega*v);

        if (cgp.use_normal_eqns)
            v = trans(A)*A*p;
        else
            v = A*p;

        alpha = rho / dot(v, r0);
        s = r - alpha*v;

        if (cgp.use_normal_eqns)
            t = trans(A)*A*s;
        else
            t = A*s;

        omega = dot(t,s)/dot(t,t);

        x = x + alpha * p + omega * s;
        r = s - omega*t;

        nr = norm(r);
        iter++;
    }

    if (cgp.save_iteration_history)
    {
        iter_hist_ofs << nr/nr0 << std::endl;
        iter_hist_ofs.close();
    }

    if (cgp.verbose)
        std::cout << " -> Iteration " << iter << ", rr = " << nr/nr0 << std::endl;

    return true;
}

template<typename T>
bool
bicgstab(const conjugated_gradient_params<T>& cgp,
         const blaze::CompressedMatrix<T>& A,
         const blaze::DynamicVector<T>& b,
         blaze::DynamicVector<T>& x,
         const blaze::CompressedMatrix<T>& iM)
{
    if ( A.rows() != A.columns() )
    {
        if (cgp.verbose)
            std::cout << "[CG solver] A square matrix is required" << std::endl;

        return false;
    }

    size_t N = A.columns();

    if (b.size() != N)
    {
        if (cgp.verbose)
            std::cout << "[CG solver] Wrong size of RHS vector" << std::endl;

        return false;
    }

    if (x.size() != N)
    {
        if (cgp.verbose)
            std::cout << "[CG solver] Wrong size of solution vector" << std::endl;

        return false;
    }

    if (!cgp.use_initial_guess)
        x = blaze::DynamicVector<T>(N, 0.0);

    size_t  iter = 0;
    T       nr, nr0;
    T       alpha = 1.0, omega = 1.0, rho = 1.0;

    blaze::DynamicVector<T> r(N), r0(N), p(N, 0.0), v(N, 0.0);
    blaze::DynamicVector<T> s(N), t(N);
    blaze::DynamicVector<T> y(N), z(N);

    r0 = r = b - A*x;
    nr = nr0 = norm(r);

    std::ofstream iter_hist_ofs;
    if (cgp.save_iteration_history)
        iter_hist_ofs.open(cgp.history_filename);

    while ( nr/nr0 > cgp.rr_tol && iter < cgp.max_iter && nr/nr0 < cgp.rr_max )
    {
        if ( cgp.verbose )
        {
            std::cout << "                                                 \r";
            std::cout << " -> Iteration " << iter << ", rr = ";
            std::cout << nr/nr0 << "\b\r";
            std::cout.flush();
        }

        if (cgp.save_iteration_history)
            iter_hist_ofs << nr/nr0 << std::endl;

        T rho_old = rho;

        rho = dot(r,r0);
        if ( std::abs(rho) < 1e-9 )
        {
            r = b - A*x;
            r0 = r;
            rho = dot(r,r0);
        }

        T beta = (rho/rho_old)*(alpha/omega);
        p = r + beta * (p - omega*v);
        y = iM*p;
        v = A*y;
        alpha = rho / dot(v, r0);
        s = r - alpha*v;
        z = iM*s;
        t = A*z;

        auto iMt = iM*t;
        omega = dot(iMt,z)/dot(iMt,iMt);

        x = x + alpha * y + omega * z;
        r = s - omega*t;

        nr = norm(r);
        iter++;
    }

    if (cgp.save_iteration_history)
    {
        iter_hist_ofs << nr/nr0 << std::endl;
        iter_hist_ofs.close();
    }

    if (cgp.verbose)
        std::cout << " -> Iteration " << iter << ", rr = " << nr/nr0 << std::endl;

    return true;
}
template<typename T>
bool
qmr(const blaze::CompressedMatrix<T>& A,
    const blaze::DynamicVector<T>& b,
    blaze::DynamicVector<T>& x)
{
    size_t  N = A.columns();
    size_t  iter = 0;
    T       nr, nr0;
    T       alpha, rho, rho1, xi, beta, gamma, gamma1, eta, delta, epsilon, theta, theta1;
    
    blaze::DynamicVector<T> r(N), r0(N);
    blaze::DynamicVector<T> d, s, p, p_tilde, q, v, v_tilde, w, w_tilde, y, y_tilde, z, z_tilde;
    
    r0 = r = b - A*x;
    nr = nr0 = norm(r);
    
    v_tilde = r;
    y = /*eiM **/ v_tilde; // pre
    rho = norm(y);
    w_tilde = r;
    z = /*eiM **/ w_tilde; // pre
    xi = norm(z);
    gamma = 1;
    eta = -1;
    
    std::ofstream ofs("qmr_nopre_convergence.txt");
    
    while ( nr/nr0 > 1e-8 && iter < N*10 && nr/nr0 < 10000 )
    {
        std::cout << "                                                 \r";
        std::cout << " -> Iteration " << iter << ", rr = ";
        std::cout << nr/nr0 << "\b\r";
        std::cout.flush();
        
        ofs << nr/nr0 << std::endl;
        
        if (abs(rho) < 1e-15 or abs(xi) < 1e-15)
        {
            std::cout << "QMR failed (rho, xi)" << std::endl;
            return false;
        }
        
        v = v_tilde / rho;
        y = y / rho;
        w = w_tilde / xi;
        z = z / xi;
        delta = dot(z,y);
        if (abs(delta) < 1e-15)
        {
            std::cout << "QMR failed (delta)" << std::endl;
            return false;
        }
        y_tilde = /*eiM **/ y; // pre
        z_tilde = /*eiM **/ z; // pre
        
        if (iter == 0)
        {
            p = y_tilde;
            q = z_tilde;
        }
        else
        {
            p = y_tilde - (xi*delta/epsilon)*p;
            q = z_tilde - (rho*(delta/epsilon))*q;
        }
        
        p_tilde = A*p;
        epsilon = dot(q, p_tilde);
        if (abs(epsilon) < 1e-15)
        {
            std::cout << "QMR failed (epsilon)" << std::endl;
            return false;
        }
        beta = epsilon/delta;
        if (abs(beta) < 1e-15)
        {
            std::cout << "QMR failed (beta)" << std::endl;
            return false;
        }
        v_tilde = p_tilde - beta*v;
        y = /*eiM **/ v_tilde; // pre
        rho1 = rho;
        rho = norm(y);
        w_tilde = trans(A)*q - beta*w;
        z = /*eiM **/ w_tilde; // pre
        xi = norm(z);
        
        if (iter > 0)
            theta1 = theta;
        
        theta = rho/(gamma * std::abs(beta));
        gamma1 = gamma;
        
        gamma = 1.0/sqrt(1.0+theta*theta);
        if (abs(gamma) < 1e-15)
        {
            std::cout << "QMR failed (gamma)" << std::endl;
            return false;
        }
        eta = -eta*rho1*gamma*gamma/(beta*gamma1*gamma1);
        
        if (iter == 0)
        {
            d = eta*p;
            s = eta*p_tilde;
        }
        else
        {
            d = eta*p + (theta1*gamma)*(theta1*gamma)*d;
            s = eta*p_tilde + (theta1*gamma)*(theta1*gamma)*s;
        }
        
        x = x + d;
        r = r - s;
        
        nr = norm(r);
        iter++;
    }
    
    ofs << nr/nr0 << std::endl;
    ofs.close();
    
    std::cout << " -> Iteration " << iter << ", rr = " << nr/nr0 << std::endl;
    
    return true;
}

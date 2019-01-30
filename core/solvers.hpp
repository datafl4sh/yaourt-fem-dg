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
    std::string     history_filename;

    conjugated_gradient_params() : rr_tol(1e-8),
                                   rr_max(20),
                                   max_iter(100),
                                   verbose(false),
                                   save_iteration_history(false),
                                   use_initial_guess(false) {}
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
/*
 * Copyright (C) 2019, Matteo Cicuttin - datafl4sh@toxicnet.eu
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of Udine nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(s) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(s) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


/* Surprisingly blaze-lib does not offer triplet initialization for sparse
 * matrices. This is a quick and dirty implementation that allows it.
 * See the main() for the details, the code should be self-explaining.
 *
 * One day I will do a PR to the blaze-lib, however now I do not have much
 * time to make this code PR-ready. Sorry.
 */

#pragma once

#include <iostream>
#include <blaze/Math.h>

namespace blaze {

/* The triplet class with its basic operations */
template<typename T, typename Idx = size_t>
class triplet
{
    Idx     i, j;
    T       val;

public:
    triplet() : i(0), j(0), val(0)
    {}

    triplet(Idx p_i, Idx p_j, T p_val)
        : i(p_i), j(p_j), val(p_val)
    {}

    bool operator<(const triplet& other) const {
        return (i == other.i and j < other.j) or (i < other.i);
    }

    auto row() const { return i; }
    auto col() const { return j; }
    auto value() const { return val; }
    
    bool same_position_of(const triplet& other)
    {
        return (i == other.i) and (j == other.j);
    }
    
    triplet& operator+=(const triplet& other)
    {
        if ( !same_position_of(other) )
            throw std::invalid_argument("Added values in different positions");
            
        val += other.val;
        return (*this);
    }
    
    triplet operator+(const triplet& other)
    {
        triplet ret = (*this);
        ret += other;
        return ret;
    }
};

/* Streaming operator for the triplet class */
template<typename T, typename Idx>
std::ostream&
operator<<(std::ostream& os, const triplet<T, Idx>& t)
{
    os << "(" << t.row() << ", " << t.col() << ", " << t.value() << ")";
    return os;
}

/* Reduce triplets: given a *sorted* array of triplets (see the operator< in
 * the triplet class), take all the triplets in the same position and
 * accumulate them on a single triplet. */
template<typename ForwardIterator>
ForwardIterator
reduce_triplets(ForwardIterator first, ForwardIterator last)
{
    if ( first == last )
        return last;

    std::sort(first, last);

    ForwardIterator result = first;
    while ( ++first != last )
    {
        auto r = *result;
        auto f = *first;
        if ( r.same_position_of(f) )
            *result += f;
        else
            *(++result) = f;
    }
    
    return ++result;
}

/* Do the actual initialization using the API provided by blaze-lib. */
template<typename T, typename ForwardIterator>
void
init_from_triplets(CompressedMatrix<T>& M, ForwardIterator first,
                   ForwardIterator last)
{
    auto real_last = reduce_triplets(first, last);

    size_t elems = std::distance(first, real_last);

    M.reserve(elems);
        
    size_t cur_row = 0;
    while ( first != real_last )
    {
        auto f = *first;
        if ( cur_row < f.row() )
        {
            M.finalize(cur_row);
            ++cur_row;
        }
        
        if ( cur_row == f.row() )
        {
            M.append(f.row(), f.col(), f.value());
            ++first;
        }
    }
    
    for (size_t row = cur_row; row < M.rows(); row++)
        M.finalize(row);
}

/* Don't use BEGIN */
//getrf/getrs don't give good solution
template< typename T, bool SO, bool TF >
const DynamicVector<T,TF>
solve_LU(const DynamicMatrix<T,SO>& A, const DynamicVector<T,TF>& b)
{
    DynamicMatrix<T,SO> At = A;
    DynamicVector<T,TF> ret = b;

    auto size = A.rows();

    const std::unique_ptr<int[]> ipiv( new int[ size ] );
    getrf( At, ipiv.get() );
    getrs( At, ret, 'N', ipiv.get() );

    return ret;
}

template< typename T, bool SO >
const DynamicMatrix<T,SO>
solve_LU(const DynamicMatrix<T,SO>& A, const DynamicMatrix<T,SO>& B)
{
    DynamicMatrix<T,SO> At = A;
    DynamicMatrix<T,SO> ret = trans(B);

    auto size = A.rows();

    const std::unique_ptr<int[]> ipiv( new int[ size ] );
    getrf( At, ipiv.get() );
    getrs( At, ret, 'N', ipiv.get() );

    return trans(ret);
}

template< typename T, bool SO >
class LU {

    DynamicMatrix<T,SO>             At;
    const std::unique_ptr<int[]>    ipiv;

public:
    LU(const DynamicMatrix<T,SO>& A)
        : ipiv(new int[ A.rows() ])
    {
        At = A;
        getrf( At, ipiv.get() );
    }

    DynamicMatrix<T,SO>
    solve(const DynamicMatrix<T,SO>& rhs)
    {
        DynamicMatrix<T,SO> ret = rhs;
        getrs( At, ret, 'N', ipiv.get() );
        return ret;
    }
};

template< typename T, bool SO >
auto
make_LU(const DynamicMatrix<T,SO>& A)
{
    return LU<T,SO>(A);
}
/* Don't use END */

} // namespace blaze




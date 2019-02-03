
#pragma once

#include <iostream>
#include <array>

template<typename T, size_t DIM>
class point
{
    std::array<T, DIM>     m_coords;

public:
    typedef T                                   value_type;
    const static size_t                         dimension = DIM;

    point()
    {
        for (size_t i = 0; i < DIM; i++)
            m_coords[i] = T(0);
    }

    point(const point& other)
        : m_coords(other.m_coords)
    {}
    
    point operator=(const point& other)
    {
        m_coords = other.m_coords;
        return *this;
    }
    
    template<typename U = T>
    point(const typename std::enable_if<DIM == 1, U>::type& x)
    {
        m_coords[0] = x;
    }
    
    template<typename U = T>
    point(const typename std::enable_if<DIM == 2, U>::type& x, const U& y)
    {
        m_coords[0] = x;
        m_coords[1] = y;
    }
    
    template<typename U = T>
    point(const typename std::enable_if<DIM == 3, U>::type& x, const U& y, const U& z)
    {
        m_coords[0] = x;
        m_coords[1] = y;
        m_coords[2] = z;
    }

    T   at(size_t pos) const { return m_coords.at(pos); }
    T&  at(size_t pos)       { return m_coords.at(pos); }

    T   operator[](size_t pos) const { return m_coords[pos]; }
    T&  operator[](size_t pos)       { return m_coords[pos]; }

    point   operator-() const {
        auto ret = -1.0 * (*this);
        return ret;
    }

    template<typename U = T>
    typename std::enable_if<DIM == 1 || DIM == 2 || DIM == 3, U>::type
    x() const { return m_coords[0]; }

    template<typename U = T>
    typename std::enable_if<DIM == 1 || DIM == 2 || DIM == 3, U>::type&
    x() { return m_coords[0]; }

    template<typename U = T>
    typename std::enable_if<DIM == 2 || DIM == 3, U>::type
    y() const { return m_coords[1]; }

    template<typename U = T>
    typename std::enable_if<DIM == 2 || DIM == 3, U>::type&
    y() { return m_coords[1]; }

    template<typename U = T>
    typename std::enable_if<DIM == 3, U>::type
    z() const { return m_coords[2]; }

    template<typename U = T>
    typename std::enable_if<DIM == 3, U>::type&
    z() { return m_coords[2]; }

    point&
    operator+=(const point& other)
    {
        for (size_t i = 0; i < DIM; i++)
            m_coords[i] += other.m_coords[i];

        return *this;
    }

    point
    operator+(const point& other) const
    {
        point ret = *this;
        ret += other;
        return ret;
    }

    point&
    operator-=(const point& other)
    {
        for (size_t i = 0; i < DIM; i++)
            m_coords[i] -= other.m_coords[i];

        return *this;
    }

    point
    operator-(const point& other) const
    {
        point ret = *this;
        ret -= other;
        return ret;
    }

    point&
    operator*=(const T& scalefactor)
    {
        for (size_t i = 0; i < DIM; i++)
            m_coords[i] *= scalefactor;

        return *this;
    }

    point
    operator*(const T& scalefactor) const
    {
        point ret = *this;
        ret *= scalefactor;
        return ret;
    }

    friend point
    operator*(T scalefactor, const point& p)
    {
        return p * scalefactor;
    }

    point
    operator/=(const T& scalefactor)
    {
        point ret;
        for (size_t i = 0; i < DIM; i++)
            m_coords[i] /= scalefactor;
        
        return ret;
    }

    point
    operator/(const T& scalefactor) const
    {
        point ret = *this;
        ret /= scalefactor;
        return ret;
    }
};

template<typename T, size_t DIM>
T
distance(const point<T,DIM>& p1, const point<T,DIM>& p2)
{
    auto acc = 0.0;

    for (size_t i = 0; i < DIM; i++)
        acc += (p1[i] - p2[i])*(p1[i] - p2[i]);

    return std::sqrt(acc);
}

template<typename T>
T
det(const point<T,2>& p1, const point<T,2>& p2)
{
    return p1.x() * p2.y() - p1.y() * p2.x();
}

template<typename T, size_t DIM>
std::ostream&
operator<<(std::ostream& os, const point<T, DIM>& pt)
{
    os << "( ";
    for (size_t i = 0; i < DIM; i++)
    {
        os << pt[i];

        if (i < DIM-1)
            os << ", ";
    }
    os << " )";
    return os;
}


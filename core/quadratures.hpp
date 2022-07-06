/*
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
 *
 * Toy implememtation of Discontinuous Galerkin for teaching purposes
 *
 * Matteo Cicuttin (c) 2018
 */

/*
 *       /\        Matteo Cicuttin (C) 2016, 2017, 2018
 *      /__\       matteo.cicuttin@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    DISK++, a template library for DIscontinuous SKeletal
 *  /_\/_\/_\/_\   methods.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * If you use this code or parts of it for scientific publications, you
 * are required to cite it as following:
 *
 * Implementation of Discontinuous Skeletal methods on arbitrary-dimensional,
 * polytopal meshes using generic programming.
 * M. Cicuttin, D. A. Di Pietro, A. Ern.
 * Journal of Computational and Applied Mathematics.
 * DOI: 10.1016/j.cam.2017.09.017
 */

#pragma once

#include <vector>
#include <cmath>

#include "core/point.hpp"
#include "core/mesh.hpp"
#include "core/refelem.hpp"

#define USE_DUNAVANT

namespace yaourt {
namespace quadratures {


template<typename T, size_t DIM>
class quadrature_point
{
    point<T,DIM>    q_point;
    T               q_weight;

public:
    typedef point<T,DIM>    point_type;
    typedef T               weight_type;

    quadrature_point()
    {}

    quadrature_point(const point_type& qp, const weight_type& qw)
        : q_point(qp), q_weight(qw)
    {}

    auto point() const { return q_point; }
    auto weight() const { return q_weight; }
};

template<typename T, size_t DIM>
auto make_qp(const point<T, DIM>& qp, const T& qw)
{
    return quadrature_point<T, DIM>(qp, qw);
}


/* Gauss-Legendre 1D quadrature. */
template<typename T>
std::vector<std::pair<point<T, 1>, T>>
gauss_legendre(size_t degree)
{
    std::vector<std::pair<point<T, 1>, T>> ret;

    point<T, 1> qp;
    T           qw;
    T           a1, a2;

    switch (degree)
    {
        case 0:
        case 1:
            qp = point<T, 1>({0.0});
            qw = 2.0;
            ret.push_back(std::make_pair(qp, qw));
            return ret;

        case 2:
        case 3:
            qp = point<T, 1>({1.0 / std::sqrt(3.0)});
            qw = 1.0;
            ret.push_back(std::make_pair(-qp, qw));
            ret.push_back(std::make_pair(qp, qw));
            return ret;

        case 4:
        case 5:
            qp = point<T, 1>({std::sqrt(3.0 / 5.0)});
            qw = 5.0 / 9.0;
            ret.push_back(std::make_pair(-qp, qw));
            ret.push_back(std::make_pair(qp, qw));

            qp = point<T, 1>({0.0});
            qw = 8.0 / 9.0;
            ret.push_back(std::make_pair(qp, qw));

            return ret;

        case 6:
        case 7:
            a1 = 15;
            a2 = 2.0 * std::sqrt(30.0);

            qp = point<T, 1>({std::sqrt((a1 - a2) / 35.0)});
            qw = (18.0 + std::sqrt(30.0)) / 36.0;
            ret.push_back(std::make_pair(-qp, qw));
            ret.push_back(std::make_pair(qp, qw));

            qp = point<T, 1>({std::sqrt((a1 + a2) / 35.0)});
            qw = (18.0 - std::sqrt(30.0)) / 36.0;
            ret.push_back(std::make_pair(-qp, qw));
            ret.push_back(std::make_pair(qp, qw));

            return ret;

        case 8:
        case 9:
            qp = point<T, 1>({0.0});
            qw = 128.0 / 225.0;
            ret.push_back(std::make_pair(qp, qw));

            a1 = 5.0;
            a2 = 2.0 * std::sqrt(10.0 / 7.0);
            qp = point<T, 1>({std::sqrt(a1 - a2) / 3.0});
            qw = (322 + 13.0 * std::sqrt(70.0)) / 900.0;
            ret.push_back(std::make_pair(-qp, qw));
            ret.push_back(std::make_pair(qp, qw));

            qp = point<T, 1>({std::sqrt(a1 + a2) / 3.0});
            qw = (322 - 13.0 * std::sqrt(70.0)) / 900.0;
            ret.push_back(std::make_pair(-qp, qw));
            ret.push_back(std::make_pair(qp, qw));
            return ret;

        default:
            throw std::invalid_argument("Gauss quadrature: degree too high");
    }

    return ret;
}


namespace detail {

static double dunavant_rule_1[][4] = {
    { 0.333333333333333, 0.333333333333333, 0.333333333333333,  1.000000000000000 }
};

static double dunavant_rule_2[][4] = {
    { 0.666666666666667, 0.166666666666667, 0.166666666666667,  0.333333333333333 },
    { 0.166666666666667, 0.666666666666667, 0.166666666666667,  0.333333333333333 },
    { 0.166666666666667, 0.166666666666667, 0.666666666666667,  0.333333333333333 }
};

static double dunavant_rule_3[][4] = {
    { 0.333333333333333, 0.333333333333333, 0.333333333333333, -0.562500000000000 },
    { 0.600000000000000, 0.200000000000000, 0.200000000000000,  0.520833333333333 },
    { 0.200000000000000, 0.600000000000000, 0.200000000000000,  0.520833333333333 },
    { 0.200000000000000, 0.200000000000000, 0.600000000000000,  0.520833333333333 }
};

static double dunavant_rule_4[][4] = {
    { 0.108103018168070, 0.445948490915965, 0.445948490915965,  0.223381589678011 },
    { 0.445948490915965, 0.108103018168070, 0.445948490915965,  0.223381589678011 },
    { 0.445948490915965, 0.445948490915965, 0.108103018168070,  0.223381589678011 },
    { 0.816847572980459, 0.091576213509771, 0.091576213509771,  0.109951743655322 },
    { 0.091576213509771, 0.816847572980459, 0.091576213509771,  0.109951743655322 },
    { 0.091576213509771, 0.091576213509771, 0.816847572980459,  0.109951743655322 }
};

static double dunavant_rule_5[][4] = {
    { 0.333333333333333, 0.333333333333333, 0.333333333333333,  0.225000000000000 },
    { 0.059715871789770, 0.470142064105115, 0.470142064105115,  0.132394152788506 },
    { 0.470142064105115, 0.059715871789770, 0.470142064105115,  0.132394152788506 },
    { 0.470142064105115, 0.470142064105115, 0.059715871789770,  0.132394152788506 },
    { 0.797426985353087, 0.101286507323456, 0.101286507323456,  0.125939180544827 },
    { 0.101286507323456, 0.797426985353087, 0.101286507323456,  0.125939180544827 },
    { 0.101286507323456, 0.101286507323456, 0.797426985353087,  0.125939180544827 },
};

static double dunavant_rule_6[][4] = {
    { 0.501426509658179, 0.249286745170910, 0.249286745170910,  0.116786275726379 },
    { 0.249286745170910, 0.501426509658179, 0.249286745170910,  0.116786275726379 },
    { 0.249286745170910, 0.249286745170910, 0.501426509658179,  0.116786275726379 },
    { 0.873821971016996, 0.063089014491502, 0.063089014491502,  0.050844906370207 },
    { 0.063089014491502, 0.873821971016996, 0.063089014491502,  0.050844906370207 },
    { 0.063089014491502, 0.063089014491502, 0.873821971016996,  0.050844906370207 },
    { 0.053145049844817, 0.310352451033784, 0.636502499121399,  0.082851075618374 },
    { 0.053145049844817, 0.636502499121399, 0.310352451033784,  0.082851075618374 },
    { 0.310352451033784, 0.053145049844817, 0.636502499121399,  0.082851075618374 },
    { 0.310352451033784, 0.636502499121399, 0.053145049844817,  0.082851075618374 },
    { 0.636502499121399, 0.053145049844817, 0.310352451033784,  0.082851075618374 },
    { 0.636502499121399, 0.310352451033784, 0.053145049844817,  0.082851075618374 },
};

static double dunavant_rule_7[][4] = {
    { 0.333333333333333, 0.333333333333333, 0.333333333333333, -0.149570044467682 },
    { 0.479308067841920, 0.260345966079040, 0.260345966079040,  0.175615257433208 },
    { 0.260345966079040, 0.479308067841920, 0.260345966079040,  0.175615257433208 },
    { 0.260345966079040, 0.260345966079040, 0.479308067841920,  0.175615257433208 },
    { 0.869739794195568, 0.065130102902216, 0.065130102902216,  0.053347235608838 },
    { 0.065130102902216, 0.869739794195568, 0.065130102902216,  0.053347235608838 },
    { 0.065130102902216, 0.065130102902216, 0.869739794195568,  0.053347235608838 },
    { 0.048690315425316, 0.312865496004874, 0.638444188569810,  0.077113760890257 },
    { 0.048690315425316, 0.638444188569810, 0.312865496004874,  0.077113760890257 },
    { 0.312865496004874, 0.048690315425316, 0.638444188569810,  0.077113760890257 },
    { 0.312865496004874, 0.638444188569810, 0.048690315425316,  0.077113760890257 },
    { 0.638444188569810, 0.048690315425316, 0.312865496004874,  0.077113760890257 },
    { 0.638444188569810, 0.312865496004874, 0.048690315425316,  0.077113760890257 }
};

static double dunavant_rule_8[][4] = {
    { 0.333333333333333, 0.333333333333333, 0.333333333333333,  0.144315607677787 },
    { 0.081414823414554, 0.459292588292723, 0.459292588292723,  0.095091634267285 },
    { 0.459292588292723, 0.081414823414554, 0.459292588292723,  0.095091634267285 },
    { 0.459292588292723, 0.459292588292723, 0.081414823414554,  0.095091634267285 },
    { 0.658861384496480, 0.170569307751760, 0.170569307751760,  0.103217370534718 },
    { 0.170569307751760, 0.658861384496480, 0.170569307751760,  0.103217370534718 },
    { 0.170569307751760, 0.170569307751760, 0.658861384496480,  0.103217370534718 },
    { 0.898905543365938, 0.050547228317031, 0.050547228317031,  0.032458497623198 },
    { 0.050547228317031, 0.898905543365938, 0.050547228317031,  0.032458497623198 },
    { 0.050547228317031, 0.050547228317031, 0.898905543365938,  0.032458497623198 },
    { 0.008394777409958, 0.263112829634638, 0.728492392955404,  0.027230314174435 },
    { 0.008394777409958, 0.728492392955404, 0.263112829634638,  0.027230314174435 },
    { 0.263112829634638, 0.008394777409958, 0.728492392955404,  0.027230314174435 },
    { 0.263112829634638, 0.728492392955404, 0.008394777409958,  0.027230314174435 },
    { 0.728492392955404, 0.008394777409958, 0.263112829634638,  0.027230314174435 },
    { 0.728492392955404, 0.263112829634638, 0.008394777409958,  0.027230314174435 }
};

static double dunavant_rule_9[][4] = {
    { 0.333333333333333, 0.333333333333333, 0.333333333333333,  0.097135796282799 },

    { 0.020634961602525, 0.489682519198738, 0.489682519198738,  0.031334700227139 },
    { 0.489682519198738, 0.020634961602525, 0.489682519198738,  0.031334700227139 },
    { 0.489682519198738, 0.489682519198738, 0.020634961602525,  0.031334700227139 },
    
    { 0.125820817014127, 0.437089591492937, 0.437089591492937,  0.077827541004774 },
    { 0.437089591492937, 0.125820817014127, 0.437089591492937,  0.077827541004774 },
    { 0.437089591492937, 0.437089591492937, 0.125820817014127,  0.077827541004774 },
    
    { 0.623592928761935, 0.188203535619033, 0.188203535619033,  0.079647738927210 },
    { 0.188203535619033, 0.623592928761935, 0.188203535619033,  0.079647738927210 },
    { 0.188203535619033, 0.188203535619033, 0.623592928761935,  0.079647738927210 },
    
    { 0.910540973211095, 0.044729513394453, 0.044729513394453,  0.025577675658698 },
    { 0.044729513394453, 0.910540973211095, 0.044729513394453,  0.025577675658698 },
    { 0.044729513394453, 0.044729513394453, 0.910540973211095,  0.025577675658698 },

    { 0.036838412054736, 0.221962989160766, 0.741198598784498,  0.043283539377289 },
    { 0.036838412054736, 0.221962989160766, 0.741198598784498,  0.043283539377289 },
    { 0.036838412054736, 0.221962989160766, 0.741198598784498,  0.043283539377289 },
    { 0.036838412054736, 0.221962989160766, 0.741198598784498,  0.043283539377289 },
    { 0.036838412054736, 0.221962989160766, 0.741198598784498,  0.043283539377289 },
    { 0.036838412054736, 0.221962989160766, 0.741198598784498,  0.043283539377289 },
};

struct dunavant_rule {
    size_t num_points;
    double (*data)[4];
};

static struct dunavant_rule dunavant_rules[] = {
    { sizeof(dunavant_rule_1)/(sizeof(double)*4), dunavant_rule_1 },
    { sizeof(dunavant_rule_2)/(sizeof(double)*4), dunavant_rule_2 },
    { sizeof(dunavant_rule_3)/(sizeof(double)*4), dunavant_rule_3 },
    { sizeof(dunavant_rule_4)/(sizeof(double)*4), dunavant_rule_4 },
    { sizeof(dunavant_rule_5)/(sizeof(double)*4), dunavant_rule_5 },
    { sizeof(dunavant_rule_6)/(sizeof(double)*4), dunavant_rule_6 },
    { sizeof(dunavant_rule_7)/(sizeof(double)*4), dunavant_rule_7 },
    { sizeof(dunavant_rule_8)/(sizeof(double)*4), dunavant_rule_8 },
    { 0, NULL }
};

void print_rule_details(void)
{
    for (size_t i = 0; i < 7; i++)
    {
        std::cout << "Rule " << i+1 << std::endl;
        auto num_points = dunavant_rules[i].num_points;

        double tot_w = 0;

        for (size_t j = 0; j < num_points; j++)
        {
            auto l0 = dunavant_rules[i].data[j][0];
            auto l1 = dunavant_rules[i].data[j][1];
            auto l2 = dunavant_rules[i].data[j][2];
            auto w = dunavant_rules[i].data[j][3];
            tot_w += w;
            std::cout << "Coord sum: " << l0 + l1 + l2 << std::endl;
        }
        std::cout << "Tot weight: " << tot_w << std::endl;
    }
}

/* See Ern & Guermond - Theory and practice of FEM, pag 360. */
template<typename T>
std::vector<quadrature_point<T,2>>
triangle_quadrature_low_order(const point<T,2>& p0,
                              const point<T,2>& p1, 
                              const point<T,2>& p2, size_t deg)
{
    std::vector<quadrature_point<T,2>>   ret;
    auto v0 = p1 - p0;
    auto v1 = p2 - p0;
    auto area = std::abs( (v0.x() * v1.y() - v0.y() * v1.x())/2.0 );
    point<T,2>      qp;
    T               qw;
    T               a1 = (6. - std::sqrt(15.)) / 21;
    T               a2 = (6. + std::sqrt(15.)) / 21;
    T               w1 = (155. - std::sqrt(15.)) / 1200;
    T               w2 = (155. + std::sqrt(15.)) / 1200;
    switch(deg)
    {
        case 0:
        case 1:
            qw = area;
            qp = (p0 + p1 + p2)/3;      ret.push_back( make_qp(qp, qw) );
            return ret;
        case 2:
            qw = area/3;
            qp = p0/6 + p1/6 + 2*p2/3;  ret.push_back( make_qp(qp, qw) );
            qp = p0/6 + 2*p1/3 + p2/6;  ret.push_back( make_qp(qp, qw) );
            qp = 2*p0/3 + p1/6 + p2/6;  ret.push_back( make_qp(qp, qw) );
            return ret;
        case 3:
            qw = 9*area/20;
            qp = (p0 + p1 + p2)/3;      ret.push_back( make_qp(qp, qw) );
            qw = 2*area/15;
            qp = (p0 + p1)/2;           ret.push_back( make_qp(qp, qw) );
            qp = (p0 + p2)/2;           ret.push_back( make_qp(qp, qw) );
            qp = (p1 + p2)/2;           ret.push_back( make_qp(qp, qw) );
            qw = area/20;
            qp = p0;                    ret.push_back( make_qp(qp, qw) );
            qp = p1;                    ret.push_back( make_qp(qp, qw) );
            qp = p2;                    ret.push_back( make_qp(qp, qw) );
            return ret;
        case 4:
        case 5:
            qw = 9*area/40;
            qp = (p0 + p1 + p2)/3;      ret.push_back( make_qp(qp, qw) );
            qw = w1 * area;
            qp = a1*p0 + a1*p1 + (1-2*a1)*p2;   ret.push_back( make_qp(qp, qw) );
            qp = a1*p0 + (1-2*a1)*p1 + a1*p2;   ret.push_back( make_qp(qp, qw) );
            qp = (1-2*a1)*p0 + a1*p1 + a1*p2;   ret.push_back( make_qp(qp, qw) );
            qw = w2 * area;
            qp = a2*p0 + a2*p1 + (1-2*a2)*p2;   ret.push_back( make_qp(qp, qw) );
            qp = a2*p0 + (1-2*a2)*p1 + a2*p2;   ret.push_back( make_qp(qp, qw) );
            qp = (1-2*a2)*p0 + a2*p1 + a2*p2;   ret.push_back( make_qp(qp, qw) );
            return ret;
            
        default:
            throw std::invalid_argument("Triangle quadrature: requested order too high");
    }
    return ret;
}

template<typename T>
std::vector<quadrature_point<T,2>>
triangle_quadrature_dunavant(const point<T,2>& p0,
                             const point<T,2>& p1, 
                             const point<T,2>& p2,
                             size_t degree)
{
    if (degree > 8)
        throw std::invalid_argument("Dunavant quadrature: degree too high");

    size_t rule_num = (degree == 0) ? 0 : degree - 1;
    assert(rule_num < 8);

    auto v0 = p1 - p0;
    auto v1 = p2 - p0;
    auto area = std::abs( (v0.x() * v1.y() - v0.y() * v1.x())/2.0 );

    auto num_points = detail::dunavant_rules[rule_num].num_points;

    std::vector<quadrature_point<T,2>> ret;
    ret.reserve(num_points);

    for (size_t i = 0; i < num_points; i++)
    {
        auto l0 = detail::dunavant_rules[rule_num].data[i][0];
        auto l1 = detail::dunavant_rules[rule_num].data[i][1];
        auto l2 = detail::dunavant_rules[rule_num].data[i][2];
        auto w = detail::dunavant_rules[rule_num].data[i][3];

        auto qp = p0*l0 + p1*l1 + p2*l2;
        auto qw = w*area;

        ret.push_back({qp,qw});
    }

    return ret;
}

} // namespace detail


template<typename Mesh>
std::vector<quadrature_point<typename Mesh::coordinate_type,2>>
integrate(const Mesh& msh,
          const typename Mesh::face_type& fc,
          size_t degree)
{
    using T = typename Mesh::coordinate_type;
    std::vector<quadrature_point<T,2>> ret;
    auto raw_qps = gauss_legendre<T>(degree);
    auto meas = measure(msh, fc);
    auto pts  = points(msh, fc);

    for (auto itor = raw_qps.begin(); itor != raw_qps.end(); itor++)
    {
        auto raw_qp = *itor;
        auto t  = raw_qp.first.x();
        auto qp  = 0.5 * (1 - t) * pts[0] + 0.5 * (1 + t) * pts[1];
        auto qw  = raw_qp.second * meas * 0.5;

        ret.push_back({qp,qw});
    }

    return ret;
}

template<typename T>
std::vector<quadrature_point<T,2>>
integrate(const simplicial_mesh<T>& msh,
          const typename simplicial_mesh<T>::cell_type& cl,
          size_t degree)
{
#ifdef USE_DUNAVANT
    auto pts = points(msh, cl);
    return detail::triangle_quadrature_dunavant(pts[0], pts[1], pts[2], degree);
#else /* USE_DUNAVANT */
    auto pts = points(msh, cl);
    return detail::triangle_quadrature_low_order(pts[0], pts[1], pts[2], degree);
#endif /* USE_DUNAVANT */
}

template<typename T>
std::vector<quadrature_point<T,2>>
integrate(const refelem::reference_triangle<T>& t,
          size_t degree)
{
    return detail::triangle_quadrature_dunavant(t.points[0],
                                                t.points[1],
                                                t.points[2],
                                                degree);
}

/* Quadrature for cartesian quadrangles, it is just tensorized Gauss points. */
template<typename T>
std::vector<std::pair<point<T, 2>, T>>
quadrangle_quadrature(const size_t degree)
{
    auto qps = gauss_legendre<T>(degree);

    std::vector<std::pair<point<T, 2>, T>> ret;
    ret.reserve(qps.size() * qps.size());

    for (auto jtor = qps.begin(); jtor != qps.end(); jtor++)
    {
        auto qp_y = *jtor;
        auto eta  = qp_y.first.x();

        for (auto itor = qps.begin(); itor != qps.end(); itor++)
        {
            auto qp_x = *itor;
            auto xi   = qp_x.first.x();

            auto qw2d = qp_x.second * qp_y.second;
            auto qp2d = point<T, 2>({xi, eta});

            ret.push_back({qp2d, qw2d});
        }
    }

    return ret;
}

template<typename T>
std::vector<quadrature_point<T, 2>>
integrate(const quad_mesh<T>& msh,
          const typename quad_mesh<T>::cell_type& cl,
          size_t degree)
{
    auto raw_qps = quadrangle_quadrature<T>(degree);

    std::vector<quadrature_point<T, 2>> ret;
    ret.reserve(raw_qps.size());

    auto pts = points(msh, cl);

    auto P = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].x() * (1 - xi) * (1 - eta) +
               0.25 * pts[1].x() * (1 + xi) * (1 - eta) +
               0.25 * pts[2].x() * (1 + xi) * (1 + eta) +
               0.25 * pts[3].x() * (1 - xi) * (1 + eta);
    };

    auto Q = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].y() * (1 - xi) * (1 - eta) +
               0.25 * pts[1].y() * (1 + xi) * (1 - eta) +
               0.25 * pts[2].y() * (1 + xi) * (1 + eta) +
               0.25 * pts[3].y() * (1 - xi) * (1 + eta);
    };

    auto J = [&](T xi, T eta) -> T {
        auto j11 = 0.25 * ( (pts[1].x() - pts[0].x()) * (1 - eta) +
                            (pts[2].x() - pts[3].x()) * (1 + eta) );

        auto j12 = 0.25 * ( (pts[1].y() - pts[0].y()) * (1 - eta) +
                            (pts[2].y() - pts[3].y()) * (1 + eta) );

        auto j21 = 0.25 * ( (pts[3].x() - pts[0].x()) * (1 - xi) +
                            (pts[2].x() - pts[1].x()) * (1 + xi) );

        auto j22 = 0.25 * ( (pts[3].y() - pts[0].y()) * (1 - xi) +
                            (pts[2].y() - pts[1].y()) * (1 + xi) );

        return std::abs(j11 * j22 - j12 * j21);
    };

    for (auto& raw_qp : raw_qps)
    {
        auto xi  = raw_qp.first.x();
        auto eta = raw_qp.first.y();

        auto px = P(xi, eta);
        auto py = Q(xi, eta);

        auto qw = raw_qp.second * J(xi, eta);
        auto qp = point<T, 2>({px, py});
        ret.push_back({qp, qw});
    }

    return ret;
}

} // namespace quadratures
} // namespace yaourt

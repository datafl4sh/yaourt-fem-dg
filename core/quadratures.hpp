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

namespace dg2d {
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

} // namespace detail


template<typename T>
std::vector<quadrature_point<T,2>>
integrate(const simplicial_mesh<T>& msh,
          const typename simplicial_mesh<T>::face_type& fc,
          size_t degree)
{
    std::vector<quadrature_point<T,2>> ret;
    auto raw_qps = gauss_legendre<T>(degree);
    auto meas = measure(msh, fc);

    return ret;
}


template<typename T>
std::vector<quadrature_point<T,2>>
integrate(const simplicial_mesh<T>& msh,
          const typename simplicial_mesh<T>::cell_type& cl,
          size_t degree)
{
    if (degree > 16)
        throw std::invalid_argument("Dunavant quadrature: degree too high");

    size_t rule_num = degree/2;

    auto pts = points(msh, cl);
    auto meas = measure(msh, cl);

    auto num_points = detail::dunavant_rules[rule_num].num_points;

    std::vector<quadrature_point<T,2>> ret;
    ret.reserve(num_points);

    for (size_t i = 0; i < num_points; i++)
    {
        auto l0 = detail::dunavant_rules[rule_num].data[i][0];
        auto l1 = detail::dunavant_rules[rule_num].data[i][1];
        auto l2 = detail::dunavant_rules[rule_num].data[i][2];
        auto w = detail::dunavant_rules[rule_num].data[i][3];

        auto qp = pts[0]*l0 + pts[1]*l1 + pts[2]*l2;
        auto qw = w*meas;

        ret.push_back({qp,qw});  
    }

    return ret;
}

} // namespace quadratures
} // namespace dg2d
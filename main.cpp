#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor-blas/xlinalg.hpp"

struct test
{
    bool one;
    bool two;
    bool tree;
    test() : one(false), two(false) {}
};

using Vector3 = xt::xtensor_fixed<double, xt::xshape<3>>;
using Matrix3 = xt::xtensor_fixed<double, xt::xshape<3,3>>;

Matrix3 skew(const Vector3& angles) {
	return {{0, -angles[2], angles[1]}, {angles[2], 0, -angles[0]}, {-angles[1], angles[0], 0}};
}


int main(int argc, char* argv[])
{
    Vector3 phi = {.5, 2, 7};
    Matrix3 eye3 = {{1, 0, 0},{0, 1, 0},{0, 0, 1}};
	auto phi_mag   = xt::norm_l2(phi * 2);
	auto phi_mag2  = phi_mag * phi_mag;
	auto phi_mag4  = phi_mag2 * phi_mag2;
	auto term1     = 1 - phi_mag2 / 6 + phi_mag4 / 120;
	auto term2     = 0.5 - phi_mag2 / 24 + phi_mag4 / 720;
	auto phi_cross = skew(phi);
	// auto result =  eye3 + term1 * phi_cross + term2 * xt::lin stk::mmul(phi_cross, phi_cross);
	auto result =  eye3 + term1 * phi_cross + term2 * xt::linalg::dot(phi_cross,phi_cross);

    double value = result[0];
    std::cout << value <<std::endl;

    return 0;
}
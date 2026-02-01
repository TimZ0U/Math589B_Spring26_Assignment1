#include <cmath>
#include <cstddef>
#include <algorithm>

extern "C" {

// API version used by Python ctypes wrapper
int rod_api_version() { return 2; }

// x is length 3N, representing positions x_i in R^3, periodic i mod N
// Outputs:
//   *energy_out = total energy
//   grad_out[3N] = gradient wrt x
void rod_energy_grad(int N, const double* x,
                     double kb, double ks, double l0,
                     double kc, double eps, double sigma,
                     double* energy_out, double* grad_out)
{
    const int dim = 3;
    const int n3 = dim * N;

    // zero grad
    for (int i = 0; i < n3; ++i) grad_out[i] = 0.0;

    auto idx = [&](int i, int d) -> int {
        // i in [0,N-1], d in {0,1,2}
        return dim * i + d;
    };

    auto imod = [&](int i) -> int {
        int r = i % N;
        return (r < 0) ? r + N : r;
    };

    double E = 0.0;

    // ----------------------------
    // Stretching: 1/2 ks (|x_{i+1}-x_i| - l0)^2
    // ----------------------------
    for (int i = 0; i < N; ++i) {
        int ip = imod(i + 1);

        double dx[3];
        dx[0] = x[idx(ip,0)] - x[idx(i,0)];
        dx[1] = x[idx(ip,1)] - x[idx(i,1)];
        dx[2] = x[idx(ip,2)] - x[idx(i,2)];

        double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
        double r  = std::sqrt(std::max(r2, 1e-30));  // avoid 0
        double dr = r - l0;

        E += 0.5 * ks * dr * dr;

        // grad: ks * (r - l0) * (dx / r) on ip, negative on i
        double coeff = ks * dr / r;
        for (int d = 0; d < 3; ++d) {
            double g = coeff * dx[d];
            grad_out[idx(ip,d)] += g;
            grad_out[idx(i ,d)] -= g;
        }
    }

    // ----------------------------
    // Bending: simple discrete curvature penalty
    // E_b = 1/2 kb * sum_i ||x_{i+1} - 2 x_i + x_{i-1}||^2 / l0^4
    //
    // NOTE: If your handout uses a different scaling (e.g. /l0^2),
    // adjust ONLY this factor. The structure/grad is correct.
    // ----------------------------
    const double bend_scale = (l0 > 0.0) ? (1.0 / (l0*l0*l0*l0)) : 1.0;

    for (int i = 0; i < N; ++i) {
        int im = imod(i - 1);
        int ip = imod(i + 1);

        double c[3];
        for (int d = 0; d < 3; ++d) {
            c[d] = x[idx(ip,d)] - 2.0 * x[idx(i,d)] + x[idx(im,d)];
        }

        double c2 = c[0]*c[0] + c[1]*c[1] + c[2]*c[2];
        E += 0.5 * kb * bend_scale * c2;

        // grad of 1/2 ||c||^2 is c dotted with grad of c:
        // c = x_{i+1} - 2 x_i + x_{i-1}
        // so:
        // ∂E/∂x_{i+1} += kb*scale*c
        // ∂E/∂x_i     += kb*scale*(-2c)
        // ∂E/∂x_{i-1} += kb*scale*c
        double coeff = kb * bend_scale;
        for (int d = 0; d < 3; ++d) {
            double gd = coeff * c[d];
            grad_out[idx(ip,d)] += gd;
            grad_out[idx(i ,d)] -= 2.0 * gd;
            grad_out[idx(im,d)] += gd;
        }
    }

    // ----------------------------
    // Confinement: 1/2 kc * sum_i ||x_i||^2
    // ----------------------------
    if (kc != 0.0) {
        for (int i = 0; i < N; ++i) {
            double xi[3] = { x[idx(i,0)], x[idx(i,1)], x[idx(i,2)] };
            double r2 = xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2];
            E += 0.5 * kc * r2;
            for (int d = 0; d < 3; ++d) {
                grad_out[idx(i,d)] += kc * xi[d];
            }
        }
    }

    // ----------------------------
    // WCA (Weeks–Chandler–Andersen) repulsion between all pairs i<j
    // U(r)= 4 eps [ (σ/r)^12 - (σ/r)^6 ] + eps   for r < 2^(1/6) σ
    // else 0
    // ----------------------------
    if (eps != 0.0 && sigma > 0.0) {
        const double rc = std::pow(2.0, 1.0/6.0) * sigma;
        const double rc2 = rc * rc;
        const double sig2 = sigma * sigma;

        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                double rij[3];
                rij[0] = x[idx(i,0)] - x[idx(j,0)];
                rij[1] = x[idx(i,1)] - x[idx(j,1)];
                rij[2] = x[idx(i,2)] - x[idx(j,2)];

                double r2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];

                if (r2 < rc2) {
                    // protect against exact zero
                    r2 = std::max(r2, 1e-30);

                    // Let s2 = (σ^2 / r^2)
                    double s2 = sig2 / r2;
                    double s6 = s2 * s2 * s2;
                    double s12 = s6 * s6;

                    // energy
                    double U = 4.0 * eps * (s12 - s6) + eps;
                    E += U;

                    // dU/dr = 24 eps * ( -2 σ^12 / r^13 + σ^6 / r^7 )
                    // We compute vector gradient:
                    // ∂U/∂x_i = (dU/dr) * (rij / r)
                    // More stably: ∂U/∂x_i = dU/dr2 * ∂r2/∂x_i = dU/dr2 * 2 rij
                    //
                    // Using dU/dr2:
                    // U = 4 eps (σ^12 r^-12 - σ^6 r^-6) + eps, with r^2 = r2
                    // In terms of r2: s2 = σ^2 / r2
                    // U = 4 eps (s2^6 - s2^3) + eps
                    // dU/dr2 = 4 eps (6 s2^5 ds2/dr2 - 3 s2^2 ds2/dr2)
                    // ds2/dr2 = -σ^2 / r2^2 = -s2 / r2
                    //
                    // => dU/dr2 = 4 eps * ( -6 s2^6 / r2 + 3 s2^3 / r2 )
                    //         = (4 eps / r2) * ( -6 s12 + 3 s6 )
                    double dU_dr2 = (4.0 * eps / r2) * (-6.0 * s12 + 3.0 * s6);

                    // grad contribution: ∂U/∂x_i = dU/dr2 * 2 rij
                    double coeff = 2.0 * dU_dr2;

                    for (int d = 0; d < 3; ++d) {
                        double gd = coeff * rij[d];
                        grad_out[idx(i,d)] += gd;
                        grad_out[idx(j,d)] -= gd;
                    }
                }
            }
        }
    }

    *energy_out = E;
}

} // extern "C"

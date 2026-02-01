#include <cmath>
#include <algorithm>

extern "C" {

// Bump when you change the exported function signatures.
int rod_api_version() { return 2; }

// Exported API (students may extend, but autograder only requires Python-level RodEnergy.value_and_grad).
// x: length 3N (xyzxyz...)
// grad_out: length 3N
// Periodic indexing enforces a closed loop.
void rod_energy_grad(
    int N,
    const double* x,
    double kb,
    double ks,
    double l0,
    double kc,     // confinement strength
    double eps,    // WCA epsilon
    double sigma,  // WCA sigma
    double* energy_out,
    double* grad_out
) {
    const int M = 3*N;
    for (int i = 0; i < M; ++i) grad_out[i] = 0.0;
    double E = 0.0;

    auto idx = [N](int i) {
        int r = i % N;
        return (r < 0) ? (r + N) : r;
    };
    auto get = [&](int i, int d) -> double {
        return x[3*idx(i) + d];
    };
    auto addg = [&](int i, int d, double v) {
        grad_out[3*idx(i) + d] += v;
    };

    // ---- Bending: kb * ||x_{i+1} - 2 x_i + x_{i-1}||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double b = get(i+1,d) - 2.0*get(i,d) + get(i-1,d);
            E += kb * b * b;
            const double c = 2.0 * kb * b;
            addg(i-1, d, c);
            addg(i,   d, -2.0*c);
            addg(i+1, d, c);
        }
    }

    // ---- Stretching: ks * (||x_{i+1}-x_i|| - l0)^2
    for (int i = 0; i < N; ++i) {
        double dx0 = get(i+1,0) - get(i,0);
        double dx1 = get(i+1,1) - get(i,1);
        double dx2 = get(i+1,2) - get(i,2);
        double r = std::sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2);
        r = std::max(r, 1e-12);
        double diff = r - l0;
        E += ks * diff * diff;

        double coeff = 2.0 * ks * diff / r;
        addg(i+1,0,  coeff * dx0);
        addg(i+1,1,  coeff * dx1);
        addg(i+1,2,  coeff * dx2);
        addg(i,0,   -coeff * dx0);
        addg(i,1,   -coeff * dx1);
        addg(i,2,   -coeff * dx2);
    }

    // ---- Confinement: kc * sum ||x_i||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            double xi = get(i,d);
            E += kc * xi * xi;
            addg(i,d, 2.0 * kc * xi);
        }
    }

    // ---- TODO: Segment–segment WCA self-avoidance ----
    //
    // For each non-adjacent segment pair (i,i+1) and (j,j+1):
    //  1) Compute closest points parameters u*, v* in [0,1]
    //  2) Compute r = p_i(u*) - p_j(v*),  d = ||r||
    //  3) If d < 2^(1/6)*sigma:
    //       U(d) = 4 eps [ (sigma/d)^12 - (sigma/d)^6 ] + eps
    //       Accumulate E += U(d)
    //       Accumulate gradient to endpoints x_i, x_{i+1}, x_j, x_{j+1}
    //
    // Exclusions: skip adjacent segments (including wrap neighbors).
    //
    // IMPORTANT: You must include the dependence of (u*, v*) on endpoints in your gradient.
    const double wca_cutoff = std::pow(2.0, 1.0 / 6.0) * sigma;
    auto clamp = [](double v, double lo, double hi) {
        return std::max(lo, std::min(hi, v));
    };

    for (int i = 0; i < N; ++i) {
        int i1 = idx(i + 1);
        for (int j = i + 1; j < N; ++j) {
            int j1 = idx(j + 1);
            if (i == j || i1 == j || j1 == i) {
                continue;
            }

            double p0[3] = {get(i,0), get(i,1), get(i,2)};
            double p1[3] = {get(i1,0), get(i1,1), get(i1,2)};
            double q0[3] = {get(j,0), get(j,1), get(j,2)};
            double q1[3] = {get(j1,0), get(j1,1), get(j1,2)};

            double d1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            double d2[3] = {q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]};
            double w0[3] = {p0[0] - q0[0], p0[1] - q0[1], p0[2] - q0[2]};

            double a = d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2];
            double b = d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2];
            double c = d2[0]*d2[0] + d2[1]*d2[1] + d2[2]*d2[2];
            double d = d1[0]*w0[0] + d1[1]*w0[1] + d1[2]*w0[2];
            double e = d2[0]*w0[0] + d2[1]*w0[1] + d2[2]*w0[2];

            double u = 0.0;
            double v = 0.0;
            double denom = a * c - b * b;
            if (denom > 1e-12) {
                u = clamp((b * e - c * d) / denom, 0.0, 1.0);
            }
            if (c > 1e-12) {
                v = (b * u + e) / c;
            }
            if (v < 0.0) {
                v = 0.0;
                if (a > 1e-12) {
                    u = clamp(-d / a, 0.0, 1.0);
                }
            } else if (v > 1.0) {
                v = 1.0;
                if (a > 1e-12) {
                    u = clamp((b - d) / a, 0.0, 1.0);
                }
            }

            double rvec[3] = {
                (1.0 - u) * p0[0] + u * p1[0] - (1.0 - v) * q0[0] - v * q1[0],
                (1.0 - u) * p0[1] + u * p1[1] - (1.0 - v) * q0[1] - v * q1[1],
                (1.0 - u) * p0[2] + u * p1[2] - (1.0 - v) * q0[2] - v * q1[2]
            };
            double dist2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];
            double dist = std::sqrt(std::max(dist2, 1e-24));
            if (dist >= wca_cutoff) {
                continue;
            }

            double inv = sigma / dist;
            double inv2 = inv * inv;
            double inv6 = inv2 * inv2 * inv2;
            double inv12 = inv6 * inv6;
            double U = 4.0 * eps * (inv12 - inv6) + eps;
            E += U;

            double coeff = 24.0 * eps * (inv6 - 2.0 * inv12) / (dist * dist);
            for (int dch = 0; dch < 3; ++dch) {
                double g = coeff * rvec[dch];
                addg(i, dch, (1.0 - u) * g);
                addg(i1, dch, u * g);
                addg(j, dch, -(1.0 - v) * g);
                addg(j1, dch, -v * g);
            }
        }
    }

    *energy_out = E;
}

} // extern "C"

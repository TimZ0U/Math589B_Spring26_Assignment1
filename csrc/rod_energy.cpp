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
    const double clamp_eps = 1e-12;

    for (int i = 0; i < N; ++i) {
        int i1 = idx(i + 1);
        for (int j = i + 1; j < N; ++j) {
            int j1 = idx(j + 1);
            int diff = j - i;
            if (diff <= 2 || diff >= N - 2) {
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
            bool u_from_line = false;
            bool u_clamped = false;
            bool v_clamped = false;
            if (denom > 1e-12) {
                double u_raw = (b * e - c * d) / denom;
                u_from_line = true;
                u = clamp(u_raw, 0.0, 1.0);
                u_clamped = (u != u_raw);
            }
            if (c > 1e-12) {
                v = (b * u + e) / c;
            }
            if (v < 0.0) {
                v = 0.0;
                v_clamped = true;
                if (a > 1e-12) {
                    double u_raw = -d / a;
                    u = clamp(u_raw, 0.0, 1.0);
                    u_clamped = (u != u_raw);
                }
            } else if (v > 1.0) {
                v = 1.0;
                v_clamped = true;
                if (a > 1e-12) {
                    double u_raw = (b - d) / a;
                    u = clamp(u_raw, 0.0, 1.0);
                    u_clamped = (u != u_raw);
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
            double gvec[3] = {coeff * rvec[0], coeff * rvec[1], coeff * rvec[2]};

            double du_dp0[3] = {0.0, 0.0, 0.0};
            double du_dp1[3] = {0.0, 0.0, 0.0};
            double du_dq0[3] = {0.0, 0.0, 0.0};
            double du_dq1[3] = {0.0, 0.0, 0.0};
            double dv_dp0[3] = {0.0, 0.0, 0.0};
            double dv_dp1[3] = {0.0, 0.0, 0.0};
            double dv_dq0[3] = {0.0, 0.0, 0.0};
            double dv_dq1[3] = {0.0, 0.0, 0.0};

            bool u_on_boundary = (u <= clamp_eps || u >= 1.0 - clamp_eps);
            bool v_on_boundary = (v <= clamp_eps || v >= 1.0 - clamp_eps);
            bool use_line_line = (!v_clamped && !u_clamped && u_from_line && !u_on_boundary && !v_on_boundary && denom > 1e-12);
            bool use_u_fixed = (!v_clamped && (u_clamped || !u_from_line || u_on_boundary));
            bool use_v_fixed = v_clamped;

            auto fill_du_dv = [&](const double da[3], const double db[3], const double dc[3],
                                  const double ddv[3], const double dev[3],
                                  double du_out[3], double dv_out[3]) {
                for (int k = 0; k < 3; ++k) {
                    double rhs1 = ddv[k] - da[k] * u + db[k] * v;
                    double rhs2 = dev[k] + db[k] * u - dc[k] * v;
                    du_out[k] = (c * rhs1 + b * rhs2) / denom;
                    dv_out[k] = (b * rhs1 + a * rhs2) / denom;
                }
            };

            double da_dp0[3] = {-2.0 * d1[0], -2.0 * d1[1], -2.0 * d1[2]};
            double da_dp1[3] = { 2.0 * d1[0],  2.0 * d1[1],  2.0 * d1[2]};
            double db_dp0[3] = {-d2[0], -d2[1], -d2[2]};
            double db_dp1[3] = { d2[0],  d2[1],  d2[2]};
            double db_dq0[3] = {-d1[0], -d1[1], -d1[2]};
            double db_dq1[3] = { d1[0],  d1[1],  d1[2]};
            double dc_dq0[3] = {-2.0 * d2[0], -2.0 * d2[1], -2.0 * d2[2]};
            double dc_dq1[3] = { 2.0 * d2[0],  2.0 * d2[1],  2.0 * d2[2]};
            double dd_dp0[3] = {d1[0] - w0[0], d1[1] - w0[1], d1[2] - w0[2]};
            double dd_dp1[3] = {w0[0], w0[1], w0[2]};
            double dd_dq0[3] = {-d1[0], -d1[1], -d1[2]};
            double de_dp0[3] = {d2[0], d2[1], d2[2]};
            double de_dq0[3] = {-(w0[0] + d2[0]), -(w0[1] + d2[1]), -(w0[2] + d2[2])};
            double de_dq1[3] = {w0[0], w0[1], w0[2]};

            if (use_line_line) {
                double da_zero[3] = {0.0, 0.0, 0.0};
                double dc_zero[3] = {0.0, 0.0, 0.0};
                double dd_zero[3] = {0.0, 0.0, 0.0};
                double de_zero[3] = {0.0, 0.0, 0.0};
                fill_du_dv(da_dp0, db_dp0, dc_zero, dd_dp0, de_dp0, du_dp0, dv_dp0);
                fill_du_dv(da_dp1, db_dp1, dc_zero, dd_dp1, de_zero, du_dp1, dv_dp1);
                fill_du_dv(da_zero, db_dq0, dc_dq0, dd_dq0, de_dq0, du_dq0, dv_dq0);
                fill_du_dv(da_zero, db_dq1, dc_dq1, dd_zero, de_dq1, du_dq1, dv_dq1);
            } else if (use_u_fixed) {
                if (!v_on_boundary && c > 1e-12) {
                    double denom_c = c * c;
                    for (int k = 0; k < 3; ++k) {
                        dv_dp0[k] = ((db_dp0[k] * u + de_dp0[k]) * c - (b * u + e) * 0.0) / denom_c;
                        dv_dp1[k] = ((db_dp1[k] * u + 0.0) * c - (b * u + e) * 0.0) / denom_c;
                        dv_dq0[k] = ((db_dq0[k] * u + de_dq0[k]) * c - (b * u + e) * dc_dq0[k]) / denom_c;
                        dv_dq1[k] = ((db_dq1[k] * u + de_dq1[k]) * c - (b * u + e) * dc_dq1[k]) / denom_c;
                    }
                }
            } else if (use_v_fixed) {
                if (!u_on_boundary && a > 1e-12) {
                    double num = (v <= clamp_eps) ? -d : (b - d);
                    double denom_a = a * a;
                    for (int k = 0; k < 3; ++k) {
                        double dnum_dp0 = (v <= clamp_eps) ? -dd_dp0[k] : (db_dp0[k] - dd_dp0[k]);
                        double dnum_dp1 = (v <= clamp_eps) ? -dd_dp1[k] : (db_dp1[k] - dd_dp1[k]);
                        double dnum_dq0 = (v <= clamp_eps) ? -dd_dq0[k] : (db_dq0[k] - dd_dq0[k]);
                        double dnum_dq1 = (v <= clamp_eps) ? 0.0 : db_dq1[k];

                        du_dp0[k] = (dnum_dp0 * a - num * da_dp0[k]) / denom_a;
                        du_dp1[k] = (dnum_dp1 * a - num * da_dp1[k]) / denom_a;
                        du_dq0[k] = (dnum_dq0 * a - num * 0.0) / denom_a;
                        du_dq1[k] = (dnum_dq1 * a - num * 0.0) / denom_a;
                    }
                }
            }

            double d1_dot_g = d1[0] * gvec[0] + d1[1] * gvec[1] + d1[2] * gvec[2];
            double d2_dot_g = d2[0] * gvec[0] + d2[1] * gvec[1] + d2[2] * gvec[2];

            for (int dch = 0; dch < 3; ++dch) {
                double g = gvec[dch];
                addg(i, dch, (1.0 - u) * g + d1_dot_g * du_dp0[dch] - d2_dot_g * dv_dp0[dch]);
                addg(i1, dch, u * g + d1_dot_g * du_dp1[dch] - d2_dot_g * dv_dp1[dch]);
                addg(j, dch, -(1.0 - v) * g + d1_dot_g * du_dq0[dch] - d2_dot_g * dv_dq0[dch]);
                addg(j1, dch, -v * g + d1_dot_g * du_dq1[dch] - d2_dot_g * dv_dq1[dch]);
            }
        }
    }

    *energy_out = E;
}

} // extern "C"

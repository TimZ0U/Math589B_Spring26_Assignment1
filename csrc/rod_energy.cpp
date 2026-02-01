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

    // Small 3D vector helper (no dynamic allocs).
    struct Vec3 {
        double x, y, z;
        Vec3() : x(0), y(0), z(0) {}
        Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
    };
    auto v_add = [](const Vec3& a, const Vec3& b) { return Vec3(a.x+b.x, a.y+b.y, a.z+b.z); };
    auto v_sub = [](const Vec3& a, const Vec3& b) { return Vec3(a.x-b.x, a.y-b.y, a.z-b.z); };
    auto v_mul = [](const Vec3& a, double s) { return Vec3(a.x*s, a.y*s, a.z*s); };
    auto dot   = [](const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; };
    auto norm2 = [&](const Vec3& a) { return dot(a,a); };
    auto norm  = [&](const Vec3& a) { return std::sqrt(norm2(a)); };

    auto node = [&](int i) -> Vec3 {
        return Vec3(get(i,0), get(i,1), get(i,2));
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

    // ---- Segment–segment WCA self-avoidance ----
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
    // IMPORTANT: dependence of the closest points on endpoints is included
    // by distributing forces to endpoints using weights (1-u), u and (1-v), v.
    {
        const double rc = std::pow(2.0, 1.0/6.0) * sigma;
        const double rc2 = rc * rc;
        const double tiny = 1e-12;

        // Closest points between segments AB and CD:
        // returns parameters s,t in [0,1] for P = A + s(B-A), Q = C + t(D-C)
        auto closest_params = [&](const Vec3& A, const Vec3& B,
                                  const Vec3& C, const Vec3& D,
                                  double& s_out, double& t_out) {
            // Based on "Real-Time Collision Detection" style segment-segment distance.
            const Vec3 u = v_sub(B, A);
            const Vec3 v = v_sub(D, C);
            const Vec3 w = v_sub(A, C);

            const double a = dot(u,u);  // always >= 0
            const double b = dot(u,v);
            const double c = dot(v,v);  // always >= 0
            const double d = dot(u,w);
            const double e = dot(v,w);
            const double Dden = a*c - b*b;
            const double EPS = 1e-15;

            double sN, sD = Dden;
            double tN, tD = Dden;

            if (Dden < EPS) {
                // Almost parallel
                sN = 0.0;
                sD = 1.0;
                tN = e;
                tD = c;
            } else {
                sN = (b*e - c*d);
                tN = (a*e - b*d);

                if (sN < 0.0) {
                    sN = 0.0;
                    tN = e;
                    tD = c;
                } else if (sN > sD) {
                    sN = sD;
                    tN = e + b;
                    tD = c;
                }
            }

            if (tN < 0.0) {
                tN = 0.0;
                // Recompute s for this t
                if (-d < 0.0) {
                    sN = 0.0;
                } else if (-d > a) {
                    sN = sD;
                } else {
                    sN = -d;
                    sD = a;
                }
            } else if (tN > tD) {
                tN = tD;
                // Recompute s for this t
                const double d2 = -d + b;
                if (d2 < 0.0) {
                    sN = 0.0;
                } else if (d2 > a) {
                    sN = sD;
                } else {
                    sN = d2;
                    sD = a;
                }
            }

            const double s = (std::abs(sN) < EPS ? 0.0 : sN / sD);
            const double t = (std::abs(tN) < EPS ? 0.0 : tN / tD);

            s_out = std::min(1.0, std::max(0.0, s));
            t_out = std::min(1.0, std::max(0.0, t));
        };

        for (int i = 0; i < N; ++i) {
            const int ip1 = idx(i+1);
            const Vec3 A = node(i);
            const Vec3 B = node(ip1);

            for (int j = i+1; j < N; ++j) {
                // Exclusions: skip same/adjacent segments, including wrap neighbors.
                // Segment i is adjacent to segments i-1 and i+1 (mod N).
                if (j == idx(i) || j == idx(i-1) || j == idx(i+1)) continue;

                const int jp1 = idx(j+1);
                const Vec3 C = node(j);
                const Vec3 D = node(jp1);

                // Compute closest parameters u*, v*
                double ustar = 0.0, vstar = 0.0;
                closest_params(A, B, C, D, ustar, vstar);

                // Closest points
                const Vec3 AB = v_sub(B, A);
                const Vec3 CD = v_sub(D, C);
                const Vec3 Pi = v_add(A, v_mul(AB, ustar));
                const Vec3 Pj = v_add(C, v_mul(CD, vstar));
                const Vec3 rvec = v_sub(Pi, Pj);

                double d2 = norm2(rvec);
                if (d2 >= rc2) continue;

                double d = std::sqrt(std::max(d2, tiny));
                double inv_d = 1.0 / d;

                // WCA potential: U(d)=4 eps[(sigma/d)^12 - (sigma/d)^6] + eps for d<rc
                const double sr = sigma * inv_d;
                const double sr2 = sr * sr;
                const double sr4 = sr2 * sr2;
                const double sr6 = sr4 * sr2;
                const double sr12 = sr6 * sr6;

                const double U = 4.0 * eps * (sr12 - sr6) + eps;
                E += U;

                // dU/dd = (24*eps/d) * (sr6 - 2*sr12)
                const double dU_dd = (24.0 * eps * inv_d) * (sr6 - 2.0 * sr12);

                // ∂U/∂r = dU/dd * r/d  => gradPi = (dU_dd/d)*r
                const double coeff = dU_dd * inv_d; // = dU/dd * (1/d)
                const Vec3 gradPi = v_mul(rvec, coeff);
                const Vec3 gradPj = v_mul(gradPi, -1.0);

                // Distribute to segment endpoints using linear weights:
                // Pi = (1-u)A + u B
                // Pj = (1-v)C + v D
                const double wA = 1.0 - ustar;
                const double wB = ustar;
                const double wC = 1.0 - vstar;
                const double wD = vstar;

                // Add gradients for x_i, x_{i+1}, x_j, x_{j+1}
                addg(i,   0, wA * gradPi.x);  addg(i,   1, wA * gradPi.y);  addg(i,   2, wA * gradPi.z);
                addg(ip1, 0, wB * gradPi.x);  addg(ip1, 1, wB * gradPi.y);  addg(ip1, 2, wB * gradPi.z);

                addg(j,   0, wC * gradPj.x);  addg(j,   1, wC * gradPj.y);  addg(j,   2, wC * gradPj.z);
                addg(jp1, 0, wD * gradPj.x);  addg(jp1, 1, wD * gradPj.y);  addg(jp1, 2, wD * gradPj.z);
            }
        }
    }

    *energy_out = E;
}

} // extern "C"

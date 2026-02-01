
extern "C" {

int rod_api_version() { return 2; }

static inline double dot3(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline void sub3(double out[3], const double a[3], const double b[3]) {
    out[0] = a[0]-b[0];
    out[1] = a[1]-b[1];
    out[2] = a[2]-b[2];
}

static inline double norm3(const double a[3]) {
    return std::sqrt(dot3(a,a));
}

// Robust closest points parameters between segments P0P1 and Q0Q1.
// Returns s,t in [0,1]x[0,1].
static inline void closest_params_segment_segment(
    const double P0[3], const double P1[3],
    const double Q0[3], const double Q1[3],
    double &s, double &t
) {
    const double EPS = 1e-12;

    double d1[3] = { P1[0]-P0[0], P1[1]-P0[1], P1[2]-P0[2] };
    double d2[3] = { Q1[0]-Q0[0], Q1[1]-Q0[1], Q1[2]-Q0[2] };
    double r[3]  = { P0[0]-Q0[0], P0[1]-Q0[1], P0[2]-Q0[2] };

    double a = dot3(d1,d1);
    double e = dot3(d2,d2);
    double f = dot3(d2,r);

    if (a <= EPS && e <= EPS) { s = 0.0; t = 0.0; return; }
    if (a <= EPS) {
        s = 0.0;
        t = (e > EPS) ? (f / e) : 0.0;
        t = std::clamp(t, 0.0, 1.0);
        return;
    }

    double c = dot3(d1,r);
    if (e <= EPS) {
        t = 0.0;
        s = -c / a;
        s = std::clamp(s, 0.0, 1.0);
        return;
    }

    double b = dot3(d1,d2);
    double denom = a*e - b*b;

    double sN, sD = denom;
    double tN, tD = denom;

    if (denom < EPS) {
        sN = 0.0; sD = 1.0;
        tN = f;   tD = e;
    } else {
        sN = (b*f - c*e);
        tN = (a*f - b*c);
    }

    if (sN < 0.0) {
        sN = 0.0;
        tN = f;
        tD = e;
    } else if (sN > sD) {
        sN = sD;
        tN = f + b;
        tD = e;
    }

    if (tN < 0.0) {
        tN = 0.0;
        sN = -c;
        sD = a;
        sN = std::clamp(sN, 0.0, sD);
    } else if (tN > tD) {
        tN = tD;
        sN = b - c;
        sD = a;
        sN = std::clamp(sN, 0.0, sD);
    }

    s = (std::abs(sD) > EPS) ? (sN / sD) : 0.0;
    t = (std::abs(tD) > EPS) ? (tN / tD) : 0.0;

    s = std::clamp(s, 0.0, 1.0);
    t = std::clamp(t, 0.0, 1.0);
}

static inline void lerp3(double out[3], const double A[3], const double B[3], double u) {
    out[0] = A[0] + u*(B[0]-A[0]);
    out[1] = A[1] + u*(B[1]-A[1]);
    out[2] = A[2] + u*(B[2]-A[2]);
}

static inline double wca_U(double d, double eps, double sigma) {
    const double rc = std::pow(2.0, 1.0/6.0) * sigma;
    if (d >= rc) return 0.0;
    d = std::max(d, 1e-12);
    double s = sigma / d;
    double s2 = s*s;
    double s6 = s2*s2*s2;
    double s12 = s6*s6;
    return 4.0*eps*(s12 - s6) + eps;
}

static inline double segment_segment_wca_energy(
    const double Pi0[3], const double Pi1[3],
    const double Pj0[3], const double Pj1[3],
    double eps, double sigma
) {
    double u, v;
    closest_params_segment_segment(Pi0, Pi1, Pj0, Pj1, u, v);
    double Ci[3], Cj[3], r[3];
    lerp3(Ci, Pi0, Pi1, u);
    lerp3(Cj, Pj0, Pj1, v);
    sub3(r, Ci, Cj);
    double d = norm3(r);
    return wca_U(d, eps, sigma);
}

void rod_energy_grad(
    int N,
    const double* x,
    double kb,
    double ks,
    double l0,
    double kc,
    double eps,
    double sigma,
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

    // ---- Bending: kb * sum ||x_{i+1} - 2 x_i + x_{i-1}||^2
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

    // ---- Stretching: ks * sum (||x_{i+1}-x_i|| - l0)^2
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

    // ---- Segment–segment WCA self-avoidance
    // Central-difference gradient over the 4 endpoints (12 scalars).
    if (eps != 0.0 && sigma > 0.0) {
        auto seg_circ_dist = [&](int a, int b) {
            int da = std::abs(a - b);
            return std::min(da, N - da);
        };

        // Central difference step (much more accurate than forward diff for stiff WCA)
        const double h = 1e-8;

        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                if (seg_circ_dist(i, j) <= 2) continue;

                double Pi0[3] = { get(i,0),   get(i,1),   get(i,2) };
                double Pi1[3] = { get(i+1,0), get(i+1,1), get(i+1,2) };
                double Pj0[3] = { get(j,0),   get(j,1),   get(j,2) };
                double Pj1[3] = { get(j+1,0), get(j+1,1), get(j+1,2) };

                double U0 = segment_segment_wca_energy(Pi0, Pi1, Pj0, Pj1, eps, sigma);
                if (U0 == 0.0) continue;
                E += U0;

                int ids[4] = { i, i+1, j, j+1 };

                for (int a = 0; a < 4; ++a) {
                    int node = ids[a];
                    for (int d = 0; d < 3; ++d) {
                        // +h endpoints
                        double Ai0p[3] = { Pi0[0], Pi0[1], Pi0[2] };
                        double Ai1p[3] = { Pi1[0], Pi1[1], Pi1[2] };
                        double Aj0p[3] = { Pj0[0], Pj0[1], Pj0[2] };
                        double Aj1p[3] = { Pj1[0], Pj1[1], Pj1[2] };

                        // -h endpoints
                        double Ai0m[3] = { Pi0[0], Pi0[1], Pi0[2] };
                        double Ai1m[3] = { Pi1[0], Pi1[1], Pi1[2] };
                        double Aj0m[3] = { Pj0[0], Pj0[1], Pj0[2] };
                        double Aj1m[3] = { Pj1[0], Pj1[1], Pj1[2] };

                        if (a == 0) { Ai0p[d] += h; Ai0m[d] -= h; }
                        if (a == 1) { Ai1p[d] += h; Ai1m[d] -= h; }
                        if (a == 2) { Aj0p[d] += h; Aj0m[d] -= h; }
                        if (a == 3) { Aj1p[d] += h; Aj1m[d] -= h; }

                        double Up = segment_segment_wca_energy(Ai0p, Ai1p, Aj0p, Aj1p, eps, sigma);
                        double Um = segment_segment_wca_energy(Ai0m, Ai1m, Aj0m, Aj1m, eps, sigma);

                        double dU = (Up - Um) / (2.0*h);
                        addg(node, d, dU);
                    }
                }
            }
        }
    }

    *energy_out = E;
}

} // extern "C"

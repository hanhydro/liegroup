import json
import os
import numpy as np

import fedopt_stability as fs
from fedopt_stability import so3, karcher_mean, clip

RNG = np.random.default_rng(0)

K = 0.25
INJ = np.pi
DIMG = 3
r = 0.30
eta, tau, gamma = 0.5, 1, 1.0
eta_eff = gamma * eta * tau
C, B = 0.35, 0.90
rho_star = 1.30
N, T = 500, 20
eps, delta = 1.0, 1e-5
R_star = r
p_tail = 0.05


def cot(x):
    return np.cos(x) / np.sin(x)


def analytic_constants():
    sK = np.sqrt(K)
    rho_loc = eta * r
    r_plus = max(r + rho_loc, rho_star)
    L = 1.0
    D = r
    mu = sK * D * cot(sK * D)
    mu_enlarged_previous = sK * r_plus * cot(sK * r_plus)
    q = sK * rho_star
    L_Pi_previous_bound = q / np.sin(q)
    L_Pi = 1.0
    eps_step = max(abs(1 - eta_eff * mu), abs(1 - eta_eff * L))
    x = sK * eta_eff * L * R_star
    beta_Jac = np.sin(x) / x - np.cos(x)
    kappa_pop = eps_step + beta_Jac
    kappa = L_Pi * eps_step
    G_max = r_plus
    D_w = r
    mu_w = sK * D_w * cot(sK * D_w)
    q_sq = max(abs(1 - eta_eff * mu_w), abs(1 - eta_eff * L))
    return dict(K=K, dimG=DIMG, inj=INJ, r=r, rho_star=rho_star, rho_loc=rho_loc,
                L_Pi_previous_bound=L_Pi_previous_bound, D_w=D_w, mu_w=mu_w, q_sq=q_sq,
                mu_enlarged_previous=mu_enlarged_previous,
                r_plus=r_plus, eta=eta, tau=tau, gamma=gamma, eta_eff=eta_eff, L=L, mu=mu,
                G_max=G_max, D=D, L_Pi=L_Pi, eps_step=eps_step, beta_Jac=beta_Jac,
                kappa_pop=kappa_pop, kappa=kappa, C=C, B=B, N=N, T=T, eps=eps, delta=delta,
                R_star=R_star, c_Exp=1.0)


def hypotheses(c):
    K_pos = c["K"]
    Rconv = np.pi / (2 * np.sqrt(K_pos))
    checks = []

    def add(name, slack, cat="feasibility"):
        checks.append({"condition": name, "slack": float(slack), "pass": bool(slack > 0),
                       "category": cat})

    add("0 <= r < rho*", min(c["r"], c["rho_star"] - c["r"]))
    add("0 < mu <= L", min(c["mu"], c["L"] - c["mu"]))
    add("0 < eta_eff < 2/L", min(c["eta_eff"], 2.0 / c["L"] - c["eta_eff"]))
    add("2 r_plus < min(inj, pi/(2 sqrt K_max))", min(c["inj"], Rconv) - 2 * c["r_plus"])
    add("eta_eff L R_star <= r_plus - r (population tube)",
        (c["r_plus"] - c["r"]) - c["eta_eff"] * c["L"] * c["R_star"])
    add("C >= U_client = rho_loc (inactive client clip)", c["C"] - c["rho_loc"])
    add("0 <= C <= B", min(c["C"], c["B"] - c["C"]))
    add("r + gamma B < rho* (deterministic containment)",
        c["rho_star"] - c["r"] - c["gamma"] * c["B"])
    add("kappa = L_Pi eps_step < 1 (one-point contraction)", 1.0 - c["kappa"])
    add("R_star <= gamma B (squared theorem)", c["gamma"] * c["B"] - c["R_star"])
    add("q_sq < 1 (squared theorem)", 1.0 - c["q_sq"])
    add("L_Pi = 1 (nonexpansive radial return)", 1.0 - abs(c["L_Pi"] - 1.0), "derived")
    add("0 <= R_star <= 2r", min(c["R_star"], 2 * c["r"] - c["R_star"]), "derived")
    add("G_max = r_plus finite (gradient bound on B^+)", c["G_max"], "derived")
    return checks


def privacy_scale(c):
    Delta = 2.0 * c["C"] / c["N"]
    Ld = -np.log(c["delta"])
    gap = c["eps"] / (np.sqrt(Ld + c["eps"]) + np.sqrt(Ld))
    sigma = Delta * np.sqrt(c["T"]) / (np.sqrt(2.0) * gap)
    S = c["T"] / sigma ** 2
    root_A = Delta * np.sqrt(S / 2.0)
    eps_rt = float(root_A * (root_A + 2 * np.sqrt(Ld)))
    return Delta, sigma, eps_rt


def certified_bounds(c, sigma):
    k, T, LPi, g, cE = c["kappa"], c["T"], c["L_Pi"], c["gamma"], c["c_Exp"]
    Rcap = np.sqrt(c["C"] ** 2 + c["B"] ** 2)
    qbar = min(sigma * np.sqrt(c["dimG"]), Rcap)
    a = np.sqrt(2.0) * c["C"] * np.sqrt(c["dimG"] * T) / (c["N"] * Rcap)
    Ld = np.log(1.0 / c["delta"])
    eps_cap = a ** 2 + 2 * a * np.sqrt(Ld)
    fin = (1 - k ** T) / (1 - k)
    bfed = 0.0
    exp_recur = k ** T * c["R_star"] + LPi * fin * (bfed + cE * g * qbar)
    w = np.array([k ** (T - 1 - t) for t in range(T)])
    Sw, Sw2 = w.sum(), (w ** 2).sum()
    tau_t = min(sigma, (c["B"] + c["C"]) / 2.0)
    dev = np.sqrt(2 * np.log(1.0 / p_tail) * Sw2 * tau_t ** 2)
    perrun_recur = k ** T * c["R_star"] + LPi * (bfed * Sw + cE * g * (Sw * qbar + dev))
    Rb = c["R_star"]
    from scipy.stats import ncx2
    qs = c["q_sq"]
    vT = g ** 2 * sigma ** 2 * (1 - qs ** (2 * T)) / (1 - qs ** 2)
    sq_bound = float(np.sqrt(qs ** (2 * T) * c["R_star"] ** 2 + c["dimG"] * vT))
    lam = c["R_star"] ** 2 * qs ** (2 * T) / vT
    sq_q95 = float(np.sqrt(vT * ncx2.ppf(1 - p_tail, c["dimG"], lam)))
    return dict(Rcap=Rcap, qbar=qbar, gaussian_branch=bool(sigma * np.sqrt(c["dimG"]) < Rcap),
                eps_cap=eps_cap, finite_horizon_factor=fin, R_star_bound=Rb, two_r=2.0 * c["r"],
                expected_recurrence=exp_recur,
                expected_certified=min(exp_recur, Rb),
                expected_active=("recurrence" if exp_recur < Rb else "target-relative"),
                perrun_recurrence=perrun_recur,
                perrun_certified=min(perrun_recur, Rb),
                perrun_active=("recurrence" if perrun_recur < Rb else "target-relative"),
                tau_t=tau_t, squared_rms_bound=min(sq_bound, Rb), squared_q95_bound=min(sq_q95, Rb),
                squared_vT=vT, squared_noncentrality=lam)


def run_mechanism(grp, thetastar, theta0, sigma, M):
    def one():
        theta = theta0.copy()
        maxreach = 0.0
        for _ in range(T):
            ti = theta.copy()
            for _ in range(tau):
                ti = grp.exp(ti, eta * grp.log(ti, thetastar))
            ubar = clip(grp, grp.log(theta, ti), C)
            z = grp.sample_tangent(sigma)
            vhat = clip(grp, ubar + z, B)
            pre = grp.exp(theta, gamma * vhat)
            maxreach = max(maxreach, grp.dist(pre, thetastar))
            theta = fs.retract(grp, thetastar, pre, r)
        return grp.dist(theta, thetastar), maxreach
    res = [one() for _ in range(M)]
    ds = np.array([d for d, _ in res])
    reach = np.array([m for _, m in res])
    return ds, reach


def main():
    c = analytic_constants()
    checks = hypotheses(c)
    Delta, sigma, eps_rt = privacy_scale(c)
    c["sigma"] = sigma
    bnd = certified_bounds(c, sigma)

    grp = so3()
    thetastar = grp.I.copy()
    xi = grp.sample_tangent(1.0)
    d0 = 0.9 * R_star
    theta0 = grp.exp(thetastar, xi * (d0 / grp.norm(xi)))
    M = 4000
    ds, reach = run_mechanism(grp, thetastar, theta0, sigma, M)
    emp_rms = float(np.sqrt(np.mean(ds ** 2)))
    emp_tail = float(np.quantile(ds, 1 - p_tail))
    max_reach = float(reach.max())
    cov = float(np.mean(ds <= bnd["squared_q95_bound"]))
    zc = 1.96; den = 1 + zc ** 2 / M; ctr = (cov + zc ** 2 / (2 * M)) / den
    half = zc * np.sqrt(cov * (1 - cov) / M + zc ** 2 / (4 * M ** 2)) / den
    cov_ci = [float(ctr - half), float(ctr + half)]

    out = {
        "setting": "certified synthetic SO(3), squared-distance objective, homogeneous clients",
        "analytic_constants": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in c.items()},
        "constant_sources": {
            "K_max": "SO(3) constant curvature 1/4 under <X,Y>=-tr(XY)/2",
            "L": "radial Hessian of squared distance = 1",
            "mu": "sqrt(K) r cot(sqrt(K) r): tangential Hessian lower bound on the working ball",
            "L_Pi": "1: the radial return is nonexpansive for a bi-invariant metric",
            "mu_w": "sqrt(K) r cot(sqrt(K) r): working-ball Hessian lower bound (squared theorem)",
            "rho_loc": "eta r one-step displacement (all anchors = center)",
            "kappa": "L_Pi eps_step, the one-point factor of Lemma 5.3, all analytic"},
        "hypotheses": checks,
        "all_hypotheses_pass": all(x["pass"] for x in checks),
        "min_feasibility_slack": min(x["slack"] for x in checks if x["category"] == "feasibility"),
        "privacy": {"epsilon": eps, "delta": delta, "sensitivity_2C_over_N": Delta,
                    "sigma": sigma, "recomputed_epsilon": eps_rt,
                    "epsilon_roundtrip_ok": bool(abs(eps_rt - eps) < 1e-6)},
        "bounds": {k: (float(v) if isinstance(v, (int, float, np.floating, bool)) else v)
                   for k, v in bnd.items()},
        "empirical": {"trials": M, "rms": emp_rms, "tail_q95": emp_tail,
                      "coverage_of_squared_q95_bound": cov, "coverage_wilson95": cov_ci,
                      "max_pre_retraction_reach": max_reach, "rho_star": rho_star,
                      "declared_domain_respected": bool(max_reach < rho_star)},
    }
    path = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "certified_so3.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"kappa (analytic) = {c['kappa']:.4f}   mu={c['mu']:.4f}  L={c['L']}  "
          f"L_Pi={c['L_Pi']:.4f}  beta_Jac={c['beta_Jac']:.5f}")
    print(f"all hypotheses pass: {out['all_hypotheses_pass']}   "
          f"min feasibility slack = {out['min_feasibility_slack']:.4f}  G_max={c['G_max']}")
    print(f"sigma={sigma:.5f}  eps round-trip={eps_rt:.6f}  Rcap={bnd['Rcap']:.4f}  "
          f"eps_cap={bnd['eps_cap']:.4f}  Gaussian branch={bnd['gaussian_branch']}")
    print(f"EXPECTED : recurrence={bnd['expected_recurrence']:.4f}  R*={bnd['R_star_bound']:.4f}  "
          f"-> certified={bnd['expected_certified']:.4f} ({bnd['expected_active']} active)")
    print(f"PER-RUN  : recurrence={bnd['perrun_recurrence']:.4f}  R*={bnd['R_star_bound']:.4f}  "
          f"-> certified={bnd['perrun_certified']:.4f} ({bnd['perrun_active']} active)")
    print(f"SQUARED  : q={c['q_sq']:.4f} mu_w={c['mu_w']:.5f} rms bound={bnd['squared_rms_bound']:.5f} q95 bound={bnd['squared_q95_bound']:.5f}")
    print(f"COVERAGE : fraction within squared q95 bound = {cov:.4f}  Wilson95 = [{cov_ci[0]:.4f}, {cov_ci[1]:.4f}]")
    print(f"EMPIRICAL: rms={emp_rms:.4f}  tail_q95={emp_tail:.4f}  "
          f"max_reach={max_reach:.4f} < rho*={rho_star} : {out['empirical']['declared_domain_respected']}")


if __name__ == "__main__":
    main()

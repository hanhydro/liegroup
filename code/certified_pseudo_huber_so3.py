import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import ncx2

P = dict(h=0.10, R_data=0.05, r=0.10, rho=0.45, N=2000, eta=0.5, gamma=1.0, B=0.30,
         eps=1.0, delta=1e-5, T=20, trials=4000, seed=20261027)
P["C"] = P["eta"] * P["h"]


def lam_min(s, h):
    radial = (1 + s ** 2 / h ** 2) ** -1.5
    transverse = (s / 2) / np.tan(s / 2) / np.sqrt(1 + s ** 2 / h ** 2)
    return min(radial, transverse)


def sigma_rdp(C, T, N, eps, delta):
    Ld = -np.log(delta); gap = eps / (np.sqrt(Ld + eps) + np.sqrt(Ld))
    return 2 * C / N * np.sqrt(T) / (np.sqrt(2) * gap)


def clip_rows(v, R):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(1.0, n / R)


def main():
    p = P
    h, r, a, rho, alpha = p["h"], p["r"], p["R_data"], p["rho"], p["gamma"] * p["eta"]
    A0, R_star = a, r + a
    r_plus = max(r + p["eta"] * h, rho)
    mu_w = lam_min(r + a, h)
    q = max(abs(1 - alpha * mu_w), abs(1 - alpha))
    mu_big = lam_min(r_plus + a, h)
    x = 0.5 * alpha * R_star
    beta = np.sin(x) / x - np.cos(x)
    kappa_pop = max(abs(1 - alpha * mu_w), abs(1 - alpha)) + beta
    kappa = max(abs(1 - alpha * mu_w), abs(1 - alpha))
    kappa_enlarged_previous = max(abs(1 - alpha * mu_big), abs(1 - alpha))
    slacks = dict(target_inside_clip=p["gamma"] * p["B"] - R_star,
                  containment=rho - r - p["gamma"] * p["B"],
                  geometry=np.pi - 2 * r_plus, q_below_one=1 - q,
                  client_clip_inactive=p["C"] - p["eta"] * h + 1e-15,
                  tube=(r_plus - r) - alpha * R_star)
    assert all(v > 0 for v in slacks.values()), slacks
    sigma = sigma_rdp(p["C"], p["T"], p["N"], p["eps"], p["delta"])
    T = p["T"]
    vT = p["gamma"] ** 2 * sigma ** 2 * (1 - q ** (2 * T)) / (1 - q ** 2)
    sq_rms = min(np.sqrt(q ** (2 * T) * A0 ** 2 + 3 * vT), R_star)
    sq_q95 = min(np.sqrt(vT * ncx2.ppf(0.95, 3, A0 ** 2 * q ** (2 * T) / vT)), R_star)
    H = (1 - kappa ** T) / (1 - kappa)
    thm71 = min(kappa ** T * A0 + H * p["gamma"] * sigma * np.sqrt(3), R_star)

    target = Rotation.from_rotvec([0.0, 0.0, 0.04])
    dirs = np.vstack((np.eye(3), np.ones(3) / np.sqrt(3)))
    anchors = target * Rotation.from_rotvec(0.005 * np.vstack((dirs, -dirs)))
    assert np.max(anchors.magnitude()) < a
    A = anchors.as_matrix()
    rng = np.random.default_rng(p["seed"])
    M = p["trials"]
    theta = np.repeat(np.eye(3)[None], M, axis=0)
    for t in range(T):
        rel = np.einsum("mji,kjl->mkil", theta, A)
        rv = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(M, len(A), 3)
        dd = np.linalg.norm(rv, axis=-1, keepdims=True)
        u = clip_rows(p["eta"] * rv / np.sqrt(1 + dd ** 2 / h ** 2), p["C"]).mean(axis=1)
        v = clip_rows(u + sigma * rng.standard_normal(u.shape), p["B"])
        end = np.einsum("mij,mjk->mik", theta, Rotation.from_rotvec(p["gamma"] * v).as_matrix())
        theta = Rotation.from_rotvec(clip_rows(Rotation.from_matrix(end).as_rotvec(), r)).as_matrix()
    err = (target.inv() * Rotation.from_matrix(theta)).magnitude()
    rms, q95 = float(np.sqrt(np.mean(err ** 2))), float(np.quantile(err, 0.95))
    cov = float(np.mean(err <= sq_q95))
    out = dict(public=p, A0=A0, R_star=R_star, r_plus=r_plus, mu_w=mu_w, q=q,
               mu_enlarged=mu_big, kappa_theorem_7_1=kappa, kappa_enlarged_previous=kappa_enlarged_previous, sigma=sigma,
               squared_rms_bound=float(sq_rms), squared_q95_bound=float(sq_q95),
               theorem_7_1_rms_bound=float(thm71), observed_rms=rms, observed_q95=q95,
               coverage_of_squared_q95_bound=cov, slacks={k: float(v) for k, v in slacks.items()})
    path = Path(__file__).resolve().parent.parent / "results" / "certified_pseudo_huber_so3.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"mu_w={mu_w:.10f} q={q:.10f} mu_enlarged={mu_big:.6g} kappa(Thm 7.1, L_Pi=1)={kappa:.6f}")
    print(f"sigma={sigma:.10f} squared RMS bound={sq_rms:.6f} q95 bound={sq_q95:.6f} Thm 7.1 bound={thm71:.4f}")
    print(f"observed RMS={rms:.6f} q95={q95:.6f} coverage={cov:.4f}")


if __name__ == "__main__":
    main()

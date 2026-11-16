import numpy as np
from fedopt_stability import so3, karcher_mean, clip, retract

GRP = so3(); D = GRP.dim
CENTER = GRP.I.copy()
R, GAMMA, C, TAU, ETA, T = 0.20, 1.0, 0.10, 1, 0.5, 10
RHO_STAR, DELTA, MTRAJ = 0.60, 1e-5, 1000
EPS = [1.0, 2.0, 4.0]
B_LIST = [0.15, 0.25, 0.35, 0.39]
N_ANCH, A_SUPP = 8, 0.15
_pub = np.random.default_rng(20260810)
def _ball(rng, radius):
    z = rng.normal(size=D); v = sum(z[k] * GRP.basis[k] for k in range(D)); n = GRP.norm(v)
    return v * (rng.uniform(0, radius) / n) if n > 0 else v * 0
ANCHORS = [GRP.exp(CENTER, _ball(_pub, A_SUPP)) for _ in range(N_ANCH)]
THETA_STAR = karcher_mean(GRP, ANCHORS)

def sigma_rdp(C, T, delta, N, eps):
    Delta = 2.0 * C / N; Ld = -np.log(delta); gap = eps / (np.sqrt(Ld + eps) + np.sqrt(Ld))
    return Delta * np.sqrt(T) / (np.sqrt(2.0) * gap)

def trajectory(sigma, B, Zstd):
    theta = CENTER.copy(); fires = 0
    for t in range(T):
        us = []
        for a in ANCHORS:
            ti = theta.copy()
            for _ in range(TAU):
                ti = GRP.exp(ti, ETA * GRP.log(ti, a))
            us.append(clip(GRP, GRP.log(theta, ti), C))
        v = sum(us) / len(ANCHORS) + sigma * sum(Zstd[t, k] * GRP.basis[k] for k in range(D))
        nv = GRP.norm(v)
        if nv > B:
            v = v * (B / nv); fires += 1
        theta = retract(GRP, CENTER, GRP.exp(theta, GAMMA * v), R)
    return fires, GRP.dist(theta, THETA_STAR)

out = {}
print("eps  B     clip-rate  RMS(rad)   (sigma, rule B_p for p=0.05)")
for eps in EPS:
    sig = sigma_rdp(C, T, DELTA, N_ANCH, eps)
    B_rule = C + sig * (np.sqrt(D) + np.sqrt(2 * np.log(20.0)))
    rng = np.random.default_rng(9000 + int(100 * eps)); Z = rng.standard_normal((MTRAJ, T, D))
    for B in B_LIST:
        fires = 0; errs = np.empty(MTRAJ)
        for m in range(MTRAJ):
            f, e = trajectory(sig, B, Z[m]); fires += f; errs[m] = e
        out[(eps, B)] = (fires / (MTRAJ * T), float(np.sqrt(np.mean(errs ** 2))))
        print(f"{eps:4.1f} {B:.2f}  {100*fires/(MTRAJ*T):5.1f}%   {out[(eps,B)][1]:.4f}    (sigma={sig:.4f}, B_0.05={B_rule:.2f})")
np.savez("../results/rq1_b_sweep.npz", eps=np.array(EPS), B=np.array(B_LIST),
         clip_rate=np.array([[out[(e, b)][0] for b in B_LIST] for e in EPS]),
         rms=np.array([[out[(e, b)][1] for b in B_LIST] for e in EPS]), mtraj=MTRAJ)
print("saved rq1_b_sweep.npz")

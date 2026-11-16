import numpy as np
from scipy.linalg import expm, logm, schur
from scipy.spatial.transform import Rotation

RNG = np.random.default_rng(0)

class LieGroup:
    def __init__(self, name, dim, identity, basis, ip):
        self.name, self.dim, self.I, self.basis, self.ip = name, dim, identity, basis, ip
    def norm(self, xi):
        return np.sqrt(max(self.ip(xi, xi).real, 0.0))
    def exp(self, g, xi):
        return g @ expm(xi)
    def log(self, g, h):
        xi = logm(np.linalg.solve(g, h))
        return self.project(xi)
    def dist(self, g, h):
        return self.norm(self.log(g, h))
    def sample_tangent(self, sigma):
        z = RNG.normal(0, sigma, self.dim)
        return sum(z[k] * self.basis[k] for k in range(self.dim))
    def project(self, xi):
        raise NotImplementedError

def _gram_schmidt(mats, ip):
    onb = []
    for M in mats:
        V = M.copy()
        for B in onb:
            V = V - ip(B, V) * B
        n = np.sqrt(max(ip(V, V).real, 0.0))
        if n > 1e-9:
            onb.append(V / n)
    return onb

def so3():
    ip = lambda X, Y: -0.5 * np.trace(X @ Y)
    G = [np.array([[0,0,0],[0,0,-1],[0,1,0]], float),
         np.array([[0,0,1],[0,0,0],[-1,0,0]], float),
         np.array([[0,-1,0],[1,0,0],[0,0,0]], float)]
    basis = _gram_schmidt(G, ip)
    grp = LieGroup("SO(3)", 3, np.eye(3), basis, ip)
    grp.project = lambda xi: 0.5 * (xi - xi.T).real
    def _vee(K): return np.array([K[2, 1], K[0, 2], K[1, 0]])
    def _exp_skew(K):
        w = _vee(K); th = np.linalg.norm(w)
        if th < 1e-12: return np.eye(3) + K
        return np.eye(3) + (np.sin(th)/th)*K + ((1-np.cos(th))/th**2)*(K @ K)
    def _exp(g, xi): return g @ _exp_skew(grp.project(xi))
    def _log(g, h):
        R = g.T @ h
        c = np.clip((np.trace(R) - 1)/2, -1.0, 1.0); th = np.arccos(c)
        if th < 1e-9: return 0.5*(R - R.T)
        if np.pi - th < 1e-4:
            a, b, c = Rotation.from_matrix(R).as_rotvec()
            return np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]])
        return (th/(2*np.sin(th)))*(R - R.T)
    grp.exp, grp.log = _exp, _log
    return grp

def sun(n):
    ip = lambda X, Y: 0.5 * np.trace(X.conj().T @ Y).real
    gens = []
    for a in range(n):
        for b in range(n):
            if a < b:
                E = np.zeros((n, n), complex); E[a, b] = 1; E[b, a] = -1; gens.append(E)
                F = np.zeros((n, n), complex); F[a, b] = 1j; F[b, a] = 1j; gens.append(F)
    for a in range(1, n):
        D = np.zeros((n, n), complex)
        for k in range(a): D[k, k] = 1j
        D[a, a] = -1j * a
        gens.append(D)
    basis = _gram_schmidt(gens, ip)
    grp = LieGroup(f"SU({n})", n*n - 1, np.eye(n, dtype=complex), basis, ip)
    def proj(xi):
        S = 0.5 * (xi - xi.conj().T)
        return S - np.trace(S) / n * np.eye(n)
    grp.project = proj
    def _exp(g, xi):
        X = proj(xi)
        mu, V = np.linalg.eigh(-1j * X)
        return g @ (V * np.exp(1j * mu)) @ V.conj().T
    def _log(g, h):
        U = np.linalg.solve(g, h)
        w, V = np.linalg.eig(U)
        phases = np.angle(w)
        winding = int(np.rint(np.sum(phases) / (2 * np.pi)))
        if winding == 0:
            return proj((V * np.log(w)) @ np.linalg.inv(V))

        triangular, V = schur(U, output="complex")
        phases = np.angle(np.diag(triangular))
        winding = int(np.rint(np.sum(phases) / (2 * np.pi)))
        order = np.argsort(phases, kind="stable")
        if winding > 0:
            phases[order[-winding:]] -= 2 * np.pi
        elif winding < 0:
            phases[order[:-winding]] += 2 * np.pi
        return proj((V * (1j * phases)) @ V.conj().T)
    grp.exp, grp.log = _exp, _log
    return grp

def karcher_mean(grp, A, iters=300, step=0.5):
    theta = A[0].copy()
    for _ in range(iters):
        g = sum(grp.log(theta, a) for a in A) / len(A)
        theta = grp.exp(theta, step * g)
    return theta

def clip(grp, xi, C):
    nrm = grp.norm(xi)
    return xi * min(1.0, C / nrm) if nrm > 0 else xi

def retract(grp, theta_star, x, r):
    xi = grp.log(theta_star, x); nrm = grp.norm(xi)
    return grp.exp(theta_star, xi * (r / nrm)) if nrm > r else x

def fedopt_run(grp, A, theta_star, theta0, eta, gamma, tau, C, sigma, r, T, noisy=True):
    theta = theta0.copy()
    for _ in range(T):
        us = []
        for a in A:
            ti = theta.copy()
            for _ in range(tau):
                ti = grp.exp(ti, eta * grp.log(ti, a))
            us.append(clip(grp, grp.log(theta, ti), C))
        ubar = sum(us) / len(A)
        noise = grp.sample_tangent(sigma) if noisy else 0 * ubar
        step = gamma * (ubar + noise)
        theta = retract(grp, theta_star, grp.exp(theta, step), r)
    return theta

def rms_to_star(grp, A, theta_star, M, **kw):
    ds = [grp.dist(fedopt_run(grp, A, theta_star, theta_star.copy(), **kw), theta_star)
          for _ in range(M)]
    return float(np.sqrt(np.mean(np.square(ds))))

def make_data(grp, N, spread):
    c = grp.exp(grp.I, grp.sample_tangent(0.0) + 0.0)
    return [grp.exp(c, grp.sample_tangent(spread)) for _ in range(N)]

def left_act(grp, h, A):
    return [h @ a for a in A]

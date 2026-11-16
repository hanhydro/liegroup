from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite, log, sqrt
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ScheduleResult:
    active_rounds: int
    precisions: Tuple[float, ...]
    sigmas: Tuple[float, ...]
    envelope_value: float
    uniform_value: float
    normalized_budget: float
    thresholds: Tuple[float, ...]
    tie_at: Optional[int] = None


def rdp_precision_budget(
    epsilon: float, delta: float, sensitivity: float
) -> float:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if sensitivity <= 0.0:
        raise ValueError("sensitivity must be positive")
    ell = -log(delta)
    gap = epsilon / (sqrt(ell + epsilon) + sqrt(ell))
    return 2.0 * (gap / sensitivity) ** 2


def contraction_weights(T: int, kappa: float) -> Tuple[float, ...]:
    if T < 1:
        raise ValueError("T must be at least one")
    if not 0.0 < kappa < 1.0:
        raise ValueError("kappa must lie in (0,1)")
    return tuple(kappa ** (T - 1 - t) for t in range(T))


def cap_height(B: float, C: float) -> float:
    if B <= 0.0 or C < 0.0 or C > B:
        raise ValueError("require B>0 and 0<=C<=B")
    return sqrt(B * B + C * C)


def hard_envelope(sigma: float, d: int, B: float, C: float) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if d < 1:
        raise ValueError("d must be positive")
    return min(sqrt(d) * sigma, cap_height(B, C))


def threshold_sequence(T: int, kappa: float) -> Tuple[float, ...]:
    if T < 1:
        raise ValueError("T must be at least one")
    if not 0.0 < kappa < 1.0:
        raise ValueError("kappa must lie in (0,1)")
    q = kappa ** (2.0 / 3.0)
    V_prev = 0.0
    out: List[float] = []
    for k in range(1, T + 1):
        u = q ** (k - 1)
        z = V_prev / u if u else inf
        if isfinite(z):
            root_next, root_prev = sqrt(z + 1.0), sqrt(z)
            D = root_next + root_prev * (root_prev / (root_next + root_prev))
            out.append(D * D)
        else:
            out.append(inf)
        V_prev += u
    return tuple(out)


def uniform_envelope_value(
    T: int, kappa: float, d: int, H: float, S: float
) -> float:
    if S <= 0.0:
        raise ValueError("S must be positive")
    total_weight = (1.0 - kappa**T) / (1.0 - kappa)
    normalized = S * H * H / d
    return H * total_weight * min(1.0, sqrt(T / normalized))


def relaxed_locality_schedule(
    T: int,
    kappa: float,
    d: int,
    B: float,
    C: float,
    S: float,
    tie_tolerance: float = 1e-12,
) -> ScheduleResult:
    if d < 1:
        raise ValueError("d must be positive")
    if S <= 0.0:
        raise ValueError("S must be positive")
    H = cap_height(B, C)
    normalized = S * H * H / d
    taus = threshold_sequence(T, kappa)

    k_star = 0
    tie_at: Optional[int] = None
    if normalized > 1.0:
        for k, tau in enumerate(taus, start=1):
            if not isfinite(tau):
                break
            scale = max(1.0, abs(tau), abs(normalized))
            if abs(normalized - tau) <= tie_tolerance * scale:
                tie_at = k
                k_star = max(0, k - 1)
                break
            if normalized > tau:
                k_star = k
            else:
                break

    weights = contraction_weights(T, kappa)
    precisions = [0.0] * T
    sigmas = [inf] * T
    if k_star == 0:
        value = H * sum(weights)
    else:
        active = range(T - k_star, T)
        V = sum(weights[t] ** (2.0 / 3.0) for t in active)
        for t in active:
            x = S * weights[t] ** (2.0 / 3.0) / V
            precisions[t] = x
            sigmas[t] = 1.0 / sqrt(x)
        value = H * sum(weights[: T - k_star])
        value += sqrt(d / S) * V**1.5

    uniform = uniform_envelope_value(T, kappa, d, H, S)
    return ScheduleResult(
        active_rounds=k_star,
        precisions=tuple(precisions),
        sigmas=tuple(sigmas),
        envelope_value=value,
        uniform_value=uniform,
        normalized_budget=normalized,
        thresholds=taus,
        tie_at=tie_at,
    )


def finite_ceiling_schedule(
    T: int,
    kappa: float,
    d: int,
    B: float,
    C: float,
    S: float,
    sigma_max: float,
) -> ScheduleResult:
    if d < 1 or S <= 0.0:
        raise ValueError("require d>=1 and S>0")
    H = cap_height(B, C)
    sigma_sat = H / sqrt(d)
    if sigma_max < sigma_sat:
        raise ValueError("sigma_max must be at least H/sqrt(d)")
    x_min = 1.0 / (sigma_max * sigma_max)
    if T * x_min > S:
        raise ValueError("finite scales already exceed the precision budget")

    weights = contraction_weights(T, kappa)
    x_cap = d / (H * H)
    best_value = H * sum(weights)
    best_k = 0
    best_x = [x_min] * T

    for k in range(1, T + 1):
        residual = S - (T - k) * x_min
        if residual <= 0.0:
            continue
        active = range(T - k, T)
        V = sum(weights[t] ** (2.0 / 3.0) for t in active)
        candidate_x = [x_min] * (T - k)
        candidate_x.extend(
            residual * weights[t] ** (2.0 / 3.0) / V for t in active
        )
        if min(candidate_x[T - k :]) <= x_cap:
            continue
        value = H * sum(weights[: T - k])
        value += sqrt(d) * V**1.5 / sqrt(residual)
        if value < best_value:
            best_value = value
            best_k = k
            best_x = candidate_x

    sigmas = tuple(1.0 / sqrt(x) for x in best_x)
    return ScheduleResult(
        active_rounds=best_k,
        precisions=tuple(best_x),
        sigmas=sigmas,
        envelope_value=best_value,
        uniform_value=uniform_envelope_value(T, kappa, d, H, S),
        normalized_budget=S * H * H / d,
        thresholds=threshold_sequence(T, kappa),
        tie_at=None,
    )


def exact_central_capped_moment(sigma: float, d: int, B: float) -> float:
    if sigma <= 0.0 or d < 1 or B <= 0.0:
        raise ValueError("require sigma>0, d>=1, B>0")
    try:
        from scipy.special import gammainc, gammaincc
    except ImportError as exc:
        raise RuntimeError("install scipy for exact moment evaluation") from exc
    s = d / 2.0
    ratio = (B / sigma) / sqrt(2.0)
    z = ratio * ratio
    if not isfinite(z):
        return sigma * sqrt(d)
    if z == 0.0:
        return B
    scaled_moment2 = s * (gammainc(s + 1.0, z) / z) + gammaincc(s, z)
    return B * sqrt(max(0.0, scaled_moment2))


def manuscript_example() -> ScheduleResult:
    T, kappa, d = 64, 0.5, 3
    B, C, N = 0.87, 0.4, 200
    epsilon, delta = 1.0, 1e-5
    sensitivity = 2.0 * C / N
    S = rdp_precision_budget(epsilon, delta, sensitivity)
    return relaxed_locality_schedule(T, kappa, d, B, C, S)


if __name__ == "__main__":
    result = manuscript_example()
    saturated = 64 - result.active_rounds
    reduction = 1.0 - result.envelope_value / result.uniform_value
    print(f"active late rounds: {result.active_rounds}")
    print(f"saturated early rounds: {saturated}/64 = {saturated/64:.4%}")
    print(f"normalized budget: {result.normalized_budget:.12f}")
    print(f"envelope optimum: {result.envelope_value:.12f}")
    print(f"uniform envelope: {result.uniform_value:.12f}")
    print(f"upper-bound reduction: {reduction:.4%}")

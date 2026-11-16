import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


PUBLIC = dict(
    anchor_radius=0.25, working_radius=0.30, declared_radius=1.00,
    eta=0.10, local_steps=2, server_step=1.0, client_clip=0.11,
    post_noise_clip=0.60, clients=2000, epsilon=1.0, privacy_delta=1e-5,
    maximum_rounds=64, curvature_min=0.25, curvature_max=0.25,
)
SEED = 20261005
TRIALS = 2000


def clip_rows(vectors, radius):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(1.0, lengths / radius)


def total_precision(client_clip):
    sensitivity = 2 * client_clip / PUBLIC["clients"]
    log_delta = -np.log(PUBLIC["privacy_delta"])
    difference = PUBLIC["epsilon"] / (np.sqrt(log_delta + PUBLIC["epsilon"]) + np.sqrt(log_delta))
    return 2 * (difference / sensitivity)**2


def constants():
    p = PUBLIC
    r, a = p["working_radius"], p["anchor_radius"]
    fraction = 1 - (1 - p["eta"]) ** p["local_steps"]
    rho_local = fraction * (r + a)
    r_plus = max(r + rho_local, p["declared_radius"])
    maximum_anchor_distance = r_plus + a
    sqrt_k = np.sqrt(p["curvature_max"])
    mu_argument = sqrt_k * maximum_anchor_distance
    mu_enlarged = mu_argument / np.tan(mu_argument)
    mu = (sqrt_k * (r + a)) / np.tan(sqrt_k * (r + a))
    q_decl = sqrt_k * p["declared_radius"]
    projection_previous_bound = q_decl / np.sin(q_decl)
    projection = 1.0
    alpha = p["server_step"] * p["eta"] * p["local_steps"]
    target_radius = r + a
    x = sqrt_k * alpha * target_radius
    jacobi = np.sin(x) / x - np.cos(x)
    derivative_defect = max(abs(1 - alpha * mu), abs(1 - alpha))
    kappa_pop = projection * (derivative_defect + jacobi)
    kappa = projection * derivative_defect
    alpha_exact = p["server_step"] * fraction
    x_exact = sqrt_k * alpha_exact * target_radius
    jacobi_exact = np.sin(x_exact) / x_exact - np.cos(x_exact)
    derivative_defect_exact = max(abs(1 - alpha_exact * mu), abs(1 - alpha_exact))
    kappa_exact_pop = projection * (derivative_defect_exact + jacobi_exact)
    kappa_exact = projection * derivative_defect_exact
    x_one = sqrt_k * 1.0 * target_radius
    jacobi_one = np.sin(x_one) / x_one - np.cos(x_one)
    kappa_one_pop = projection * (max(abs(1 - mu), 0.0) + jacobi_one)
    kappa_one = projection * max(abs(1 - mu), 0.0)
    w_arg = sqrt_k * (r + a)
    mu_w = w_arg / np.tan(w_arg)
    q_sq = max(abs(1 - alpha_exact * mu_w), abs(1 - alpha_exact))
    q_sq1 = max(abs(1 - mu_w), 0.0)
    local_argument = sqrt_k * rho_local
    telescoping = 0.25 + local_argument / np.sin(local_argument)
    drift = p["server_step"] * (
        0.5 * p["eta"] ** 2 * p["local_steps"]
        * (p["local_steps"] - 1) * maximum_anchor_distance
        + telescoping * p["curvature_max"] * rho_local**3
    )
    slacks = {
        "anchor_ball_inside_working_ball": r - a,
        "local_client_clip": p["client_clip"] - rho_local,
        "post_noise_clip_exceeds_client_clip": p["post_noise_clip"] - p["client_clip"],
        "whole_exponential_path_containment": p["declared_radius"] - r - p["server_step"] * p["post_noise_clip"],
        "enlarged_ball_geometry": np.pi - 2 * r_plus,
        "positive_squared_distance_hessian": np.pi / 2 - mu_argument,
        "population_variation_tube": r_plus - r - alpha * target_radius,
        "step_size_upper_limit": 2 - alpha,
        "contraction": 1 - kappa,
        "target_displacement_inside_clip": p["server_step"] * p["post_noise_clip"] - target_radius,
        "population_variation_tube_exact": r_plus - r - alpha_exact * target_radius,
        "contraction_exact": 1 - kappa_exact,
        "population_variation_tube_one_step": r_plus - r - 1.0 * target_radius,
    }
    assert all(value > 0 for value in slacks.values())
    return {
        "local_step_fraction": fraction, "rho_local": rho_local, "r_plus": r_plus,
        "maximum_anchor_distance": maximum_anchor_distance,
        "L": 1.0, "mu": mu, "mu_enlarged_previous": mu_enlarged, "gradient_bound": maximum_anchor_distance,
        "target_radius_bound": target_radius, "initial_error_bound": a,
        "effective_step": alpha, "projection_factor": projection,
        "fixed_base_exponential_factor": 1.0, "jacobi_defect": jacobi,
        "derivative_defect": derivative_defect, "telescoping_factor": telescoping,
        "federated_drift_bound": drift, "kappa": kappa, "kappa_pop_two_point": kappa_pop,
        "kappa_exact_pop_two_point": kappa_exact_pop, "kappa_one_step_pop_two_point": kappa_one_pop,
        "projection_factor_previous_bound": projection_previous_bound,
        "working_ball_distance": r + a, "mu_w": mu_w, "q_squared": q_sq, "q_squared_one_step": q_sq1,
        "effective_step_exact": alpha_exact, "jacobi_defect_exact": jacobi_exact,
        "derivative_defect_exact": derivative_defect_exact, "kappa_exact": kappa_exact,
        "federated_drift_bound_exact": 0.0,
        "jacobi_defect_one_step": jacobi_one, "kappa_one_step": kappa_one,
        "strict_hypothesis_slacks": slacks, "all_strict_hypotheses_pass": True,
    }


def bound_of_schedule(sigmas, cert, exact=False):
    rounds = len(sigmas)
    if rounds == 0:
        return cert["initial_error_bound"], (cert["initial_error_bound"], 0.0, 0.0)
    kappa = cert["kappa_exact"] if exact else cert["kappa"]
    drift = cert["federated_drift_bound_exact"] if exact else cert["federated_drift_bound"]
    weights = kappa ** np.arange(rounds - 1, -1, -1, dtype=float)
    initial = kappa ** rounds * cert["initial_error_bound"]
    drift_part = cert["projection_factor"] * drift * np.sum(weights)
    noise_part = cert["projection_factor"] * np.sqrt(3) * np.sum(weights * sigmas)
    return float(initial + drift_part + noise_part), (float(initial), float(drift_part), float(noise_part))


def squared_bounds(sigmas, cert, q=None, A0=None):
    from scipy.stats import ncx2
    q = cert["q_squared"] if q is None else q
    A0 = cert["initial_error_bound"] if A0 is None else A0
    T = len(sigmas)
    if T == 0:
        return A0, A0
    w2 = q ** (2 * np.arange(T - 1, -1, -1, dtype=float))
    vT = PUBLIC["server_step"] ** 2 * float(np.sum(w2 * np.asarray(sigmas) ** 2))
    rms = np.sqrt(q ** (2 * T) * A0 ** 2 + 3 * vT)
    lam = A0 ** 2 * q ** (2 * T) / vT
    q95 = np.sqrt(vT * ncx2.ppf(0.95, 3, lam))
    cap = cert["target_radius_bound"]
    return float(min(rms, cap)), float(min(q95, cap))


def variance_schedule(rounds, cert):
    q = cert["q_squared"]
    precision = total_precision(PUBLIC["client_clip"])
    H = (1 - q ** rounds) / (1 - q)
    return np.sqrt(H / precision) * q ** (-np.arange(rounds - 1, -1, -1, dtype=float) / 2)


def schedule_and_bound(rounds, weighted, cert, exact=False):
    if rounds == 0:
        return np.array([]), cert["initial_error_bound"]
    precision = total_precision(PUBLIC["client_clip"])
    kappa = cert["kappa_exact"] if exact else cert["kappa"]
    weights = kappa ** np.arange(rounds - 1, -1, -1, dtype=float)
    if weighted:
        sigmas = np.sqrt(np.sum(weights ** (2 / 3)) / precision) * weights ** (-1 / 3)
    else:
        sigmas = np.full(rounds, np.sqrt(rounds / precision))
    bound, _ = bound_of_schedule(sigmas, cert, exact)
    return sigmas, float(bound)


def make_data():
    target = Rotation.from_rotvec([0.0, 0.0, 0.20])
    directions = np.vstack((np.eye(3), np.ones(3) / np.sqrt(3)))
    offsets = 0.04 * np.vstack((directions, -directions))
    anchors = target * Rotation.from_rotvec(offsets)
    assert PUBLIC["clients"] % len(offsets) == 0
    assert np.max(anchors.magnitude()) < PUBLIC["anchor_radius"]
    np.testing.assert_allclose((target.inv() * anchors).as_rotvec().mean(axis=0), 0, atol=1e-15)
    return target, anchors


LAST_ERRORS = [None]


def run_arm(anchors, target, blocks, sigmas, step_fraction, client_clip):
    count = len(blocks)
    theta = Rotation.identity(count)
    anchor_matrices = anchors.as_matrix()
    largest_client_step = 0.0
    largest_endpoint_radius = 0.0
    post_noise_hits = 0
    rounds = len(sigmas)
    for index, sigma in enumerate(sigmas):
        relative = np.einsum("bji,kjl->bkil", theta.as_matrix(), anchor_matrices)
        tangent = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
        tangent = step_fraction * tangent.reshape(count, len(anchor_matrices), 3)
        largest_client_step = max(largest_client_step, float(np.max(np.linalg.norm(tangent, axis=-1))))
        mean = clip_rows(tangent, client_clip).mean(axis=1)
        noisy = mean + sigma * blocks[:, index]
        post_noise_hits += int(np.sum(np.linalg.norm(noisy, axis=1) > PUBLIC["post_noise_clip"]))
        update = clip_rows(noisy, PUBLIC["post_noise_clip"])
        endpoint = theta * Rotation.from_rotvec(PUBLIC["server_step"] * update)
        largest_endpoint_radius = max(largest_endpoint_radius, float(np.max(endpoint.magnitude())))
        theta = Rotation.from_rotvec(clip_rows(endpoint.as_rotvec(), PUBLIC["working_radius"]))
    errors = (target.inv() * theta).magnitude()
    LAST_ERRORS[0] = errors
    return {
        "rounds": rounds,
        "client_contacts": rounds * PUBLIC["clients"],
        "client_clip": client_clip,
        "rms_radians": float(np.sqrt(np.mean(errors**2))),
        "q95_radians": float(np.quantile(errors, 0.95)),
        "largest_client_step": largest_client_step,
        "largest_sampled_endpoint_radius": largest_endpoint_radius,
        "post_noise_clip_fraction": post_noise_hits / (count * rounds) if rounds else 0.0,
        "total_precision": float(np.sum(sigmas**-2)) if rounds else 0.0,
    }


def check_two_step_identity(anchors):
    theta = Rotation.from_rotvec([0.04, -0.03, 0.01])
    eta = PUBLIC["eta"]
    first = theta * Rotation.from_rotvec(eta * (theta.inv() * anchors).as_rotvec())
    second = first * Rotation.from_rotvec(eta * (first.inv() * anchors).as_rotvec())
    collapsed = theta * Rotation.from_rotvec((1 - (1 - eta)**2) * (theta.inv() * anchors).as_rotvec())
    residual = float(np.max((second.inv() * collapsed).magnitude()))
    assert residual < 1e-13
    return residual


def main():
    cert = constants()
    public_bounds = [schedule_and_bound(m, True, cert)[1] for m in range(PUBLIC["maximum_rounds"] + 1)]
    active_rounds = int(np.argmin(public_bounds))
    public_bounds_exact = [schedule_and_bound(m, True, cert, exact=True)[1]
                           for m in range(PUBLIC["maximum_rounds"] + 1)]
    active_rounds_exact = int(np.argmin(public_bounds_exact))
    q = cert["q_squared"]; S_prec = total_precision(PUBLIC["client_clip"]); A0 = cert["initial_error_bound"]
    V_inf = 3 * PUBLIC["server_step"] ** 2 / (S_prec * (1 - q) ** 2)
    T_cont = float(np.log(V_inf / (A0 ** 2 + V_inf)) / np.log(q))
    var_bounds = {m: squared_bounds(variance_schedule(m, cert), cert)[0] if m else A0
                  for m in range(PUBLIC["maximum_rounds"] + 1)}
    horizon_squared = int(min(var_bounds, key=lambda m: (var_bounds[m], m)))
    assert horizon_squared in (int(np.floor(T_cont)), int(np.ceil(T_cont)))
    target, anchors = make_data()
    blocks = np.random.default_rng(SEED).standard_normal((TRIALS, PUBLIC["maximum_rounds"], 3))
    arms = {}
    for label, rounds, weighted, exact in (
        ("retain_initializer", 0, False, False), ("same_update_one_round", 1, False, False),
        ("uniform_5", 5, False, False), ("uniform_20", 20, False, False),
        ("weighted_20", 20, True, False), ("publicly_selected_horizon", active_rounds, True, False),
        ("publicly_selected_horizon_exact", active_rounds_exact, True, True),
        ("variance_20", 20, "variance", True), ("variance_horizon", horizon_squared, "variance", True),
    ):
        if weighted == "variance":
            sigmas = variance_schedule(rounds, cert); bound = bound_of_schedule(sigmas, cert, True)[0]
        else:
            sigmas, bound = schedule_and_bound(rounds, weighted, cert, exact)
        noise = blocks[:, -rounds:] if rounds else blocks[:, :0]
        arm = run_arm(anchors, target, noise, sigmas, cert["local_step_fraction"], PUBLIC["client_clip"])
        arm["certified_rms_upper_bound"] = min(bound, cert["target_radius_bound"])
        arm["schedule_designed_with"] = "exact local step" if exact else "general drift bound"
        b_gen, d_gen = bound_of_schedule(sigmas, cert, exact=False)
        b_ex, d_ex = bound_of_schedule(sigmas, cert, exact=True)
        arm["bound_general_drift"] = min(b_gen, cert["target_radius_bound"])
        arm["bound_exact_step"] = min(b_ex, cert["target_radius_bound"])
        arm["bound_decomposition_general"] = dict(zip(("initial", "drift", "noise"), d_gen))
        arm["bound_decomposition_exact"] = dict(zip(("initial", "drift", "noise"), d_ex))
        arm["noise_scales"] = sigmas.tolist()
        arm["squared_rms_bound"], arm["squared_q95_bound"] = squared_bounds(sigmas, cert)
        if rounds:
            arm["coverage_of_squared_q95_bound"] = float(np.mean(LAST_ERRORS[0] <= arm["squared_q95_bound"]))
        if rounds:
            assert arm["largest_client_step"] <= cert["rho_local"] + 1e-12
            np.testing.assert_allclose(arm["total_precision"], total_precision(PUBLIC["client_clip"]), rtol=1e-12)
        arms[label] = arm

    full_clip = PUBLIC["anchor_radius"]
    full_sigmas = np.array([1 / np.sqrt(total_precision(full_clip))])
    arms["full_step_one_shot"] = run_arm(anchors, target, blocks[:, -1:], full_sigmas, 1.0, full_clip)
    arms["full_step_one_shot"]["noise_scales"] = full_sigmas.tolist()
    one_bound = cert["kappa_one_step"] * cert["initial_error_bound"] \
        + cert["projection_factor"] * np.sqrt(3) * float(full_sigmas[0])
    arms["full_step_one_shot"]["bound_exact_step"] = float(min(one_bound, cert["target_radius_bound"]))
    one_sq = squared_bounds(full_sigmas, cert, q=cert["q_squared_one_step"])
    arms["full_step_one_shot"]["squared_rms_bound"], arms["full_step_one_shot"]["squared_q95_bound"] = one_sq
    arms["full_step_one_shot"]["coverage_of_squared_q95_bound"] = float(np.mean(LAST_ERRORS[0] <= one_sq[1]))
    arms["full_step_one_shot"]["bound_decomposition_exact"] = dict(
        initial=float(cert["kappa_one_step"] * cert["initial_error_bound"]), drift=0.0,
        noise=float(cert["projection_factor"] * np.sqrt(3) * float(full_sigmas[0])))
    from scipy.stats import ncx2
    sqrt_k = np.sqrt(PUBLIC["curvature_max"]); C1 = full_clip; sigma1 = float(full_sigmas[0])
    ell_C = (sqrt_k * C1) / np.sin(sqrt_k * C1)
    c_tel_C = 0.25 + ell_C
    lam_2C = (2 * sqrt_k * C1) / np.tan(2 * sqrt_k * C1)
    D_tan = c_tel_C * PUBLIC["curvature_max"] * C1 ** 3 / lam_2C
    tangent_mean = Rotation.from_rotvec(anchors.as_rotvec().mean(axis=0))
    noiseless_gap = float((tangent_mean.inv() * target).magnitude())
    assert noiseless_gap <= D_tan
    cap = cert["target_radius_bound"]
    lam_nc = (ell_C * D_tan) ** 2 / sigma1 ** 2
    tq95 = float(min(np.sqrt(sigma1 ** 2 * ncx2.ppf(0.95, 3, lam_nc)), cap))
    arms["full_step_one_shot"]["tangent_mean_comparison"] = dict(
        anchor_radius=C1, c_tel=float(c_tel_C), lambda_2C=float(lam_2C), log_lipschitz=float(ell_C),
        noiseless_bound=float(D_tan), noiseless_gap_observed=noiseless_gap,
        additive_rms_bound=float(min(D_tan + cert["projection_factor"] * np.sqrt(3) * sigma1, cap)),
        squared_rms_bound=float(min(np.sqrt((ell_C * D_tan) ** 2 + 3 * sigma1 ** 2), cap)),
        squared_q95_bound=tq95,
        coverage_of_squared_q95_bound=float(np.mean(LAST_ERRORS[0] <= tq95)))
    for arm in arms.values():
        if arm["rounds"]:
            sensitivity = 2 * arm["client_clip"] / PUBLIC["clients"]
            a_rdp = sensitivity**2 * arm["total_precision"] / 2
            epsilon_check = a_rdp + 2 * np.sqrt(a_rdp * np.log(1 / PUBLIC["privacy_delta"]))
            arm["recomputed_epsilon"] = float(epsilon_check)
            assert abs(epsilon_check - PUBLIC["epsilon"]) < 1e-12

    report = {
        "scope": "Analytic public constants and synthetic diagnostics for a heterogeneous SO(3) squared-distance example",
        "public_inputs": PUBLIC, "analytic_constants": cert,
        "admissible_data_class": "Each client's anchor is within public radius 0.25 of identity; the optimizer lies in that convex ball",
        "target_used_only_for_data_generation_and_evaluation": True,
        "target_distance_from_public_initializer": float(target.magnitude()),
        "distinct_anchor_types": len(anchors),
        "clients_per_anchor_type": PUBLIC["clients"] // len(anchors),
        "maximum_anchor_radius": float(np.max(anchors.magnitude())),
        "target_gradient_norm": float(np.linalg.norm((target.inv() * anchors).as_rotvec().mean(axis=0))),
        "two_step_identity_residual": check_two_step_identity(anchors),
        "simulation_seed": SEED, "trials_per_arm": TRIALS,
        "paired_noise_design": "Each arm uses the final active blocks of the same standard Gaussian array",
        "public_active_horizon": active_rounds,
        "public_horizon_bounds": public_bounds,
        "public_active_horizon_exact_step": active_rounds_exact,
        "squared_theorem": {"q": cert["q_squared"], "mu_w": cert["mu_w"], "V_inf": V_inf,
                            "T_cont": T_cont, "horizon": horizon_squared,
                            "radius_condition": [cert["target_radius_bound"], PUBLIC["server_step"] * PUBLIC["post_noise_clip"],
                                                 PUBLIC["declared_radius"] - PUBLIC["working_radius"]]},
        "public_horizon_bounds_exact_step": public_bounds_exact,
        "arms": arms,
        "limits": [
            "One deterministic synthetic data set, with Monte Carlo summaries rather than a distribution over client populations.",
            "The objectives differ by their anchors but share squared-distance structure.",
            "For this objective, local steps follow one geodesic; the calculation is not a test of arbitrary local objectives.",
            "The retained general drift bound is conservative and positive.",
            "Public-horizon selection minimizes a stated upper bound, not the observed error.",
            "The full-step one-shot baseline uses a different valid clip and step size at the same privacy budget.",
        ],
    }
    destination = Path(__file__).resolve().parent.parent / "results/certified_heterogeneous_so3.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"General certificate: kappa={cert['kappa']:.6f}, drift bound={cert['federated_drift_bound']:.7f}, horizon={active_rounds}")
    print(f"Exact-step certificate: alpha={cert['effective_step_exact']:.4f}, kappa={cert['kappa_exact']:.6f}, drift=0, horizon={active_rounds_exact}")
    print(f"One-step (T=1, alpha=1) contraction factor kappa_1={cert['kappa_one_step']:.6f}; initializer error={target.magnitude():.6f}")
    print(f"Squared theorem: mu_w={cert['mu_w']:.10f} q={cert['q_squared']:.10f} q1={cert['q_squared_one_step']:.6f} T_cont={T_cont:.5f} horizon={horizon_squared}")
    for name, arm in arms.items():
        print(f"  {name}: rounds={arm['rounds']} RMS={arm['rms_radians']:.6f} q95={arm['q95_radians']:.6f} sq_rms_bound={arm.get('squared_rms_bound', float('nan')):.6f} sq_q95_bound={arm.get('squared_q95_bound', float('nan')):.6f} coverage={arm.get('coverage_of_squared_q95_bound')}")
    for name, arm in arms.items():
        print(f"{name}: rounds={arm['rounds']}, RMS={arm['rms_radians']:.6f}, bound_general={arm.get('bound_general_drift', float('nan')):.5f}, "
              f"bound_exact={arm.get('bound_exact_step', float('nan')):.5f}, "
              f"decomp_exact={arm.get('bound_decomposition_exact')}, decomp_general={arm.get('bound_decomposition_general')}")


if __name__ == "__main__":
    main()

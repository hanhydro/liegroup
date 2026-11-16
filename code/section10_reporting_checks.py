import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import t as student_t

from capped_schedule import finite_ceiling_schedule

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sources = {}


def source(relative):
    path = ROOT / relative
    if SCRIPT_DIR.name == "code" and relative.startswith("experiments/"):
        path = ROOT / "results" / relative.removeprefix("experiments/")
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def saved(name):
    return np.load(source(f"experiments/{name}.npz"), allow_pickle=False)


def noise_scale(C, N, T, epsilon, delta=1e-5):
    ell = -np.log(delta)
    gap = epsilon / (np.sqrt(ell + epsilon) + np.sqrt(ell))
    return (2 * C / N) * np.sqrt(T / 2) / gap


def paired(a, b):
    difference = a - b
    half = student_t.ppf(0.975, len(a) - 1) * difference.std(ddof=1) / np.sqrt(len(a))
    return {"mean": float(difference.mean()), "half_width_95": float(half)}


def log_slope(x, y):
    x, y = np.log(x), np.log(y)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - slope * x - intercept
    se = np.sqrt(np.sum(residual**2) / (len(x) - 2) / np.sum((x - x.mean())**2))
    half = student_t.ppf(0.975, len(x) - 2) * se
    return {"slope": float(slope), "interval_95": [float(slope - half), float(slope + half)]}


out = {"scope": "Saved-summary and analytic checks, without simulation reruns"}

a = saved("rq1_locality_ablation")
n, z = int(a["mtraj"]), 1.96
p = a["plain_any"]
center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
ci = np.c_[np.maximum(0, center - half), np.minimum(1, center + half)]
np.testing.assert_allclose(ci, a["plain_ci"], atol=1e-15)
np.testing.assert_array_equal(n - np.rint(n * p).astype(int), a["n_noexit"])
np.testing.assert_allclose(noise_scale(0.1, 8, 10, a["eps"]), a["sigma"])
np.testing.assert_allclose(a["sigma"] * np.sqrt(3) / 0.3, a["chi"])
assert np.all(np.abs(a["dmean"][a["eps"] >= 2]) < 0.001)
out["containment"] = {k: a[k].tolist() for k in
                      ["eps", "chi", "plain_any", "plain_ci", "n_noexit", "pdist", "clip_fire"]}

a = json.loads(source("experiments/results.json").read_text())
client_slope = log_slope(np.array(a["so3_Neps"][:4]), np.array(a["so3_floor"][:4]))
dimension_slope = log_slope(np.sqrt(a["dims"]), np.array(a["floors_vs_d"]))
np.testing.assert_allclose(client_slope["slope"], a["so3_slope_vs_Neps"])
np.testing.assert_allclose(dimension_slope["slope"], a["slope_vs_sqrtd"])
out["scaling"] = {
    "client_slope": client_slope, "dimension_slope": dimension_slope,
    "proxy_ratios": (np.array(a["so3_floor"][:4]) / a["so3_bound"][:4]).tolist(),
    "dimension_normalized_rms": (np.array(a["floors_vs_d"]) / np.sqrt(a["dims"])).tolist(),
    "so3_two_rplus": [2 * max(0.6 + rho, 1.5) for rho in [0.58, 0.61]],
}

a = saved("rq3_oneshot_baseline")
it, one = a["priv_iter_C0.2"], a["priv_oneshot_C0.2"]
assert it.shape == one.shape == (5, 16)
assert np.all(one < it)
out["real_task"] = {
    "epsilon": a["eps"].tolist(), "iterative_mean": it.mean(axis=1).tolist(),
    "oneshot_mean": one.mean(axis=1).tolist(),
    "iterative_to_oneshot_ratio": (it.mean(axis=1) / one.mean(axis=1)).tolist(),
    "percent_below_anchor": (100 * (one < a["anchor"]).mean(axis=1)).tolist(),
    "paired_reduction_quartiles": np.quantile(it - one, [0.25, 0.5, 0.75], axis=1).tolist(),
    "secondary_iterative_mean": a["priv_iter_C0.3"].mean(axis=1).tolist(),
    "anchor_mean": float(a["anchor"].mean()),
    "anchor_median_degrees": float(np.degrees(np.median(a["anchor"]))),
    "nonprivate_iterative_mean": float(a["nonpriv_iter_C0.2"].mean()),
    "fixed_denominator_residual_mean": float(a["oneshot_fixeddenom_resid_C020"].mean()),
    "accepted_normalization_residual_mean": float(a["oneshot_acceptednorm_resid_C020"].mean()),
    "primary_plain_noise_margin": 1.1 - 0.2 - 0.2,
    "secondary_plain_noise_margin": 1.1 - 0.2 - 0.3,
}

a = saved("real_ra_results")
out["real_reference"] = {k: a[k].tolist() for k in
                         ["A_mean", "A_fire", "deg_med", "nacc_med", "rej_med", "kappa_med"]}

a = saved("grouped_client_stress")
out["grouped"] = []
for g in a["g"]:
    rms = np.sqrt(np.mean(a[f"errs_g{g}"]**2, axis=1))
    np.testing.assert_allclose(rms, a[f"rms_g{g}"])
    np.testing.assert_allclose(a[f"sigma_g{g}"], g * a["sigma_g1"])
    assert np.all(a[f"client_clipfrac_g{g}"] == 0)
    groups = float(np.median(a[f"Ng_g{g}"]))
    out["grouped"].append({"maximum_group_size": int(g), "median_groups": groups,
                           "median_noise_scale": float(np.median(a[f"sigma_g{g}"])),
                           "mean_rms": float(rms.mean()), "median_contacts": 20 * groups})

a = saved("fixed_size_participation")
out["partial_participation"] = []
for q in a["QS"]:
    closed, empirical = a[f"svclosed_q{q}"], a[f"svemp_q{q}"]
    mean_rel = abs(empirical.mean() / closed.mean() - 1) if closed.mean() else 0
    max_rel = np.max(abs(empirical / closed - 1)) if closed.mean() else 0
    assert mean_rel < 0.002 and max_rel < 0.0191
    out["partial_participation"].append({"fraction": float(q),
                                        "mean_relative_discrepancy": float(mean_rel),
                                        "maximum_relative_discrepancy": float(max_rel)})

from scipy.special import logsumexp
from scipy.stats import binom
for q, entry in zip(a["QS"], out["partial_participation"]):
    worst, count = a[f"sigma_q{q}"], a[f"count_sigma_q{q}"]
    if q == 1.0:
        assert np.array_equal(count, worst)
        entry.update({"median_worst_scale": float(np.median(worst)), "median_count_scale": float(np.median(count)),
                      "median_scale_ratio": 1.0, "mean_rms_worst": float(a[f"rms_q{q}"].mean()),
                      "mean_rms_count": float(a[f"count_rms_q{q}"].mean()),
                      "mean_bound_worst": float(a[f"bound_q{q}"].mean()),
                      "mean_bound_count": float(a[f"count_bound_q{q}"].mean())})
        continue
    assert np.all(count < worst)
    k = np.arange(21)
    largest_eps = 0.0
    for ci in range(len(count)):
        N, m, order = int(a[f"N_q{q}"][ci]), int(a[f"m_q{q}"][ci]), float(a[f"count_order_q{q}"][ci])
        c = 2 * order * 0.20 ** 2 / (m ** 2 * count[ci] ** 2)
        log_mgf = logsumexp(binom.logpmf(k, 20, m / N) + (order - 1) * c * k)
        eps = (log_mgf - np.log(1e-5)) / (order - 1)
        largest_eps = max(largest_eps, eps)
    assert largest_eps <= 4
    entry.update({"median_worst_scale": float(np.median(worst)),
                  "median_count_scale": float(np.median(count)),
                  "median_scale_ratio": float(np.median(count / worst)),
                  "largest_recomputed_epsilon": float(largest_eps),
                  "mean_rms_worst": float(a[f"rms_q{q}"].mean()),
                  "mean_rms_count": float(a[f"count_rms_q{q}"].mean()),
                  "mean_bound_worst": float(a[f"bound_q{q}"].mean()),
                  "mean_bound_count": float(a[f"count_bound_q{q}"].mean())})
pp = {e["fraction"]: e for e in out["partial_participation"]}
participation_table = {
    1.0: (0.0110, 0.0110, 0.0230, 0.0230, 0.0245, 0.0245),
    0.75: (0.0146, 0.0132, 0.0307, 0.0277, 0.0342, 0.0311),
    0.5: (0.0219, 0.0171, 0.0461, 0.0361, 0.0521, 0.0422),
    0.25: (0.0437, 0.0274, 0.0922, 0.0579, 0.1029, 0.0687),
    0.1: (0.1104, 0.0532, 0.1715, 0.1103, 0.2513, 0.1299)}
for q, row in participation_table.items():
    e = pp[q]
    got = (e["median_worst_scale"], e["median_count_scale"], e["mean_rms_worst"], e["mean_rms_count"],
           e["mean_bound_worst"], e["mean_bound_count"])
    assert all(round(g, 4) == v for g, v in zip(got, row)), (q, got)
for q, ratio in [(0.75, 0.90), (0.5, 0.78), (0.25, 0.63), (0.1, 0.48)]:
    assert round(pp[q]["median_scale_ratio"], 2) == ratio
    assert np.all(a[f"count_rms_q{q}"] < a[f"rms_q{q}"])
    assert np.all(a[f"bound_q{q}"] > a[f"rms_q{q}"]) and np.all(a[f"count_bound_q{q}"] > a[f"count_rms_q{q}"])
count_ratios = [pp[q]["mean_bound_count"] / pp[q]["mean_rms_count"] for q in participation_table]
worst_ratios = [pp[q]["mean_bound_worst"] / pp[q]["mean_rms_worst"] for q in participation_table]
assert (round(min(count_ratios), 2), round(max(count_ratios), 2)) == (1.06, 1.19)
assert (round(min(worst_ratios), 2), round(max(worst_ratios), 2)) == (1.06, 1.47)
assert (round(pp[0.25]["mean_rms_worst"], 3), round(pp[0.25]["mean_rms_count"], 3),
        round(pp[0.25]["mean_bound_count"], 3)) == (0.092, 0.058, 0.069)
assert (round(pp[1.0]["mean_rms_worst"], 3), round(pp[0.1]["mean_rms_worst"], 3),
        round(pp[0.1]["mean_rms_count"], 3)) == (0.023, 0.171, 0.110)

a = saved("rq4_schedule_seeds")
out["schedules"] = []
S = 1 / noise_scale(0.4, 200, 1, 1)**2
H = np.hypot(0.4, 0.87)
for T in a["TS"]:
    uniform, cube, capped, skipped = (
        a[f"T{T}_{name}"] for name in ["uniform", "w13", "capped", "skipped"]
    )
    assert len(uniform) == 20
    for target in [cube, capped]:
        difference = paired(uniform, target)
        assert difference["mean"] > difference["half_width_95"]
    schedule = finite_ceiling_schedule(int(T), 0.52, 3, 0.87, 0.4, S, 1e4)
    np.testing.assert_allclose(np.sum(np.array(schedule.sigmas)**-2), S, rtol=1e-12)
    out["schedules"].append({"T": int(T), "uniform_mean": float(uniform.mean()),
                             "cube_root_mean": float(cube.mean()), "capped_mean": float(capped.mean()),
                             "skipped_mean": float(skipped.mean()),
                             "capped_minus_skipped": paired(capped, skipped),
                             "skipped_contacts": 200 * schedule.active_rounds,
                             "capped_contacts": 200 * int(T),
                             "capped_reduction_from_weighted_percent": float(100 * (cube.mean() - capped.mean()) / cube.mean()),
                             "cube_minus_capped": paired(cube, capped),
                             "uniform_minus_cube": paired(uniform, cube),
                             "uniform_minus_capped": paired(uniform, capped),
                             "finite_active_rounds": schedule.active_rounds})


def candidate_cost(m, kappa):
    weights = kappa ** np.arange(63, -1, -1)
    V = np.sum(weights[-m:] ** (2 / 3))
    return H * np.sum(weights[:-m]) + np.sqrt(3 / S) * V**1.5


switch = brentq(lambda k: candidate_cost(11, k) - candidate_cost(12, k), 0.51, 0.525)
cap_a = np.sqrt(2) * 0.4 * np.sqrt(3 * 64) / (200 * H)
out["allocation_design"] = {
    "S": float(S), "Rcap": float(H), "normalized_budget": float(S * H**2 / 3),
    "F11": float(candidate_cost(11, 0.52)), "F12": float(candidate_cost(12, 0.52)),
    "kappa_switch_11_to_12": float(switch),
    "uniform_epsilon_cap_T64": float(cap_a**2 + 2 * cap_a * np.sqrt(np.log(1e5))),
}

a = json.loads(source("results/certified_so3.json").read_text())
c = a["analytic_constants"]
mu = 0.15 / np.tan(0.15)
Lpi = 1.0
defect = np.sin(0.075) / 0.075 - np.cos(0.075)
kappa = Lpi * (1 - 0.5 * mu)
np.testing.assert_allclose(c["kappa_pop"], 1 - 0.5 * mu + defect)
sigma = noise_scale(0.35, 500, 20, 1)
w = kappa ** np.arange(19, -1, -1)
rms_bound = kappa**20 * 0.3 + Lpi * w.sum() * sigma * np.sqrt(3)
tail_bound = rms_bound + Lpi * sigma * np.sqrt(2 * np.log(20) * np.sum(w**2))
np.testing.assert_allclose([mu, Lpi, defect, kappa, sigma],
                           [c[k] for k in ["mu", "L_Pi", "beta_Jac", "kappa", "sigma"]])
np.testing.assert_allclose([rms_bound, tail_bound],
                           [a["bounds"][k] for k in ["expected_recurrence", "perrun_recurrence"]])
mu_w = 0.15 / np.tan(0.15); kw = max(abs(1 - 0.5 * mu_w), 0.5)
vT = sigma**2 * (1 - kw**40) / (1 - kw**2)
sq_rms = np.sqrt(kw**40 * 0.09 + 3 * vT)
from scipy.stats import ncx2
sq_q95 = np.sqrt(vT * ncx2.ppf(0.95, 3, 0.09 * kw**40 / vT))
np.testing.assert_allclose([mu_w, kw, sq_rms, sq_q95],
                           [c["mu_w"], c["q_sq"], a["bounds"]["squared_rms_bound"], a["bounds"]["squared_q95_bound"]])
assert round(kappa, 4) == 0.5038 and round(rms_bound, 3) == 0.107 and round(tail_bound, 3) == 0.194
assert round(sq_rms, 4) == 0.0615 and round(a["empirical"]["rms"], 4) == 0.0612
assert round(sq_q95, 4) == 0.0993 and round(a["empirical"]["tail_q95"], 4) == 0.0993
lo, hi = a["empirical"]["coverage_wilson95"]
assert lo <= 0.95 <= hi
assert all(h["pass"] for h in a["hypotheses"])
out["certified"] = {"recomputed_kappa": float(kappa), "recomputed_sigma": float(sigma),
                     "recomputed_rms_bound": float(rms_bound), "recomputed_tail_bound": float(tail_bound),
                     "recomputed_squared_rms_bound": float(sq_rms), "recomputed_squared_q95_bound": float(sq_q95),
                     "saved_empirical": a["empirical"]}

a = json.loads(source("results/certified_heterogeneous_so3.json").read_text())
c = a["analytic_constants"]
assert c["all_strict_hypotheses_pass"]
assert all(value > 0 for value in c["strict_hypothesis_slacks"].values())
assert a["target_used_only_for_data_generation_and_evaluation"]
assert a["target_gradient_norm"] < 1e-14
assert a["two_step_identity_residual"] < 1e-13
assert a["arms"]["uniform_20"]["certified_rms_upper_bound"] < a["target_distance_from_public_initializer"]
assert a["public_active_horizon"] == int(np.argmin(a["public_horizon_bounds"]))
for arm in a["arms"].values():
    if arm["rounds"]:
        np.testing.assert_allclose(arm["recomputed_epsilon"], a["public_inputs"]["epsilon"], atol=1e-12)
A = a["arms"]; sq = a["squared_theorem"]
mu_w = 0.275 / np.tan(0.275); kw = 1 - 0.19 * mu_w
np.testing.assert_allclose([mu_w, kw], [c["mu_w"], c["q_squared"]])
sig20 = noise_scale(0.11, 2000, 20, 1)
vT = sig20**2 * (1 - kw**40) / (1 - kw**2)
sq20 = np.sqrt(kw**40 * 0.25**2 + 3 * vT)
np.testing.assert_allclose(sq20, A["uniform_20"]["squared_rms_bound"])
S_prec = 20 / sig20**2
V_inf = 3 / (S_prec * (1 - kw)**2)
T_cont = np.log(V_inf / (0.25**2 + V_inf)) / np.log(kw)
assert sq["horizon"] == 38 and abs(T_cont - sq["T_cont"]) < 1e-9
assert round(sq20, 4) == 0.0083 and round(A["uniform_20"]["rms_radians"], 4) == 0.0077
assert round(A["uniform_20"]["squared_q95_bound"], 4) == 0.0133
assert round(A["variance_horizon"]["squared_rms_bound"], 4) == 0.0050 and round(A["variance_horizon"]["rms_radians"], 4) == 0.0049
assert round(A["full_step_one_shot"]["squared_rms_bound"], 4) == 0.0067 and round(A["full_step_one_shot"]["rms_radians"], 4) == 0.0021
assert round(A["uniform_20"]["bound_exact_step"], 4) == 0.0263 and round(A["uniform_20"]["bound_general_drift"], 4) == 0.0895
assert round(A["full_step_one_shot"]["bound_exact_step"], 4) == 0.0085
assert round(A["publicly_selected_horizon_exact"]["bound_exact_step"], 4) == 0.0204
assert round(100 * (1 - A["full_step_one_shot"]["bound_exact_step"] / A["publicly_selected_horizon_exact"]["bound_exact_step"])) == 59
assert round(c["kappa"], 6) == 0.805067 and round(c["kappa_exact"], 6) == 0.814814 and round(c["kappa_one_step"], 4) == 0.0253
assert round(c["kappa_pop_two_point"], 6) == 0.806075 and round(c["kappa_exact_pop_two_point"], 6) == 0.815724
assert A["weighted_20"]["schedule_designed_with"] == "general drift bound"
tm = A["full_step_one_shot"]["tangent_mean_comparison"]
assert (round(tm["c_tel"], 4), round(tm["lambda_2C"], 4), round(tm["log_lipschitz"], 4)) == (1.2526, 0.9791, 1.0026)
assert round(tm["noiseless_bound"], 4) == 0.0050 and round(tm["noiseless_gap_observed"] * 1e5, 1) == 1.8
assert (round(tm["additive_rms_bound"], 4), round(tm["squared_rms_bound"], 4), round(tm["squared_q95_bound"], 4)) \
    == (0.0071, 0.0054, 0.0073)
assert tm["coverage_of_squared_q95_bound"] == 1.0
assert round(100 * (1 - tm["additive_rms_bound"] / A["publicly_selected_horizon_exact"]["bound_exact_step"])) == 65
assert round(100 * (1 - A["variance_horizon"]["squared_rms_bound"] / tm["squared_rms_bound"])) == 7
out["tangent_mean_one_shot"] = tm
assert A["full_step_one_shot"]["bound_exact_step"] < A["publicly_selected_horizon_exact"]["bound_exact_step"]
assert A["variance_horizon"]["squared_rms_bound"] < A["full_step_one_shot"]["squared_rms_bound"]
assert all(arm["coverage_of_squared_q95_bound"] >= 0.95 for arm in A.values()
           if arm.get("coverage_of_squared_q95_bound") is not None)
assert a["public_active_horizon_exact_step"] == 37 and a["public_active_horizon"] == 32
ph = json.loads(source("results/certified_pseudo_huber_so3.json").read_text())
assert round(ph["q"], 3) == 0.915 and round(ph["squared_rms_bound"], 4) == 0.0096
assert round(ph["observed_rms"], 4) == 0.0022 and ph["coverage_of_squared_q95_bound"] >= 0.95
assert round(ph["kappa_theorem_7_1"], 4) == 0.9147 and round(ph["theorem_7_1_rms_bound"], 4) == 0.0269
assert round(ph["kappa_enlarged_previous"], 4) == 0.9962
cb = json.loads(source("results/camera_public_bound.json").read_text())
c75, c80 = cb["classes"]["0.75"], cb["classes"]["0.80"]
assert c75["cameras_in_class"] == 16 and c80["cameras_in_class"] == 15 and c80["excluded"] == [1457]
assert abs(cb["min_accepted_fraction"] - 168 / 212) < 1e-12
mu75 = 0.75 * 0.2 / np.tan(0.2)
np.testing.assert_allclose([mu75, 1 - 0.5 * mu75], [c75["constants"]["mu"], c75["constants"]["kappa_w"]])
r16 = [r for r in c75["by_budget"] if r["eps"] == 16.0][0]
assert round(r16["squared_bound"], 4) == 0.0072 and round(r16["observed_iterative"], 4) == 0.0068
assert round(r16["thm71_bound"], 4) == 0.0151 and round(r16["ratio_squared"], 2) == 1.05
np.testing.assert_allclose(c75["constants"]["kappa"], c75["constants"]["kappa_w"])
assert all(1.0 < r["ratio_squared"] < 1.08 for r in c75["by_budget"])
assert max(r["max_standardized_excess"] for r in c75["by_budget"]) < 1.0
out["camera_public_bound"] = {"squared_bound_eps16": r16["squared_bound"], "observed_eps16": r16["observed_iterative"],
                              "thm71_bound_eps16": r16["thm71_bound"]}
out["squared_theorem"] = {"heterogeneous_uniform_20_bound": float(sq20), "T_cont": float(T_cont),
                          "pseudo_huber_bound": ph["squared_rms_bound"]}
out["certified_heterogeneous"] = {
    "public_kappa": c["kappa"], "public_drift_bound": c["federated_drift_bound"],
    "public_active_horizon": a["public_active_horizon"],
    "initializer_error": a["target_distance_from_public_initializer"],
    "arms": {name: {key: arm[key] for key in ("rounds", "rms_radians", "client_contacts")}
             for name, arm in a["arms"].items()},
    "uniform_20_public_rms_bound": a["arms"]["uniform_20"]["certified_rms_upper_bound"],
}
out["source_sha256"] = sources
out["all_checks_passed"] = True

a = saved("robust_camera_task")
b = json.loads(source("experiments/robust_camera_task.json").read_text())
rob = {}
for di, row in enumerate(b["per_degree"]):
    rob[f"deg{row['degrees']:g}_init"] = (float(a["init_err"][di].mean()), row["init_err_mean"])
    for ei, rec in enumerate(row["eps"]):
        rob[f"deg{row['degrees']:g}_eps{rec['eps']:g}_iter"] = (float(a["priv_iter"][di, ei].mean()), rec["iter_mean"])
        rob[f"deg{row['degrees']:g}_eps{rec['eps']:g}_oneshot"] = (float(a["priv_oneshot"][di, ei].mean()), rec["oneshot_mean"])
        rob[f"deg{row['degrees']:g}_eps{rec['eps']:g}_ambient"] = (float(a["priv_ambient"][di, ei].mean()), rec["ambient_mean"])
        rob[f"deg{row['degrees']:g}_eps{rec['eps']:g}_sqoneshot"] = (float(a["priv_sqoneshot"][di, ei].mean()), rec["sqoneshot_mean"])
rob_pass = all(abs(x - y) <= 1e-9 for x, y in rob.values())
it10 = float(a["priv_iter"][2, 3].mean()); os10 = float(a["priv_oneshot"][2, 3].mean()); in10 = float(a["init_err"][2].mean())
sq10 = float(a["priv_sqoneshot"][2, 3].mean())
rob_pass &= abs(it10 - 0.0032) < 5e-5 and abs(sq10 - 0.0116) < 5e-5 and round(in10 / it10) == 56 \
    and round(sq10 / it10, 1) == 3.6 and round(in10 / sq10) == 15 and abs(os10 - 0.1001) < 5e-5
amb_ratio = a["priv_ambient"].mean(axis=-1) / a["priv_iter"].mean(axis=-1)
rob_pass &= round(float(amb_ratio.min()), 2) == 0.98 and round(float(amb_ratio.max()), 2) == 1.07 \
    and round(float(amb_ratio.mean()), 2) == 1.02 and amb_ratio.size == 15
out["robust_camera_task"] = {"pass": bool(rob_pass), "iter_10deg_eps8": it10, "oneshot_10deg_eps8": os10,
                             "tangent_mean_oneshot_10deg_eps8": sq10,
                             "init_10deg": in10, "n_values_compared": len(rob),
                             "ambient_over_iter_min": float(amb_ratio.min()),
                             "ambient_over_iter_max": float(amb_ratio.max()),
                             "ambient_over_iter_mean": float(amb_ratio.mean())}
out["all_checks_passed"] = bool(out["all_checks_passed"]) and bool(rob_pass)

a = saved("rq4_short_horizon")
b = json.loads(source("experiments/rq4_short_horizon.json").read_text())
dsg = b["designs"]
pooled = float(np.mean(a["one_step"] ** 2))
z_one = (pooled - dsg["one_step"]["tangent_squared"] ** 2) / (np.std(a["one_step"] ** 2, ddof=1) / np.sqrt(len(a["one_step"])))
short_pass = (len(a["retain"]) == 20 and round(float(a["retain"].mean()), 4) == 0.0055
              and round(float(a["one_step"].mean()), 4) == 0.0129
              and round(dsg["one_step"]["tangent_squared"], 4) == 0.0128
              and round(dsg["half_3"]["squared"], 3) == 0.065 and round(dsg["half_8"]["squared"], 3) == 0.071
              and min((k for k in dsg if k.startswith("half_")), key=lambda k: dsg[k]["squared"]) == "half_3"
              and round(float(a["half_1"].mean()), 3) == 0.034 and round(z_one, 1) == 0.5
              and b["public_pick_terminal"] == b["public_pick_squared"] == b["public_pick_tangent_squared"] == "one_step"
              and b["observed_best"] == "retain")
d8 = b["half_8_minus_w13_8"]
short_pass &= round(d8[0], 4) == -0.0058 and round(d8[1], 4) == 0.0042
out["allocation_short_designs"] = {"pass": bool(short_pass), "one_step_pooled_excess_se": float(z_one)}

a = saved("robust_onestep_sweep")
priv, nl = a["private"].mean(axis=-1), a["noiseless"].mean(axis=-1)
it = saved("robust_camera_task")["priv_iter"]
best_alpha = [float(a["alphas"][int(np.argmin(nl[k]))]) for k in range(3)]
sweep_pass = best_alpha == [1.5, 1.5, 2.0] and [float(a["alphas"][int(np.argmin(priv[k, :, 3]))]) for k in range(3)] == best_alpha
sweep_pass &= (round(float(priv[2, 2, 3]), 4) == 0.0228 and round(float(nl[2, 2]), 4) == 0.0227
               and round(float(nl[0, 1]), 4) == 0.0014 and round(float(nl[1, 1]), 4) == 0.0056
               and round(float(priv[0, 1, 3]), 4) == 0.0022 and round(float(priv[1, 1, 3]), 4) == 0.0059)
best_by_eps = [[float(a["alphas"][int(np.argmin(priv[k, :, e]))]) for e in range(5)] for k in range(3)]
sweep_pass &= best_by_eps == [[1.0, 1.0, 1.0, 1.5, 1.5], [1.5] * 5, [2.0] * 5]
counts5 = [int(np.sum(it[1, e] < a["private"][1, 1, e])) for e in range(5)]
sweep_pass &= all(it[1, e].mean() < priv[1, 1, e] for e in (2, 3, 4)) and all(it[1, e].mean() > priv[1, 1, e] for e in (0, 1))
sweep_pass &= counts5 == [0, 1, 8, 14, 16]
sweep_pass &= all(it[2, e].mean() < priv[2, :, e].min() for e in range(5))
sweep_pass &= round(float(priv[2, 2, 3] / it[2, 3].mean())) == 7
out["robust_onestep_sweep"] = {"pass": bool(sweep_pass), "best_alpha_by_turn": best_alpha,
                               "iter_below_alpha1.5_5deg_cams": counts5}

r = json.loads(source("experiments/target_residuals.json").read_text())
grads = [r["declared_domain"]["grad_norm"], r["allocation"]["grad_norm_max"],
         r["camera_squared"]["grad_norm_max"], r["camera_pseudo_huber"]["grad_norm_max"]]
shifts = [abs(r["declared_domain"]["refinement_shift"]), r["allocation"]["refinement_shift_max"],
          abs(r["camera_squared"]["refinement_shift_max"]), r["camera_pseudo_huber"]["refinement_shift_max"]]
ph = r["camera_pseudo_huber"]
resid_pass = (max(grads) <= 1.5e-16 and max(shifts) < 2e-15
              and [round(x, 3) for x in ph["target_to_center_max_by_turn"]] == [0.037, 0.103, 0.189]
              and all(x < y for x, y in zip(ph["target_to_center_max_by_turn"], ph["working_radius_by_turn"])))
out["target_residuals"] = {"pass": bool(resid_pass), "grad_norm_max": float(max(grads)), "shift_max": float(max(shifts))}
cb = json.loads(source("results/camera_public_bound.json").read_text())["classes"]["0.75"]["by_budget"]
pick_sq = ["iter" if row["squared_bound"] < row["oneshot_squared_bound"] else "one" for row in cb]
pick_71 = ["iter" if row["thm71_bound"] < row["oneshot_thm71_bound"] else "one" for row in cb]
cam_pass = pick_sq == ["one", "iter", "iter", "iter", "iter"] and pick_71 == ["one", "one", "iter", "iter", "iter"]
cam_pass &= (round(cb[0]["oneshot_squared_bound"], 4) == 0.0544 and round(cb[0]["observed_oneshot"], 4) == 0.0161
             and round(cb[3]["squared_bound"], 4) == 0.0130 and round(cb[3]["observed_iterative"], 4) == 0.0122
             and round(cb[3]["observed_oneshot"], 4) == 0.0026)
out["camera_design_selection"] = {"pass": bool(cam_pass), "squared_pick": pick_sq, "thm71_pick": pick_71}
_rq1 = SCRIPT_DIR / "rq1_locality_ablation.py"
_src = _rq1.read_text()
sources["experiments/rq1_locality_ablation.py"] = hashlib.sha256(_src.encode()).hexdigest()
_ns = {"__name__": "anchors_only"}
exec(compile(_src[:_src.index("def sigma_rdp")], "rq1_head", "exec"), _ns)
init_err = float(_ns["GRP"].dist(_ns["CENTER"], _ns["THETA_STAR"]))
clip_rms = np.load(source("experiments/rq1_locality_ablation.npz"))["clip_rms"]
init_pass = round(init_err, 3) == 0.021 and bool(np.all(init_err < clip_rms))
out["containment_initializer"] = {"pass": init_pass, "initializer_error": init_err,
                                  "clipped_rms": np.asarray(clip_rms).tolist()}
out["all_checks_passed"] = bool(out["all_checks_passed"]) and short_pass and sweep_pass and resid_pass and cam_pass and init_pass
assert out["all_checks_passed"], {k: v for k, v in out.items() if isinstance(v, dict) and v.get("pass") is False}

destination = ROOT / "results" / "section10_reporting_checks.json"
destination.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
print(f"All Section 10 reporting checks passed. Evidence saved to {destination.relative_to(ROOT)}.")

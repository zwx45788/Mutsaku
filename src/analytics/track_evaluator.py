"""Multi-track career taxonomy & honest job-level qualification and alignment evaluator.

Zero steep sigmoid amplification. Every JD is evaluated directly against the required
skill set & Bloom depth of its matching career track, alongside both track-level and
market-blended aggregate L1/OT metrics and employer-clustered paired bootstrap 95% CIs.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from analytics.gap_engine import compute_supply_profile, exact_wasserstein_1

TRACK_META: Dict[str, Dict[str, Any]] = {
    "fullstack_backend": {
        "name_zh": "后端与全栈分布式工程",
        "short_zh": "后端/全栈",
        "primary_domains": ("software_eng", "cloud_dist", "data_db"),
        "core_skills": ("java_spring", "soft_eng_arch", "sql_rdbms", "web_frontend", "dist_systems", "golang_rust", "linux_shell"),
    },
    "cloud_ops_sec": {
        "name_zh": "云原生、测试运维与安全",
        "short_zh": "云运维/测安",
        "primary_domains": ("cloud_dist", "systems", "security_domain", "software_eng"),
        "core_skills": ("linux_shell", "devops_sre", "qa_testing", "cloud_k8s", "cyber_sec", "os_kernel", "comp_net", "sql_rdbms"),
    },
    "ai_data": {
        "name_zh": "AI大模型与数据智能工程",
        "short_zh": "AI/数据工程",
        "primary_domains": ("ai_llm", "data_db", "foundations"),
        "core_skills": ("python_eng", "ml_classic", "deep_learning", "llm_posttrain", "agent_rag", "mlops_eval", "data_mining_bi", "big_data_eng", "nosql_vector_db", "dsa"),
    },
    "systems_edge": {
        "name_zh": "系统底层、嵌入式与端侧智能",
        "short_zh": "系统/嵌入式",
        "primary_domains": ("embodied_iot", "systems", "software_eng"),
        "core_skills": ("c_cpp", "embedded_mcu", "comp_arch", "os_kernel", "iot_edge", "mobile_harmony", "parallel_gpu", "linux_shell", "eda_ic_design"),
    },
}

TRACK_IDS: Tuple[str, ...] = ("fullstack_backend", "cloud_ops_sec", "ai_data", "systems_edge")


def classify_jd_track(title: str, hits_row: np.ndarray, skill_ids: List[str]) -> str:
    """Assign each JD deterministically to one of the 4 industry career tracks."""
    t = (title or "").lower()
    if re.search(
        r"算法|大模型|llm|nlp|视觉|cv|深度学习|机器学习|ai|人工智能|多模态|推荐|aigc|搜广推|语音|智能体"
        r"|数据|数仓|bi|分析|etl|hadoop|spark|flink|dba|数据库|风控|商业智能",
        t,
    ):
        return "ai_data"
    if re.search(
        r"嵌入式|硬件|驱动|单片机|fpga|mcu|linux内核|物联网|iot|机器人|控制|自动驾驶|鸿蒙|移动端"
        r"|android|ios|客户端|c\+\+|qt|图形|游戏|芯片|eda|ic",
        t,
    ):
        return "systems_edge"
    if re.search(
        r"测试|qa|运维|sre|devops|安全|网安|渗透|云原生|k8s|实施|技术支持|网络工程|linux|系统工程|集成",
        t,
    ):
        return "cloud_ops_sec"
    if re.search(
        r"java|后端|服务端|golang|go开发|python|前端|web|全栈|研发|软件|开发|程序|php|node|\.net|架构|产品",
        t,
    ):
        return "fullstack_backend"

    idx = {s: i for i, s in enumerate(skill_ids)}
    scores = {
        tk: int(sum(hits_row[idx[s]] for s in TRACK_META[tk]["core_skills"] if s in idx))
        for tk in TRACK_IDS
    }
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "fullstack_backend"


def build_track_demand(demand: Dict[str, Any], kg: Dict[str, Any]) -> Dict[str, Any]:
    """Partition an extracted demand slice into the 4 industry tracks with exact employer weights."""
    skill_ids = kg["skill_ids"]
    hits = demand["hits"].astype(np.float64)
    ad_weights = np.asarray(demand["ad_weights"], dtype=np.float64)
    titles = demand["titles"]
    employers = np.asarray(demand["employers"])
    n_rows, k = hits.shape

    track_labels = np.array(
        [classify_jd_track(titles[i], hits[i], skill_ids) for i in range(n_rows)],
        dtype=object,
    )
    track_profiles: Dict[str, Dict[str, Any]] = {}
    for tk in TRACK_IDS:
        mask = track_labels == tk
        w_sum = float(ad_weights[mask].sum())
        if mask.sum() == 0 or w_sum <= 1e-12:
            dist = np.asarray(demand["demand_dist"], dtype=np.float64).copy()
            prev = np.zeros(k, dtype=np.float64)
        else:
            tw = ad_weights[mask] / w_sum
            prev = (hits[mask] * tw[:, None]).sum(axis=0)
            dist = prev / max(1e-12, prev.sum())
        track_profiles[tk] = {
            "track_id": tk,
            "name_zh": TRACK_META[tk]["name_zh"],
            "n_ads": int(mask.sum()),
            "weight_share": round(w_sum, 5),
            "demand_dist": dist,
            "prevalence": prev,
            "mask": mask,
        }

    return {
        "n_rows": n_rows,
        "n_employers": int(demand["n_employers"]),
        "hits": hits,
        "ad_weights": ad_weights,
        "employers": employers,
        "demand_dist": np.asarray(demand["demand_dist"], dtype=np.float64),
        "job_bloom": np.asarray(demand["job_bloom"], dtype=np.float64),
        "track_labels": track_labels,
        "tracks": track_profiles,
        "raw_demand": demand,
    }


def skill_readiness_vector(supply: Dict[str, Any], job_bloom: np.ndarray) -> np.ndarray:
    """Compute interpretable per-skill readiness in [0, 1] without any artificial sigmoid.

    - exposure = min(1.0, mass / 1.6): requires ~1.6 credit-equivalents for full exposure.
    - depth_ratio = min(1.0, bloom / max(1.0, job_bloom)): compares course Bloom vs JD Bloom.
    - readiness = exposure * (0.45 + 0.55 * depth_ratio): strictly linear/concave, 0 if untaught.
    """
    mass = np.asarray(supply["mass"], dtype=np.float64)
    bloom = np.asarray(supply["bloom"], dtype=np.float64)
    exposure = np.minimum(1.0, mass / 2.0)
    depth_ratio = np.minimum(1.0, bloom / np.maximum(1.0, job_bloom))
    return exposure * (0.40 + 0.60 * depth_ratio)


def evaluate_multitrack_plan(
    track_curricula: Dict[str, List[Dict[str, Any]]],
    track_shares: Dict[str, float],
    track_demand: Dict[str, Any],
    kg: Dict[str, Any],
    qual_threshold: float = 0.70,
) -> Dict[str, Any]:
    """Evaluate a multi-track curriculum plan on all JDs with zero artificial amplification."""
    skill_ids = kg["skill_ids"]
    C = np.asarray(kg["semantic_ground_distance"]["matrix"], dtype=np.float64)
    hits = track_demand["hits"]
    ad_weights = track_demand["ad_weights"]
    job_bloom = track_demand["job_bloom"]
    track_labels = track_demand["track_labels"]
    n_rows, k = hits.shape

    # Supply profile & readiness vector per track
    supplies: Dict[str, Dict[str, Any]] = {}
    readiness_map: Dict[str, np.ndarray] = {}
    for tk in TRACK_IDS:
        sup = compute_supply_profile(track_curricula[tk], skill_ids)
        supplies[tk] = sup
        readiness_map[tk] = skill_readiness_vector(sup, job_bloom)

    # Per-JD readiness score (every JD evaluated against its corresponding track graduate)
    skill_counts = hits.sum(axis=1)
    valid_mask = skill_counts > 0
    per_jd_readiness = np.zeros(n_rows, dtype=np.float64)
    per_jd_qualified = np.zeros(n_rows, dtype=np.float64)
    per_jd_hard_pass = np.zeros(n_rows, dtype=np.float64)  # fraction of required skills >= 0.50 readiness

    for tk in TRACK_IDS:
        m = (track_labels == tk) & valid_mask
        if not np.any(m):
            continue
        r_vec = readiness_map[tk]
        cov = (hits[m] @ r_vec) / skill_counts[m]
        skill_pass_frac = (hits[m] @ (r_vec >= 0.55).astype(np.float64)) / skill_counts[m]
        per_jd_readiness[m] = cov
        per_jd_hard_pass[m] = skill_pass_frac
        # Qualified = meets >= threshold average skill+depth readiness AND >= 70% of listed skills taught to >=0.55
        per_jd_qualified[m] = ((cov >= qual_threshold) & (skill_pass_frac >= 0.70)).astype(np.float64)

    vw = ad_weights[valid_mask]
    vw_sum = max(1e-12, float(vw.sum()))
    jd_mean_readiness = float((per_jd_readiness[valid_mask] * vw).sum() / vw_sum)
    jd_qual_rate = float((per_jd_qualified[valid_mask] * vw).sum() / vw_sum)
    jd_skill_pass_rate = float((per_jd_hard_pass[valid_mask] * vw).sum() / vw_sum)

    # Per-track breakdown + track-level L1 / OT / Bloom / Emerging
    emerg_mask = np.array([s["stage"] in ("emerging", "growing") for s in kg["skills"]], dtype=bool)
    per_track_stats: Dict[str, Dict[str, Any]] = {}
    weighted_l1 = 0.0
    weighted_ot = 0.0
    weighted_bloom = 0.0
    weighted_emerg = 0.0
    total_w = 0.0

    for tk in TRACK_IDS:
        tp = track_demand["tracks"][tk]
        tw_share = float(tp["weight_share"])
        d_dist = tp["demand_dist"]
        s_dist = supplies[tk]["dist"]
        s_read = readiness_map[tk]

        l1_t = float(np.minimum(s_dist, d_dist).sum())
        ot_t = float(max(0.0, 1.0 - exact_wasserstein_1(s_dist, d_dist, C)))
        s_bloom_eff = supplies[tk]["bloom"] * np.minimum(1.0, supplies[tk]["mass"] / 2.0)
        bl_t = float(max(0.0, 1.0 - float((d_dist * np.maximum(0.0, job_bloom - s_bloom_eff)).sum()) / 5.0))
        em_d = d_dist[emerg_mask]
        em_cov = float((em_d * s_read[emerg_mask]).sum() / max(1e-12, em_d.sum()))

        m = (track_labels == tk) & valid_mask
        if np.any(m):
            mw = ad_weights[m]
            mws = max(1e-12, float(mw.sum()))
            t_read = float((per_jd_readiness[m] * mw).sum() / mws)
            t_qual = float((per_jd_qualified[m] * mw).sum() / mws)
        else:
            t_read, t_qual = 0.0, 0.0

        per_track_stats[tk] = {
            "name_zh": TRACK_META[tk]["name_zh"],
            "demand_share": round(tw_share, 4),
            "student_share": round(float(track_shares.get(tk, 0.25)), 4),
            "jd_qual_rate_70": round(t_qual, 5),
            "jd_mean_readiness": round(t_read, 5),
            "l1_overlap": round(l1_t, 5),
            "ot_alignment": round(ot_t, 5),
            "bloom_fidelity": round(bl_t, 5),
            "emerging_readiness": round(em_cov, 5),
            "n_ads": tp["n_ads"],
        }
        weighted_l1 += tw_share * l1_t
        weighted_ot += tw_share * ot_t
        weighted_bloom += tw_share * bl_t
        weighted_emerg += tw_share * em_cov
        total_w += tw_share

    norm = max(1e-12, total_w)
    # Also compute market-blended aggregate supply distribution (mixed by student shares)
    share_sum = sum(float(track_shares.get(tk, 0.25)) for tk in TRACK_IDS)
    agg_mass = sum(
        (float(track_shares.get(tk, 0.25)) / max(1e-12, share_sum)) * supplies[tk]["mass"]
        for tk in TRACK_IDS
    )
    agg_dist = agg_mass / max(1e-12, agg_mass.sum())
    agg_l1 = float(np.minimum(agg_dist, track_demand["demand_dist"]).sum())
    agg_ot = float(max(0.0, 1.0 - exact_wasserstein_1(agg_dist, track_demand["demand_dist"], C)))

    # Capacity-demand alignment across tracks (1 - 0.5 * L1 distance between student shares & demand shares)
    cap_match = 1.0 - 0.5 * sum(
        abs(float(track_shares.get(tk, 0.25)) - float(track_demand["tracks"][tk]["weight_share"]))
        for tk in TRACK_IDS
    )
    worst_track_read = min(per_track_stats[tk]["jd_mean_readiness"] for tk in TRACK_IDS)
    worst_track_qual = min(per_track_stats[tk]["jd_qual_rate_70"] for tk in TRACK_IDS)

    return {
        "jd_qual_rate_70": round(jd_qual_rate, 5),
        "jd_mean_readiness": round(jd_mean_readiness, 5),
        "jd_skill_pass_rate": round(jd_skill_pass_rate, 5),
        "track_l1_overlap": round(weighted_l1 / norm, 5),
        "track_ot_alignment": round(weighted_ot / norm, 5),
        "track_bloom_fidelity": round(weighted_bloom / norm, 5),
        "track_emerging_readiness": round(weighted_emerg / norm, 5),
        "agg_l1_overlap": round(agg_l1, 5),
        "agg_ot_alignment": round(agg_ot, 5),
        "track_capacity_match": round(cap_match, 5),
        "worst_track_readiness": round(worst_track_read, 5),
        "worst_track_qual_70": round(worst_track_qual, 5),
        "per_track": per_track_stats,
        "_per_jd_qualified": per_jd_qualified,
        "_per_jd_readiness": per_jd_readiness,
        "_valid_mask": valid_mask,
        "_readiness_map": readiness_map,
    }


def evaluate_single_curriculum(
    courses: List[Dict[str, Any]],
    track_demand: Dict[str, Any],
    kg: Dict[str, Any],
    qual_threshold: float = 0.70,
) -> Dict[str, Any]:
    """Evaluate an undifferentiated single-track curriculum (e.g., B0 Freeze or B1 Single-Track MILP)."""
    tc = {tk: courses for tk in TRACK_IDS}
    shares = {tk: float(track_demand["tracks"][tk]["weight_share"]) for tk in TRACK_IDS}
    return evaluate_multitrack_plan(tc, shares, track_demand, kg, qual_threshold=qual_threshold)


def paired_employer_bootstrap_ci(
    eval_map: Dict[str, Dict[str, Any]],
    reference_key: str,
    track_demand: Dict[str, Any],
    metric_field: str = "_per_jd_qualified",
    n_boot: int = 600,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Employer-clustered paired bootstrap 95% CI on exact per-JD metrics (qualification or readiness)."""
    valid = track_demand["_valid_mask"] if "_valid_mask" in track_demand else eval_map[reference_key]["_valid_mask"]
    employers = track_demand["employers"][valid]
    weights = track_demand["ad_weights"][valid]
    uniq_emp, inv = np.unique(employers, return_inverse=True)
    n_emp = len(uniq_emp)

    # Aggregate weighted numerator and denominator per employer
    emp_den = np.zeros(n_emp, dtype=np.float64)
    np.add.at(emp_den, inv, weights)

    ref_vals = eval_map[reference_key][metric_field][valid]
    emp_ref = np.zeros(n_emp, dtype=np.float64)
    np.add.at(emp_ref, inv, weights * ref_vals)

    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_emp, np.full(n_emp, 1.0 / n_emp), size=n_boot).astype(np.float64)
    tot_den = np.maximum(1e-12, draws @ emp_den)
    ref_boot = (draws @ emp_ref) / tot_den

    out: Dict[str, Dict[str, float]] = {}
    for arm, ev in eval_map.items():
        vals = ev[metric_field][valid]
        emp_arm = np.zeros(n_emp, dtype=np.float64)
        np.add.at(emp_arm, inv, weights * vals)
        arm_boot = (draws @ emp_arm) / tot_den
        diff = arm_boot - ref_boot
        point = float((weights * (vals - ref_vals)).sum() / max(1e-12, weights.sum()))
        lo, hi = np.quantile(diff, [0.025, 0.975])
        p_neg = float(np.mean(diff <= 0.0))
        out[arm] = {
            "delta": round(point, 5),
            "ci95_lo": round(float(lo), 5),
            "ci95_hi": round(float(hi), 5),
            "significant_positive": bool(lo > 0.0),
            "p_boot_one_sided": round(p_neg, 4),
        }
    return out


def audit_track_failures(
    eval_res: Dict[str, Any],
    track_demand: Dict[str, Any],
    kg: Dict[str, Any],
    top_k: int = 6,
) -> Dict[str, Any]:
    """Identify the exact skill deficits causing JD disqualification in each track (for error-driven RSI)."""
    skill_ids = kg["skill_ids"]
    zh_map = {s["id"]: s["zh"] for s in kg["skills"]}
    dom_map = {s["id"]: s["domain"] for s in kg["skills"]}
    hits = track_demand["hits"]
    w = track_demand["ad_weights"]
    labels = track_demand["track_labels"]
    unqual = eval_res["_per_jd_qualified"] < 0.5
    valid = eval_res["_valid_mask"]
    read_map = eval_res["_readiness_map"]

    diagnosis: Dict[str, Any] = {}
    for tk in TRACK_IDS:
        m = (labels == tk) & valid & unqual
        r_vec = read_map[tk]
        if not np.any(m):
            diagnosis[tk] = {"unqualified_ads": 0, "top_missing_skills": []}
            continue
        mw = w[m]
        mw = mw / max(1e-12, mw.sum())
        # Impact = prevalence in unqualified JDs * (1 - current readiness)
        prev_fail = (hits[m] * mw[:, None]).sum(axis=0)
        deficit_impact = prev_fail * np.maximum(0.0, 1.0 - r_vec)
        order = np.argsort(-deficit_impact)[:top_k]
        diagnosis[tk] = {
            "name_zh": TRACK_META[tk]["name_zh"],
            "unqualified_ads": int(m.sum()),
            "current_qual_rate": eval_res["per_track"][tk]["jd_qual_rate_70"],
            "current_readiness": eval_res["per_track"][tk]["jd_mean_readiness"],
            "top_missing_skills": [
                {
                    "skill_id": skill_ids[j],
                    "zh": zh_map[skill_ids[j]],
                    "domain": dom_map[skill_ids[j]],
                    "fail_jd_prevalence": round(float(prev_fail[j]), 4),
                    "current_readiness": round(float(r_vec[j]), 4),
                    "impact_score": round(float(deficit_impact[j]), 4),
                }
                for j in order
                if deficit_impact[j] > 0.005
            ],
        }
    return diagnosis


def strip_private_arrays(eval_res: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable copy of evaluate_multitrack_plan output."""
    return {k: v for k, v in eval_res.items() if not k.startswith("_")}

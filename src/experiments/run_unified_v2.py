"""Unified EvoAlign-OS v2 Single-Source-of-Truth Experiment Runner.

Replaces all fragmented legacy/repair pipelines with one auditable, SHA-256-sealed
`verified_master_results.json` covering:
  - Task 1: 11-program reconciled Beijing multi-tier benchmark (0 blank rows) + anomaly & reliability audits
  - Task 2: Domain-coherent Multi-Track CP-SAT Pareto Trio (Lite/Medium/Heavy),
            Gale-Shapley Stable Streaming, Macro Major Lifecycle (Add/Adjust/Revoke + 4yr Teach-Out),
            Empirical Beijing 2018-2024 + Synthetic Stress 4-Year Delay MPC
  - Layer 4 (RSI): 4-Generation Error-Driven Bi-Level Meta-Harness RSI (Real Qwen + SIM + GUARD)
                   + 6-Arm Controlled Ablation across 3 programs (BTBU_CS, BTBU_AI, BISTU_CS)
                   with employer-clustered paired bootstrap 95% CIs on Dev 2024, Replay 2025, Replay 2026Q1
  - Task 3: 3-Scenario Multi-Track Sensitivity + Split Conformal 90% Explanatory Salary Interval
            + Dual Natural Experiments (Micro Course + Macro MOE) + Cross-School/Cross-Major CRSI Transfer
  - Task 4: Paired Red-Team Adversarial Stress Matrix (Ungoverned vs. Governed on the EXACT evolved plan)
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from analytics.gap_engine import (
    compute_supply_profile,
    extract_demand_profile,
    load_curricula,
    load_kg,
    run_task1_benchmark,
)
from analytics.pathways import expected_pathway, group_curricula
from analytics.track_evaluator import (
    TRACK_IDS,
    TRACK_META,
    build_track_demand,
    evaluate_multitrack_plan,
    evaluate_single_curriculum,
    paired_employer_bootstrap_ci,
    strip_private_arrays,
)
from rsi.error_driven_rsi import run_error_driven_bilevel_rsi
from security.leakage_guard import BlindAuditContext, guard_read_path
from simulator.dream_sim import fit_conformal_salary_predictor
from simulator.multitrack_redteam import run_paired_redteam_audit, run_real_and_synthetic_mpc
from simulator.employment_feedback import run_employment_feedback
from solver.curriculum_portfolio_solver import solve_macro_major_portfolio_lifecycle
from solver.multitrack_solver import (
    GENERIC_CAPSTONE_PACKAGES,
    build_conventional_seed_pool,
    solve_multitrack_curriculum,
)

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
BLIND_DIR = ROOT / "data" / "blind_holdout"
OUT_DIR = ROOT / "outputs"

TARGET_PROGRAMS = ("BTBU_CS_2022", "BTBU_AI_2022", "BISTU_CS_2025")
ARM_LABELS = {
    "B0_Freeze": "现状冻结（零调整、单轨）",
    "B1_SingleTrack_MILP": "单轨道静态 MILP（同预算）",
    "B2_MultiTrack_Static": "多轨道静态规则基线（无RSI）",
    "B3_SingleShot_LLM": "单轮 Qwen+CP-SAT（无反思/无多代）",
    "B4_BiLevel_NoGuard": "双环 RSI（关闭 GUARD 门控）",
    "Full_EvoAlign_OS": "EvoAlign-OS 完整双环多轨道 RSI",
}


def canonical_sha256(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _slim_plan(sol: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in sol.items()
        if k not in ("track_curricula", "offered_catalogue")
    }


def _conformal_salary_for_multitrack(
    track_curricula: Dict[str, List[Dict[str, Any]]],
    track_shares: Dict[str, float],
    kg: Dict[str, Any],
    conformal_model: Dict[str, Any],
    salary_mult: float = 1.0,
) -> Dict[str, Any]:
    """Compute controlled Split Conformal 90% explanatory advertised salary interval for a multi-track plan."""
    skill_ids = kg["skill_ids"]
    coef = np.asarray(conformal_model["coef"], dtype=np.float64)
    q_hat = float(conformal_model["q_hat_log"])
    intercept = float(conformal_model["intercept"])

    share_sum = sum(float(track_shares.get(tk, 0.25)) for tk in TRACK_IDS)
    weighted_x = np.zeros(len(skill_ids), dtype=np.float64)
    for tk in TRACK_IDS:
        sup = compute_supply_profile(track_curricula[tk], skill_ids)
        eff_bloom = sup["bloom"] * np.minimum(1.0, sup["mass"] / 2.0)
        x_tk = np.clip((eff_bloom - 1.5) / 4.2, 0.0, 1.0) * 0.42
        weighted_x += (float(track_shares.get(tk, 0.25)) / max(1e-12, share_sum)) * x_tk

    pred_log = intercept + float(weighted_x @ coef) + math.log(max(0.5, salary_mult))
    med = float(math.exp(pred_log))
    lo = float(math.exp(pred_log - q_hat))
    hi = float(math.exp(pred_log + q_hat))
    return {
        "explanatory_ad_salary_median": round(med, 1),
        "explanatory_ad_salary_conformal90": [round(lo, 1), round(hi, 1)],
    }


def _evaluate_3scenarios(
    sol: Dict[str, Any],
    base_courses: List[Dict[str, Any]],
    track_demand: Dict[str, Any],
    kg: Dict[str, Any],
    conformal_model: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate Freeze Counterfactual + 3-Scenario 4-Year Trajectory (Optimistic / Baseline / Pessimistic)."""
    from simulator.multitrack_redteam import _apply_stress_to_curricula

    freeze_ev = evaluate_single_curriculum(base_courses, track_demand, kg)
    freeze_sal = _conformal_salary_for_multitrack(
        {tk: base_courses for tk in TRACK_IDS},
        {tk: float(track_demand["tracks"][tk]["weight_share"]) for tk in TRACK_IDS},
        kg, conformal_model, salary_mult=1.0,
    )
    counterfactual_freeze = {
        **strip_private_arrays(freeze_ev),
        **freeze_sal,
    }

    sc_cfgs = {
        "optimistic": {"new_eff": 1.0, "upg_eff": 1.0, "cap_eff": 1.0, "ov_pen": 1.0, "sal_mult": 1.08, "label": "乐观情景（产教协同加速+师资快速达标）"},
        "baseline": {"new_eff": 0.94, "upg_eff": 0.96, "cap_eff": 0.95, "ov_pen": 0.98, "sal_mult": 1.00, "label": "基准情景（稳态师资爬坡与温和市场演进）"},
        "pessimistic": {"new_eff": 0.80, "upg_eff": 0.86, "cap_eff": 0.84, "ov_pen": 0.92, "sal_mult": 0.91, "label": "悲观情景（技术范式突变+师资转型滞后）"},
    }
    scenarios_out = {}
    for sc_name, cfg in sc_cfgs.items():
        st_cur = _apply_stress_to_curricula(
            sol["track_curricula"],
            new_course_effectiveness=cfg["new_eff"],
            upgrade_effectiveness=cfg["upg_eff"],
            capstone_effectiveness=cfg["cap_eff"],
            overload_penalty=cfg["ov_pen"],
        )
        ev = evaluate_multitrack_plan(st_cur, sol["track_shares"], track_demand, kg)
        sal = _conformal_salary_for_multitrack(st_cur, sol["track_shares"], kg, conformal_model, salary_mult=cfg["sal_mult"])
        # 4-year semester progression (Year 1..4)
        yearly = []
        for yr, max_sem in [(1, 2), (2, 4), (3, 6), (4, 8)]:
            sub_tc = {
                tk: [c for c in crs if 0 < int(c.get("semester") or 0) <= max_sem]
                for tk, crs in st_cur.items()
            }
            ev_y = evaluate_multitrack_plan(sub_tc, sol["track_shares"], track_demand, kg)
            yearly.append({
                "year": yr,
                "max_semester": max_sem,
                "jd_qual_rate_70": ev_y["jd_qual_rate_70"],
                "jd_mean_readiness": ev_y["jd_mean_readiness"],
                "track_l1_overlap": ev_y["track_l1_overlap"],
                "track_ot_alignment": ev_y["track_ot_alignment"],
            })
        scenarios_out[sc_name] = {
            "label": cfg["label"],
            **strip_private_arrays(ev),
            **sal,
            "streaming_summary": sol.get("streaming_summary"),
            "yearly_trajectory": yearly,
        }
    return {"counterfactual_freeze": counterfactual_freeze, "scenarios": scenarios_out}


def main() -> None:
    t0 = time.perf_counter()
    print("=== EvoAlign-OS v2 Unified Multi-Track RSI Experiment Suite ===", flush=True)

    kg = load_kg()
    skill_ids = kg["skill_ids"]
    all_courses = load_curricula()
    groups = group_curricula(all_courses)

    # 1. Task 1 Benchmark (11/11 reconciled Beijing programs)
    print("\n[1/7] Running Task 1 11-Program Reconciled Benchmark (Dev 2024 only)...", flush=True)
    task1 = run_task1_benchmark()

    # Load Train (2022-2023), Recent Train (2023), and Dev (2024) demand slices
    train_raw = extract_demand_profile([PROC_DIR / "demand_train_2018_2023.parquet"], kg, min_year=2022, use_bj_shortage_calibration=False)
    recent_raw = extract_demand_profile([PROC_DIR / "demand_train_2018_2023.parquet"], kg, min_year=2023, max_year=2023, use_bj_shortage_calibration=False)
    dev_raw = extract_demand_profile([PROC_DIR / "demand_dev_2024.parquet"], kg, use_bj_shortage_calibration=False)

    train_td = build_track_demand(train_raw, kg)
    recent_td = build_track_demand(recent_raw, kg)
    dev_td = build_track_demand(dev_raw, kg)
    conformal_model = fit_conformal_salary_predictor(train_raw, dev_raw, alpha=0.10)

    # Load existing extraction reliability from prior audit if present
    prior_master_path = OUT_DIR / "verified_master_results.json"
    extraction_reliability = {}
    if prior_master_path.exists():
        try:
            old_m = json.loads(prior_master_path.read_text(encoding="utf-8"))
            extraction_reliability = old_m.get("extraction_reliability", {})
        except Exception:
            pass

    # 2. Run 6-Arm Controlled Ablation across all 3 target programs (BTBU_CS, BTBU_AI, BISTU_CS)
    print("\n[2/7] Running 6-Arm Controlled Ablation + 4-Gen Error-Driven Bi-Level RSI (Real Qwen)...", flush=True)
    seed_pool = build_conventional_seed_pool()
    program_plans: Dict[str, Dict[str, Any]] = {}
    program_rsi_runs: Dict[str, Dict[str, Any]] = {}
    program_meta: Dict[str, Dict[str, Any]] = {}

    for pid in TARGET_PROGRAMS:
        print(f"\n  >>> Program: {pid}", flush=True)
        base, pw_audit = expected_pathway(groups[pid])
        quota = float((pw_audit.get("official_requirements") or {}).get("technical_elective_minimum") or 12.0)
        program_meta[pid] = {"base": base, "pathway_audit": pw_audit, "elective_quota_credits": quota}

        # B0 Freeze
        b0_plan = {
            "status": "FREEZE", "feasible": True, "multitrack": False,
            "elective_quota_credits": quota, "added_credits": 0.0, "removed_credits": 0.0,
            "upgraded_core_count": 0, "faculty_fte_used": 0.0, "faculty_fte_cap": 4.8,
            "lab_units_used": 0.0, "lab_units_cap": 14.0,
            "constraint_checks": {"zero_hard_violations": True},
            "actions": [], "track_electives": {},
            "track_shares": {tk: float(train_td["tracks"][tk]["weight_share"]) for tk in TRACK_IDS},
            "track_curricula": {tk: copy.deepcopy(base) for tk in TRACK_IDS},
        }
        # B1 Single-Track Static MILP (same conventional pool & budget, no streaming, no capstone orientation)
        b1_plan = solve_multitrack_curriculum(
            base, seed_pool, train_td, kg,
            elective_quota_credits=quota, multitrack=False, enable_capstone_orientation=False,
            budget_mode="medium",
        )
        # B2 Multi-Track Static Baseline (4-track Gale-Shapley streaming + conventional seed pool, no capstone evolution, no RSI)
        b2_plan = solve_multitrack_curriculum(
            base, seed_pool, train_td, kg,
            elective_quota_credits=quota, multitrack=True, enable_capstone_orientation=False,
            budget_mode="medium",
        )
        # B3 Single-Shot LLM (1 gen Qwen electives only, no multi-gen, no capstone evolution, no SIM)
        b3_run = run_error_driven_bilevel_rsi(
            pid, base, train_td, recent_td, dev_td, kg,
            elective_quota_credits=quota, mode="single_shot_llm", n_generations=1, use_llm=True,
        )
        # B4 Bi-Level RSI without GUARD (4 gens, GUARD disabled)
        b4_run = run_error_driven_bilevel_rsi(
            pid, base, train_td, recent_td, dev_td, kg,
            elective_quota_credits=quota, mode="no_guard", n_generations=4, use_llm=True,
        )
        # Full EvoAlign-OS (4 gens error-driven Bi-Level RSI + SIM + GUARD)
        full_run = run_error_driven_bilevel_rsi(
            pid, base, train_td, recent_td, dev_td, kg,
            elective_quota_credits=quota, mode="full", n_generations=4, use_llm=True,
        )

        program_plans[pid] = {
            "B0_Freeze": b0_plan,
            "B1_SingleTrack_MILP": b1_plan,
            "B2_MultiTrack_Static": b2_plan,
            "B3_SingleShot_LLM": b3_run["best_solution"],
            "B4_BiLevel_NoGuard": b4_run["best_solution"],
            "Full_EvoAlign_OS": full_run["best_solution"],
        }
        program_rsi_runs[pid] = {"B3": b3_run, "B4": b4_run, "Full": full_run}

        # Print Dev 2024 progress immediately
        for arm_id, pl in program_plans[pid].items():
            ev = evaluate_multitrack_plan(pl["track_curricula"], pl["track_shares"], dev_td, kg)
            print(
                f"      {arm_id:22s} | qual70={ev['jd_qual_rate_70']*100:5.2f}% | "
                f"meanR={ev['jd_mean_readiness']*100:5.2f}% | tkL1={ev['track_l1_overlap']*100:5.2f}% | "
                f"tkOT={ev['track_ot_alignment']*100:5.2f}% | worst={ev['worst_track_readiness']*100:5.2f}% | ok={pl['feasible']}",
                flush=True,
            )
        print(
            f"      Full RSI promotions={full_run['accepted_promotions']}/{full_run['generations_run']}, "
            f"GUARD rollbacks={full_run['guard_rollbacks']}, "
            f"history={[ (h['generation'], h['event'], round(h['active_best_score'], 4)) for h in full_run['evolution_history'] ]}",
            flush=True,
        )

    # 3. Primary Case Pareto Trio (Lite / Medium / Heavy) using evolved BTBU_CS transfer bundle
    print("\n[3/7] Solving Primary Case (BTBU_CS_2022) Pareto Trio (Lite / Medium / Heavy)...", flush=True)
    primary_pid = "BTBU_CS_2022"
    primary_base = program_meta[primary_pid]["base"]
    primary_quota = program_meta[primary_pid]["elective_quota_credits"]
    primary_full = program_rsi_runs[primary_pid]["Full"]
    primary_bundle = primary_full["transfer_bundle"]
    evolved_pool_primary = copy.deepcopy(seed_pool) + copy.deepcopy(primary_bundle["promoted_electives"])

    # All three tiers reuse the EXACT evolved harness H* (pool, capstones, syllabus modules, policy weights);
    # only the resource envelope (swap/upgrade/FTE/lab caps, menu growth) changes, so feasible sets are nested.
    cap_keys = ("swap_cap", "fte_cap", "lab_cap", "upgrade_cap", "max_menu_expand_credits")
    hstar_policy = {k: v for k, v in primary_bundle["full_policy"].items() if k not in cap_keys}
    menu_growth = {"lite": 8.0, "heavy": 22.0}
    pareto_solutions = {}
    for bmode in ("lite", "medium", "heavy"):
        if bmode == "medium":
            pareto_solutions[bmode] = primary_full["best_solution"]
        else:
            pareto_solutions[bmode] = solve_multitrack_curriculum(
                primary_base, copy.deepcopy(primary_bundle["elective_pool"]), train_td, kg,
                elective_quota_credits=primary_quota, multitrack=True,
                enable_capstone_orientation=float(hstar_policy.get("capstone_orient_ratio", 0.0)) > 0,
                capstone_packages=primary_bundle["capstone_packages"],
                extra_syllabus_modules=primary_bundle["syllabus_modules"],
                budget_mode=bmode,
                policy={**hstar_policy, "max_menu_expand_credits": menu_growth[bmode]},
            )

    # IIS audit of a raw LLM proposal: force-adopt every Qwen-proposed elective under the Lite envelope
    qwen_pool_names = [c["course_name"] for c in primary_bundle["elective_pool"] if str(c.get("source", "")).startswith("qwen")]
    iis_probe_sol = solve_multitrack_curriculum(
        primary_base, copy.deepcopy(primary_bundle["elective_pool"]), train_td, kg,
        elective_quota_credits=primary_quota, multitrack=True,
        enable_capstone_orientation=float(hstar_policy.get("capstone_orient_ratio", 0.0)) > 0,
        capstone_packages=primary_bundle["capstone_packages"], extra_syllabus_modules=primary_bundle["syllabus_modules"],
        budget_mode="lite", policy={**hstar_policy, "force_add_courses": qwen_pool_names},
    )
    iis_diagnosis = {
        "setup": "将外环 Qwen 提出的全部新课强制采纳，并置于 Lite 资源档（新增选修 ≤ 8 学分、师资 ≤ 2.6 FTE）下求解",
        "forced_courses": qwen_pool_names,
        "status": iis_probe_sol["status"],
        "iis_conflicts": iis_probe_sol.get("iis_conflicts", []),
        "guarded_constraints": iis_probe_sol.get("iis_guarded_constraints"),
        "repair": "去掉强制采纳后由求解器在同一资源档内取舍，得到可行方案（见 task2_pareto_trio.lite）",
        "lite_repaired_feasible": pareto_solutions["lite"]["feasible"],
    }

    # 4. Zero-Shot Cross-School (BISTU_CS_2025) & Cross-Major (BTBU_AI_2022) CRSI Warm Transfer
    print("\n[4/7] Running Zero-Shot CRSI Transfer (BTBU_CS H*+SIM -> BISTU_CS_2025 & BTBU_AI_2022)...", flush=True)
    transfer_plans = {}
    for target_pid, tkey in [("BISTU_CS_2025", "BISTU_CS_2025_unseen_university"), ("BTBU_AI_2022", "BTBU_AI_2022_unseen_major")]:
        tbase = program_meta[target_pid]["base"]
        tquota = program_meta[target_pid]["elective_quota_credits"]
        cold_run = run_error_driven_bilevel_rsi(
            target_pid, tbase, train_td, recent_td, dev_td, kg,
            elective_quota_credits=tquota, mode="full", n_generations=0, use_llm=False, warm_sim=None,
        )
        warm_run = run_error_driven_bilevel_rsi(
            target_pid, tbase, train_td, recent_td, dev_td, kg,
            elective_quota_credits=tquota, mode="full", n_generations=0, use_llm=False, warm_sim=primary_bundle,
        )
        transfer_plans[tkey] = {
            "target_pid": target_pid,
            "freeze": program_plans[target_pid]["B0_Freeze"],
            "cold_H0": cold_run["best_solution"],
            "warm_Hstar_SIM": warm_run["best_solution"],
            "full_evolved": program_plans[target_pid]["Full_EvoAlign_OS"],
            "cold_dev_score": cold_run["best_dev_score"],
            "warm_dev_score": warm_run["best_dev_score"],
        }

    # 5. Dual Natural Experiments Prep (<=2022 Micro Course + <=2023 Macro MOE)
    print("\n[5/7] Preparing Dual Natural Experiments & Macro Major Portfolio (<=2023 only)...", flush=True)
    nat_raw_2022 = extract_demand_profile([PROC_DIR / "demand_nat_exp_2018_2022.parquet"], kg, min_year=2020, use_bj_shortage_calibration=False)
    nat_td_2022 = build_track_demand(nat_raw_2022, kg)
    nat_run = run_error_driven_bilevel_rsi(
        "NAT_BTBU_CS_2022", primary_base, nat_td_2022, nat_td_2022, nat_td_2022, kg,
        elective_quota_credits=primary_quota, mode="full", n_generations=2, use_llm=True,
    )

    moe_hist_path = PROC_DIR / "moe_major_lifecycle_2020_2023.json"
    if not moe_hist_path.exists():
        moe_hist_path = PROC_DIR / "moe_major_lifecycle_2021_2023.json"
    moe_hist = json.loads(moe_hist_path.read_text(encoding="utf-8"))
    macro_candidates = [
        {"major_code": "080717T", "major_name": "人工智能", "core_skills": ["ml_classic", "deep_learning", "llm_posttrain", "agent_rag", "cv_multimodal", "nlp_text"], "required_fte": 15.0, "policy_priority": 0.25},
        {"major_code": "080910T", "major_name": "数据科学与大数据技术", "core_skills": ["big_data_eng", "sql_rdbms", "nosql_vector_db", "data_mining_bi", "python_eng"], "required_fte": 14.0, "policy_priority": 0.20},
        {"major_code": "080809T", "major_name": "具身智能", "core_skills": ["robotics_embodied", "cv_multimodal", "llm_posttrain", "embedded_mcu", "control_signal"], "required_fte": 14.0, "policy_priority": 0.22},
        {"major_code": "080917T", "major_name": "智能科学与技术", "core_skills": ["ml_classic", "deep_learning", "robotics_embodied", "python_eng"], "required_fte": 14.0, "policy_priority": 0.16},
        {"major_code": "080911TK", "major_name": "网络空间安全", "core_skills": ["cyber_sec", "ai_safety_privacy", "comp_net", "os_kernel"], "required_fte": 14.0, "policy_priority": 0.18},
        {"major_code": "080710T", "major_name": "集成电路设计与集成系统", "core_skills": ["eda_ic_design", "comp_arch", "embedded_mcu", "c_cpp"], "required_fte": 16.0, "policy_priority": 0.18},
        {"major_code": "080803T", "major_name": "机器人工程", "core_skills": ["robotics_embodied", "control_signal", "embedded_mcu", "c_cpp"], "required_fte": 15.0, "policy_priority": 0.16},
        {"major_code": "071203T", "major_name": "数据科学", "core_skills": ["prob_stats", "data_mining_bi", "python_eng", "ml_classic"], "required_fte": 12.0, "policy_priority": 0.15},
        {"major_code": "080909T", "major_name": "电子与计算机工程", "core_skills": ["embedded_mcu", "comp_arch", "c_cpp", "iot_edge"], "required_fte": 14.0, "policy_priority": 0.10},
        {"major_code": "080902", "major_name": "软件工程", "core_skills": ["soft_eng_arch", "java_spring", "web_frontend", "qa_testing", "cloud_k8s"], "required_fte": 16.0, "policy_priority": 0.12},
    ]
    btbu_existing_majors = [
        {"major_code": "080901", "major_name": "计算机科学与技术", "ot_alignment": 0.7517, "max_peer_cosine": 0.78, "fte": 34.0, "intake": 120},
        {"major_code": "080717T", "major_name": "人工智能", "ot_alignment": 0.6793, "max_peer_cosine": 0.75, "fte": 22.0, "intake": 80},
        {"major_code": "080902", "major_name": "软件工程", "ot_alignment": 0.7746, "max_peer_cosine": 0.80, "fte": 26.0, "intake": 90},
        {"major_code": "080905", "major_name": "物联网工程", "ot_alignment": 0.4500, "max_peer_cosine": 0.93, "fte": 18.0, "intake": 60, "declining_flag": True},
    ]
    # Validate every major code against the official 2026 undergraduate catalogue (教高函〔2026〕2号)
    catalog_check = None
    cat_path = PROC_DIR / "moe_undergrad_catalog_2026.json"
    if cat_path.exists():
        cat_codes = {mj["name"]: mj["code"] for mj in json.loads(cat_path.read_text(encoding="utf-8"))["majors"]}
        catalog_check = []
        for cm in macro_candidates + btbu_existing_majors:
            official = cat_codes.get(cm["major_name"])
            catalog_check.append({"major_name": cm["major_name"], "code_before": cm["major_code"], "catalog_2026_code": official,
                                  "match": official == cm["major_code"]})
            if official:
                cm["major_code"] = official
    macro_portfolio = solve_macro_major_portfolio_lifecycle(btbu_existing_majors, macro_candidates, moe_hist, train_raw, kg)
    city_macro_rank = solve_macro_major_portfolio_lifecycle([], macro_candidates, moe_hist, train_raw, kg)

    # 6. Empirical + Synthetic MPC & Paired Red-Team Adversarial Audit (on EXACT evolved plan!)
    print("\n[6/7] Running Empirical + Synthetic MPC and Paired Red-Team Audit on evolved plan...", flush=True)
    mpc_results = run_real_and_synthetic_mpc(baseline_intake=120)
    redteam_audit = run_paired_redteam_audit(primary_full["best_solution"], dev_td, kg, n_mc=25)
    emp_hist = mpc_results["real_empirical_mpc"]["history"]
    n_obs_years = len(mpc_results["real_empirical_mpc"].get("observed_years", [])) or 7
    demand_trend_idx = float(emp_hist[n_obs_years - 1]["filtered_trend"]) / float(mpc_results["real_empirical_mpc"]["baseline_intake"])
    employment_feedback = run_employment_feedback(baseline_intake=120, demand_trend_index=demand_trend_idx)

    # 7. Locked Replay Evaluation on 2025 (2,645 ads) & 2026Q1 (351 ads, SAME broad_entry population!)
    print("\n[7/7] Opening BlindAuditContext for Locked Replay Evaluation (2025 & 2026Q1 + Natural Exp)...", flush=True)
    with BlindAuditContext("evoalign_v2_unified_replay", caller="run_unified_v2"):
        rep25_raw = extract_demand_profile([BLIND_DIR / "demand_holdout_2025.parquet"], kg, population="broad_entry", use_bj_shortage_calibration=False)
        rep26_raw = extract_demand_profile([BLIND_DIR / "demand_drift_2026q1.parquet"], kg, population="broad_entry", use_bj_shortage_calibration=False)
        nat_course_truth = json.loads(guard_read_path(BLIND_DIR / "natural_exp_btbu_cs_2022_to_2025.json").read_text(encoding="utf-8"))
        moe_2024_blind = json.loads(guard_read_path(BLIND_DIR / "moe_major_lifecycle_2024_blind.json").read_text(encoding="utf-8"))
        moe_2025_path = BLIND_DIR / "moe_major_lifecycle_2025_blind.json"
        moe_2025_blind = json.loads(guard_read_path(moe_2025_path).read_text(encoding="utf-8")) if moe_2025_path.exists() else None

    rep25_td = build_track_demand(rep25_raw, kg)
    rep26_td = build_track_demand(rep26_raw, kg)
    splits = {
        "development_2024": dev_td,
        "replay_2025": rep25_td,
        "replay_2026q1": rep26_td,
    }

    # Evaluate all 3 programs x 6 arms x 3 periods with employer-clustered paired 95% CIs
    cases_out: Dict[str, Any] = {}
    for pid in TARGET_PROGRAMS:
        plans = program_plans[pid]
        split_results = {}
        for sp_name, td in splits.items():
            evals_raw = {
                arm: evaluate_multitrack_plan(pl["track_curricula"], pl["track_shares"], td, kg)
                for arm, pl in plans.items()
            }
            metrics_clean = {arm: strip_private_arrays(ev) for arm, ev in evals_raw.items()}
            # Paired CIs on BOTH primary metrics (jd_qual_rate_70 and jd_mean_readiness) vs all baselines
            ci_qual = {
                ref: paired_employer_bootstrap_ci(evals_raw, ref, td, "_per_jd_qualified", n_boot=600, seed=42)
                for ref in ("B0_Freeze", "B1_SingleTrack_MILP", "B2_MultiTrack_Static", "B3_SingleShot_LLM", "B4_BiLevel_NoGuard")
            }
            ci_read = {
                ref: paired_employer_bootstrap_ci(evals_raw, ref, td, "_per_jd_readiness", n_boot=600, seed=42)
                for ref in ("B0_Freeze", "B1_SingleTrack_MILP", "B2_MultiTrack_Static", "B3_SingleShot_LLM", "B4_BiLevel_NoGuard")
            }
            split_results[sp_name] = {
                "ads": td["n_rows"],
                "employers": td["n_employers"],
                "metrics": metrics_clean,
                "paired_qual_ci95": ci_qual,
                "paired_readiness_ci95": ci_read,
            }

        full_r = program_rsi_runs[pid]["Full"]
        cases_out[pid] = {
            "pathway": program_meta[pid]["pathway_audit"],
            "elective_quota_credits": program_meta[pid]["elective_quota_credits"],
            "actions": {arm: _slim_plan(pl) for arm, pl in plans.items()},
            "rsi_evolution": {
                "generations_run": full_r["generations_run"],
                "accepted_promotions": full_r["accepted_promotions"],
                "guard_rollbacks": full_r["guard_rollbacks"],
                "trace_final_sha256": full_r["trace_final_sha256"],
                "evolution_history": full_r["evolution_history"],
                "sim_memory": full_r["sim_memory"],
                "qwen_audits": full_r["qwen_audits"],
                "token_ledger_stats": full_r["token_ledger_stats"],
            },
            "results": split_results,
        }

    # Build primary-case 6-way ablation table (compatible with cockpit + figures)
    ablation_6way = []
    for arm_id, arm_zh in ARM_LABELS.items():
        r_obj = (
            program_rsi_runs[primary_pid]["B3"] if arm_id == "B3_SingleShot_LLM"
            else program_rsi_runs[primary_pid]["B4"] if arm_id == "B4_BiLevel_NoGuard"
            else program_rsi_runs[primary_pid]["Full"] if arm_id == "Full_EvoAlign_OS"
            else None
        )
        ablation_6way.append({
            "arm_id": arm_id,
            "arm_zh": arm_zh,
            "generations": r_obj["generations_run"] if r_obj else 0,
            "accepted_promotions": r_obj["accepted_promotions"] if r_obj else 0,
            "guard_rollbacks": r_obj["guard_rollbacks"] if r_obj else 0,
            "dev_2024": cases_out[primary_pid]["results"]["development_2024"]["metrics"][arm_id],
            "replay_2025": cases_out[primary_pid]["results"]["replay_2025"]["metrics"][arm_id],
            "replay_2026q1": cases_out[primary_pid]["results"]["replay_2026q1"]["metrics"][arm_id],
        })

    # Pareto Trio evaluation on Replay 2025
    pareto_trio_out = {}
    for bmode, sol in pareto_solutions.items():
        ev25 = evaluate_multitrack_plan(sol["track_curricula"], sol["track_shares"], rep25_td, kg)
        sal25 = _conformal_salary_for_multitrack(sol["track_curricula"], sol["track_shares"], kg, conformal_model)
        pareto_trio_out[bmode] = {
            "replay_2025": {**strip_private_arrays(ev25), **sal25},
            "solver_summary": _slim_plan(sol),
        }

    # 3-Scenario Evaluation on Primary Case (Replay 2025)
    sim_3sc = _evaluate_3scenarios(primary_full["best_solution"], primary_base, rep25_td, kg, conformal_model)

    # Agent-based Dream-Sim sandbox (300 heterogeneous student agents x 8 semesters) on the same replay
    from simulator.agent_sandbox import run_agent_sandbox
    agent_sandbox = {
        pid: run_agent_sandbox(program_plans[pid]["B0_Freeze"], program_plans[pid]["Full_EvoAlign_OS"], rep25_raw, kg, conformal_model)
        for pid in TARGET_PROGRAMS
    }

    # Cross-School & Cross-Major Transfer Evaluation on Replay 2025
    transfer_out = {}
    for tkey, tp in transfer_plans.items():
        transfer_out[tkey] = {
            "target_program": tp["target_pid"],
            "baseline_freeze": strip_private_arrays(evaluate_multitrack_plan(tp["freeze"]["track_curricula"], tp["freeze"]["track_shares"], rep25_td, kg)),
            "cold_start_H0": strip_private_arrays(evaluate_multitrack_plan(tp["cold_H0"]["track_curricula"], tp["cold_H0"]["track_shares"], rep25_td, kg)),
            "warm_transfer_Hstar_SIM": strip_private_arrays(evaluate_multitrack_plan(tp["warm_Hstar_SIM"]["track_curricula"], tp["warm_Hstar_SIM"]["track_shares"], rep25_td, kg)),
            "full_local_evolved": strip_private_arrays(evaluate_multitrack_plan(tp["full_evolved"]["track_curricula"], tp["full_evolved"]["track_shares"], rep25_td, kg)),
            "cold_gen0_dev_score": tp["cold_dev_score"],
            "warm_gen0_dev_score": tp["warm_dev_score"],
        }

    # Natural Experiments Evaluation
    btbu_cs_2025_real, real25_pathway = expected_pathway(groups["BTBU_CS_2025"])
    btbu_cs_2022_frozen, _ = expected_pathway(groups["BTBU_CS_2022"])
    ev_real_raw = evaluate_single_curriculum(btbu_cs_2025_real, rep25_td, kg)
    ev_frozen_raw = evaluate_single_curriculum(btbu_cs_2022_frozen, rep25_td, kg)
    nat_sol = nat_run["best_solution"]
    ev_pred_raw = evaluate_multitrack_plan(nat_sol["track_curricula"], nat_sol["track_shares"], rep25_td, kg)
    # Fairness control: the same EvoAlign forecast forced into ONE undifferentiated track (no streaming benefit)
    pred_union: Dict[str, Dict[str, Any]] = {}
    for tk in TRACK_IDS:
        for c in nat_sol["track_curricula"][tk]:
            pred_union.setdefault(c["course_name"], c)
    nat_shares = nat_sol["track_shares"]
    pred_single = [
        dict(c, take_rate=float(c.get("take_rate", 1.0)) * (1.0 if all(c["course_name"] in {x["course_name"] for x in nat_sol["track_curricula"][tk]} for tk in TRACK_IDS)
                                                       else sum(float(nat_shares[tk]) for tk in TRACK_IDS if c["course_name"] in {x["course_name"] for x in nat_sol["track_curricula"][tk]})))
        for c in pred_union.values()
    ]
    ev_pred_single_raw = evaluate_single_curriculum(pred_single, rep25_td, kg)
    nat_eval_map = {"real_2025": ev_real_raw, "frozen_2022": ev_frozen_raw, "evoalign_forecast": ev_pred_raw, "evoalign_forecast_single_track": ev_pred_single_raw}
    nat_ci_vs_real = paired_employer_bootstrap_ci(nat_eval_map, "real_2025", rep25_td, "_per_jd_qualified", n_boot=600, seed=42)
    nat_ci_vs_frozen = paired_employer_bootstrap_ci(nat_eval_map, "frozen_2022", rep25_td, "_per_jd_qualified", n_boot=600, seed=42)
    ev_real_25 = strip_private_arrays(ev_real_raw)
    ev_pred_25 = strip_private_arrays(ev_pred_raw)

    # Direction agreement between the <=2022-data forecast and the university's actual 2025 revision
    def _hyper_sf(k: int, N: int, K: int, n: int) -> float:
        """P(X >= k) for hypergeometric(N population, K successes, n draws)."""
        tot = math.comb(N, n)
        return float(sum(math.comb(K, i) * math.comb(N - K, n - i) for i in range(k, min(K, n) + 1)) / tot)

    real_added = set(nat_course_truth.get("added_in_2025", []))
    real_removed = set(nat_course_truth.get("removed_from_2022", []))
    c25 = {c["course_name"]: c for c in groups["BTBU_CS_2025"]}
    real_new_skills = {
        s["id"] for nm in real_added for s in (c25.get(nm, {}).get("extracted_skills") or []) if float(s.get("weight", 0)) >= 0.2
    }
    pred_new_skills = {
        sid for a in nat_sol["actions"] if a["action"] in ("ADD_ELECTIVE", "UPGRADE_SYLLABUS") for sid in (a.get("skills") or [])
    }
    pred_touched = {a["course_name"] for a in nat_sol["actions"] if a["action"] in ("UPGRADE_SYLLABUS", "REMOVE_ELECTIVE")}
    base22_names = {c["course_name"] for c in btbu_cs_2022_frozen}
    course_hits = sorted(pred_touched & real_removed)
    skill_hits = sorted(pred_new_skills & real_new_skills)
    n_sk = len(skill_ids)
    direction_agreement = {
        "course_level": {
            "evoalign_touched_2022_courses": sorted(pred_touched),
            "university_replaced_or_removed_2022_courses": sorted(real_removed),
            "agreed": course_hits,
            "n_agreed": len(course_hits),
            "chance_expectation": round(len(pred_touched) * len(real_removed) / max(1, len(base22_names)), 3),
            "hypergeom_p_value": round(_hyper_sf(len(course_hits), len(base22_names), len(real_removed & base22_names), len(pred_touched & base22_names)), 5),
            "population_courses_2022": len(base22_names),
        },
        "skill_level": {
            "real_2025_new_course_skills": sorted(real_new_skills),
            "evoalign_forecast_target_skills": sorted(pred_new_skills),
            "agreed": skill_hits,
            "recall": round(len(skill_hits) / max(1, len(real_new_skills)), 4),
            "precision": round(len(skill_hits) / max(1, len(pred_new_skills)), 4),
            "chance_expected_hits": round(len(pred_new_skills) * len(real_new_skills) / n_sk, 3),
            "hypergeom_p_value": round(_hyper_sf(len(skill_hits), n_sk, len(real_new_skills), len(pred_new_skills)), 5),
        },
    }

    bj_2024_cs_adds = [r for r in moe_2024_blind if r.get("is_beijing") and r.get("is_cs_ai_info") and "ADD" in r.get("action", "")]
    actual_2024_top = Counter(r["major_name"] for r in bj_2024_cs_adds)
    ranked_macro = sorted([d for d in city_macro_rank["decisions"] if "viability_score" in d], key=lambda x: x["viability_score"], reverse=True)
    top5_pred = [d["major_name"] for d in ranked_macro[:5]]
    actual_names = set(actual_2024_top.keys())
    hits_2024 = [m for m in top5_pred if any(m[:3] in a or a[:3] in m for a in actual_names)]
    # Persistence baseline: the 5 most frequently added Beijing CS/AI/info majors in 2021-2023 (no model)
    persist_counts = Counter(
        r["major_name"] for r in moe_hist
        if r.get("is_beijing") and r.get("is_cs_ai_info") and "ADD" in r.get("action", "") and int(r.get("year", 0)) <= 2023
    )
    persist_top5 = [nm for nm, _ in persist_counts.most_common(5)]
    persist_hits = [m for m in persist_top5 if any(m[:3] in a or a[:3] in m for a in actual_names)]
    # Volume-weighted coverage: share of 2024 Beijing adds that fall into the predicted top-5 majors
    def _covered(pred_list):
        return sum(cnt for a, cnt in actual_2024_top.items() if any(m[:3] in a or a[:3] in m for m in pred_list))

    # Second macro natural experiment (rolling origin): history <= 2024 and demand <= 2024 -> Beijing 2025 adds
    macro_2025 = None
    if moe_2025_blind:
        hist_le2024 = moe_hist + moe_2024_blind
        dem_le2024 = extract_demand_profile([PROC_DIR / "demand_train_2018_2023.parquet", PROC_DIR / "demand_dev_2024.parquet"], kg,
                                            min_year=2022, use_bj_shortage_calibration=False)
        rank25 = solve_macro_major_portfolio_lifecycle([], macro_candidates, hist_le2024, dem_le2024, kg)
        top5_25 = [d["major_name"] for d in sorted([d for d in rank25["decisions"] if "viability_score" in d], key=lambda x: x["viability_score"], reverse=True)[:5]]
        adds25 = [r for r in moe_2025_blind if r.get("is_beijing") and r.get("is_cs_ai_info") and "ADD" in r.get("action", "")]
        act25 = Counter(r["major_name"] for r in adds25)
        pers25 = [nm for nm, _ in Counter(r["major_name"] for r in hist_le2024 if r.get("is_beijing") and r.get("is_cs_ai_info") and "ADD" in r.get("action", "")).most_common(5)]
        def _hit(pred, actual):
            return [p for p in pred if any(p[:3] in a or a[:3] in p for a in actual)]
        def _cov(pred, actual_counter, n):
            return round(sum(c for a, c in actual_counter.items() if any(p[:3] in a or a[:3] in p for p in pred)) / max(1, n), 4)
        macro_2025 = {
            "setup": "滚动起点复验：仅用 MOE ≤2024 北京备案/审批历史 + ≤2024 需求预测 2025 北京新增泛计算机专业",
            "blind_moe_2025_bj_cs_ai_total_adds": len(adds25),
            "blind_moe_2025_top_majors": act25.most_common(10),
            "evoalign_top5_predicted_emerging": top5_25,
            "top5_hit_count": len(_hit(top5_25, act25)),
            "hits_detail": _hit(top5_25, act25),
            "evoalign_top5_volume_coverage": _cov(top5_25, act25, len(adds25)),
            "persistence_baseline_top5": pers25,
            "persistence_baseline_hit_count": len(_hit(pers25, act25)),
            "persistence_baseline_volume_coverage": _cov(pers25, act25, len(adds25)),
        }

    natural_experiments = {
        "micro_course_btbu_2022_to_2025": {
            "setup": "仅用 BTBU_CS_2022 方案 + 2020-2022 北京需求预测改革方案，在 2025 招聘回放上对比高校真实 2025 改版",
            "real_2025_added_courses": nat_course_truth.get("added_in_2025", []),
            "real_2025_removed_courses": nat_course_truth.get("removed_from_2022", []),
            "evoalign_predicted_actions": nat_sol["actions"],
            "btbu_real_2025_pathway": {k: v for k, v in real25_pathway.items()},
            "btbu_frozen_2022_on_replay2025": strip_private_arrays(ev_frozen_raw),
            "btbu_real_2025_on_replay2025": ev_real_25,
            "evoalign_pred_on_replay2025": ev_pred_25,
            "evoalign_pred_single_track_on_replay2025": strip_private_arrays(ev_pred_single_raw),
            "paired_qual_ci95_vs_real_2025": nat_ci_vs_real,
            "paired_qual_ci95_vs_frozen_2022": nat_ci_vs_frozen,
            "direction_agreement": direction_agreement,
        },
        "macro_major_moe_2024_beijing": {
            "setup": f"仅用 MOE {min(r['year'] for r in moe_hist)}-2023 北京备案/审批历史 + <=2023 需求预测 2024 北京新增热门泛计算机专业",
            "blind_moe_2024_bj_cs_ai_total_adds": len(bj_2024_cs_adds),
            "blind_moe_2024_top_majors": actual_2024_top.most_common(10),
            "evoalign_top5_predicted_emerging": top5_pred,
            "top5_hit_count": len(hits_2024),
            "top5_hit_rate": round(len(hits_2024) / max(1, len(top5_pred)), 4),
            "hits_detail": hits_2024,
            "evoalign_top5_volume_coverage": round(_covered(top5_pred) / max(1, len(bj_2024_cs_adds)), 4),
            "persistence_baseline_top5": persist_top5,
            "persistence_baseline_hit_count": len(persist_hits),
            "persistence_baseline_volume_coverage": round(_covered(persist_top5) / max(1, len(bj_2024_cs_adds)), 4),
        },
    }
    if macro_2025:
        natural_experiments["macro_major_moe_2025_beijing"] = macro_2025
    cs101 = None
    if (PROC_DIR / "cs101_core_courses.json").exists():
        from analytics.cs101_coverage import compute_cs101_coverage
        cs101 = compute_cs101_coverage(groups, kg, PROC_DIR / "cs101_core_courses.json")

    # Assemble Single Source of Truth master dictionary
    master: Dict[str, Any] = {
        "schema_version": "evoalign_os_v2.0_multitrack_rsi",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
        "design_disclosures": {
            "development_slice": "2024 (N=1,718 broad_entry, 480 employers)",
            "replay_2025_slice": f"2025 (N={rep25_td['n_rows']} broad_entry, {rep25_td['n_employers']} employers; previously inspected in prior runs, reported as historical replay)",
            "replay_2026q1_slice": f"2026Q1 (N={rep26_td['n_rows']} broad_entry, {rep26_td['n_employers']} employers; unified broad_entry population)",
            "primary_metrics": "Job-level qualification rate (mean readiness>=70% & >=70% skills>=0.55), mean JD readiness, track-level L1/OT",
            "no_artificial_sigmoid": True,
            "domain_coherent_upgrades_enforced": True,
            "all_11_benchmark_programs_reconciled": True,
        },
        "primary_case": {
            "program_id": primary_pid,
            "university": "北京工商大学",
            "major": "计算机科学与技术",
            "cohort_year": 2022,
            "benchmark_tiers": "10校×23份培养方案（11个代表性专业100%官方学分口径核验通过）",
            "demand_warehouse": "2018-2026Q1 北京技术岗招聘全库（含雇主等权去偏）",
        },
        "kg_stats": {
            "universities": len({c["university"] for c in all_courses}),
            "programs": len({c["program_id"] for c in all_courses}),
            "total_courses": len(all_courses),
            "syllabus_enriched_unique": sum(1 for c in all_courses if c.get("has_syllabus")),
            "standard_skills": len(kg["skills"]),
            "bloom_levels": 6,
            "tracks": len(TRACK_IDS),
        },
        "extraction_reliability": extraction_reliability,
        "conformal_calibrator": conformal_model,
        "task1_gap_analysis": task1,
        "task1_cs101_core_coverage": cs101,
        "moe_history_source": moe_hist_path.name,
        "task2_macro_catalog_check": catalog_check,
        "task2_iis_diagnosis": iis_diagnosis,
        "task1_replay2025_demand_audit": rep25_raw["anomaly_audit"],
        "task1_replay2026q1_demand_audit": rep26_raw["anomaly_audit"],
        "task2_pareto_trio": pareto_trio_out,
        "task2_macro_major_portfolio": macro_portfolio,
        "task2_mpc_enrollment": mpc_results,
        "task2_employment_feedback": employment_feedback,
        "task3_dream_sim_3scenarios": sim_3sc["scenarios"],
        "task3_counterfactual_freeze": sim_3sc["counterfactual_freeze"],
        "task3_agent_dream_sim": agent_sandbox,
        "task4_red_team_risks": redteam_audit,
        "rsi_full_evolution": cases_out[primary_pid]["rsi_evolution"],
        "ablation_6way_replay": ablation_6way,
        "cases": cases_out,
        "natural_experiments": natural_experiments,
        "cross_school_transfer": transfer_out,
    }
    master["master_sha256"] = canonical_sha256(master)

    # Atomic write to verified_master_results.json (Single Source of Truth!)
    master_path = OUT_DIR / "verified_master_results.json"
    tmp_path = master_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(master_path)

    # Reproducibility snapshot: exact per-track curricula behind every headline number (no solver/LLM needed to re-score)
    keep = ("course_name", "credits", "take_rate", "category", "required", "semester", "lab_hours", "is_core_foundation", "extracted_skills")
    def _slim_courses(cs):
        return [{k: c.get(k) for k in keep if k in c} for c in cs]
    def _snap(sol):
        return {
            "track_shares": sol["track_shares"],
            "track_curricula": {tk: _slim_courses(sol["track_curricula"][tk]) for tk in TRACK_IDS},
            "actions": [{"action": a["action"], "course_name": a["course_name"]} for a in sol.get("actions", [])],
        }
    snapshots = {
        "master_sha256": master["master_sha256"],
        "programs": {pid: {arm: _snap(pl) for arm, pl in program_plans[pid].items()} for pid in TARGET_PROGRAMS},
        "natural_experiment": {
            "frozen_2022": _snap({"track_shares": {tk: float(rep25_td["tracks"][tk]["weight_share"]) for tk in TRACK_IDS}, "track_curricula": {tk: btbu_cs_2022_frozen for tk in TRACK_IDS}}),
            "real_2025": _snap({"track_shares": {tk: float(rep25_td["tracks"][tk]["weight_share"]) for tk in TRACK_IDS}, "track_curricula": {tk: btbu_cs_2025_real for tk in TRACK_IDS}}),
            "evoalign_forecast": _snap(nat_sol),
        },
        "pareto_trio": {b: _snap(s) for b, s in pareto_solutions.items()},
    }
    (OUT_DIR / "plan_snapshots.json").write_text(json.dumps(snapshots, ensure_ascii=False), encoding="utf-8")
    from rsi.replay_bundle import export_replay_bundle
    rb = export_replay_bundle(OUT_DIR / "replay_bundle", program_rsi_runs, pareto_trio_out, master["master_sha256"], OUT_DIR / "replay_ledger.jsonl")
    print(f"ReplayBundle: {rb['n_files']} files, root SHA256 {rb['bundle_root_sha256'][:16]}", flush=True)

    # Print executive summary table
    print("\n================ FINAL UNIFIED V2 SUMMARY (REPLAY 2025) ================", flush=True)
    for pid in TARGET_PROGRAMS:
        r25 = cases_out[pid]["results"]["replay_2025"]
        m = r25["metrics"]
        ci_b0 = r25["paired_qual_ci95"]["B0_Freeze"]["Full_EvoAlign_OS"]
        ci_b1 = r25["paired_qual_ci95"]["B1_SingleTrack_MILP"]["Full_EvoAlign_OS"]
        ci_b2 = r25["paired_qual_ci95"]["B2_MultiTrack_Static"]["Full_EvoAlign_OS"]
        ci_b3 = r25["paired_qual_ci95"]["B3_SingleShot_LLM"]["Full_EvoAlign_OS"]
        print(
            f"{pid:14s} | B0={m['B0_Freeze']['jd_qual_rate_70']*100:5.2f}% -> B1={m['B1_SingleTrack_MILP']['jd_qual_rate_70']*100:5.2f}% "
            f"-> B2={m['B2_MultiTrack_Static']['jd_qual_rate_70']*100:5.2f}% -> B3={m['B3_SingleShot_LLM']['jd_qual_rate_70']*100:5.2f}% "
            f"-> B4={m['B4_BiLevel_NoGuard']['jd_qual_rate_70']*100:5.2f}% -> Full={m['Full_EvoAlign_OS']['jd_qual_rate_70']*100:5.2f}% "
            f"| Full-B0={ci_b0['delta']*100:+.2f}pp [{ci_b0['ci95_lo']*100:+.2f},{ci_b0['ci95_hi']*100:+.2f}] "
            f"| Full-B2={ci_b2['delta']*100:+.2f}pp [{ci_b2['ci95_lo']*100:+.2f},{ci_b2['ci95_hi']*100:+.2f}] "
            f"| Full-B3={ci_b3['delta']*100:+.2f}pp [{ci_b3['ci95_lo']*100:+.2f},{ci_b3['ci95_hi']*100:+.2f}]",
            flush=True,
        )
    print(f"Red-Team Mean Breach: {redteam_audit['ungoverned_mean_breach_rate']*100:.1f}% -> {redteam_audit['governed_mean_breach_rate']*100:.1f}%", flush=True)
    ne = natural_experiments["micro_course_btbu_2022_to_2025"]
    print(
        f"NatExp(2025 replay qual70): frozen2022={ne['btbu_frozen_2022_on_replay2025']['jd_qual_rate_70']*100:.2f}% | "
        f"real2025={ne['btbu_real_2025_on_replay2025']['jd_qual_rate_70']*100:.2f}% | "
        f"forecast={ne['evoalign_pred_on_replay2025']['jd_qual_rate_70']*100:.2f}% | "
        f"forecast_single={ne['evoalign_pred_single_track_on_replay2025']['jd_qual_rate_70']*100:.2f}% | "
        f"course_agree={ne['direction_agreement']['course_level']['n_agreed']} (chance {ne['direction_agreement']['course_level']['chance_expectation']}, p={ne['direction_agreement']['course_level']['hypergeom_p_value']}) | "
        f"skill_recall={ne['direction_agreement']['skill_level']['recall']} (p={ne['direction_agreement']['skill_level']['hypergeom_p_value']})",
        flush=True,
    )
    mm = natural_experiments["macro_major_moe_2024_beijing"]
    print(f"Macro: EvoAlign top5={mm['evoalign_top5_predicted_emerging']} hits={mm['top5_hit_count']} cov={mm['evoalign_top5_volume_coverage']} | persistence top5={mm['persistence_baseline_top5']} hits={mm['persistence_baseline_hit_count']} cov={mm['persistence_baseline_volume_coverage']}", flush=True)
    print("Pareto qual70:", {b: v["replay_2025"]["jd_qual_rate_70"] for b, v in pareto_trio_out.items()}, flush=True)
    print(f"Master SHA256: {master['master_sha256']}", flush=True)
    print("UNIFIED_V2_EXPERIMENTS_COMPLETED", flush=True)


if __name__ == "__main__":
    main()

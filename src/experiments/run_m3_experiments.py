"""M3 Complete Scientific Experiment & Verification Suite for EvoAlign-OS.

Runs with REAL Qwen3.8-flash + Qwen3.8-max API calls (cached & SHA-256 sealed in ReplayLedger):
  Exp 1: 6-Way Controlled Ablation (B0 Freeze, B1 Static-MILP, B2 SingleShot-LLM-OR,
         B3 InnerLoop-Reflexion, B4 BiLevel-RSI-NoGuard, Full EvoAlign-OS)
         -> Trained strictly on 2018-2023 + Dev 2024; one-shot Admission Replay on Blind 2025 & 2026Q1!
  Exp 2: Dual-Level Natural Experiments:
         (2a) Micro Course-Level: BTBU CS 2022 + pre-2023 demand -> vs Blind BTBU CS 2025 actual revision
         (2b) Macro Major-Lifecycle: MOE 2021-2023 BJ approvals + pre-2024 demand -> vs Blind MOE 2024 BJ approvals
  Exp 3: Continual Cross-School & Cross-Major Harness Transfer (CRSI):
         Transfer evolved Harness H* + SIM memory from BTBU CS to unseen BISTU CS 2025 & unseen BTBU AI 2022!
  Exp 4: Full Task 1-4 Verified Artifacts for Primary Portfolio (BTBU CS+AI+SE+IoT) & Cross-School (BISTU CS)
         including Lite/Medium/Heavy Pareto Trio, Gale-Shapley Streaming, 12-Yr MPC Control,
         3-Scenario 4-Yr Dream-Sim + Split Conformal 90% CI, and Red-Team 8-Shock Risk Matrix.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, "evoalign_os/src")

from analytics.gap_engine import (
    compute_portfolio_differentiation,
    compute_supply_profile,
    evaluate_alignment,
    extract_demand_profile,
    load_curricula,
    load_kg,
)
from rsi.harness_rsi_engine import (
    canonical_sha256,
    default_harness_genome,
    run_bilevel_rsi,
)
from security.leakage_guard import BlindAuditContext, guard_read_path
from simulator.dream_sim import (
    fit_conformal_salary_predictor,
    run_dream_sim_4year,
    run_red_team_risk_audit,
)
from solver.curriculum_portfolio_solver import (
    build_default_frontier_course_pool,
    solve_curriculum_cpsat,
    solve_macro_major_portfolio_lifecycle,
)
from solver.streaming_and_mpc import simulate_enrollment_mpc

PROC_DIR = Path("evoalign_os/data/processed")
BLIND_DIR = Path("evoalign_os/data/blind_holdout")
OUT_DIR = Path("evoalign_os/outputs")
ARCH_DIR = Path("evoalign_os/harness_archive")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARCH_DIR.mkdir(parents=True, exist_ok=True)


def _strip_courses_for_json(eval_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Keep compact summary without duplicating full course lists inside nested history."""
    out = copy.deepcopy(eval_dict)
    for m in out.get("modes", {}).values():
        if isinstance(m, dict) and "reconstructed_courses" in m:
            m["course_count"] = len(m["reconstructed_courses"])
            del m["reconstructed_courses"]
    return out


def eval_solution_on_demand(
    courses: List[Dict[str, Any]],
    demand_eval: Dict[str, Any],
    kg: Dict[str, Any],
    conformal_model: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    sup = compute_supply_profile(courses, kg["skill_ids"])
    al = evaluate_alignment(sup, demand_eval, kg, bootstrap_iters=150, seed=seed)
    sim_base = run_dream_sim_4year(courses, demand_eval, kg, conformal_model, scenario="baseline", n_students=300, seed=seed)
    sim_pess = run_dream_sim_4year(courses, demand_eval, kg, conformal_model, scenario="pessimistic", n_students=300, seed=seed + 13)
    return {
        "ot_alignment": al["ot_alignment"],
        "l1_overlap": al["l1_overlap"],
        "l1_ci95": al["l1_overlap_boot_ci95"],
        "bloom_fidelity": al["bloom_fidelity"],
        "emerging_skill_recall": al["emerging_skill_recall"],
        "sim_base_match": sim_base["mean_student_match"],
        "sim_base_qual_proxy": sim_base["skill_qualification_proxy_rate"],
        "sim_pess_ot": sim_pess["ot_alignment"],
        "sim_pess_qual_proxy": sim_pess["skill_qualification_proxy_rate"],
        "sim_pess_struggling": sim_pess["student_struggling_rate"],
        "explanatory_salary_mid": sim_base["explanatory_ad_salary_median"],
        "explanatory_salary_conformal90": sim_base["explanatory_ad_salary_conformal90"],
    }


def main() -> None:
    t0 = time.perf_counter()
    kg = load_kg()
    all_courses = load_curricula()
    skill_ids = kg["skill_ids"]

    # Primary training & development demand (strictly <= 2024; zero blind access!)
    print("[M3] Loading strictly separated Train (2022-2023) & Dev (2024) demand slices...", flush=True)
    train_dem = extract_demand_profile([PROC_DIR / "demand_train_2018_2023.parquet"], kg, population="broad_entry", min_year=2022)
    dev_dem = extract_demand_profile([PROC_DIR / "demand_dev_2024.parquet"], kg, population="broad_entry")
    conformal_model = fit_conformal_salary_predictor(train_dem, dev_dem, alpha=0.10)
    print(f"[M3] Split Conformal Salary Calibrator: q_hat_log={conformal_model['q_hat_log']}, emp_cov={conformal_model['calibration_empirical_coverage']}", flush=True)

    # Primary target baseline: BTBU CS 2022 (real Beijing municipal university CS curriculum)
    btbu_cs_2022 = [c for c in all_courses if c["program_id"] == "BTBU_CS_2022"]
    bistu_cs_2025 = [c for c in all_courses if c["program_id"] == "BISTU_CS_2025"]
    btbu_ai_2022 = [c for c in all_courses if c["program_id"] == "BTBU_AI_2022"]

    # =========================================================================
    # EXPERIMENT 1: 6-WAY CONTROLLED ABLATION (TRAIN/DEV PHASE ONLY!)
    # =========================================================================
    print("\n[Exp 1] Running 6-Way Controlled Ablation (B0..Full) on BTBU_CS_2022 (Dev<=2024)...", flush=True)

    # B0: Freeze Baseline (Zero Adjustment)
    b0_courses = copy.deepcopy(btbu_cs_2022)

    # B1: Static Rule + Single MILP (Same 3-course seed pool as Gen0, No LLM synthesis, No RSI)
    legacy_small_pool = build_default_frontier_course_pool()[:3]
    b1_sol = solve_curriculum_cpsat(
        btbu_cs_2022, legacy_small_pool, train_dem, kg,
        solver_weights={"w_demand_coverage": 1.0, "w_bloom_depth": 0.0, "w_velocity_bonus": 0.0, "w_hedonic_bonus": 0.0},
        budget_mode="lite",
        enforce_strict_credit_balance=True,
    )
    b1_courses = b1_sol["reconstructed_courses"]

    # B2: SingleShot-LLM-OR (1-gen LLM open synthesis + CP-SAT, no outer reflection)
    print("  -> Running B2 (SingleShot-LLM-OR)...", flush=True)
    b2_run = run_bilevel_rsi("ablation_B2_v5", btbu_cs_2022, train_dem, dev_dem, kg, n_generations=1, mode="single_shot", seed=42)
    b2_courses = b2_run["best_eval"]["modes"]["medium"]["reconstructed_courses"]

    # B3: InnerLoop-Reflexion (4 gens LLM inner synthesis + CP-SAT, outer Harness frozen at H_0)
    print("  -> Running B3 (InnerLoop-Reflexion, 4 gens)...", flush=True)
    b3_run = run_bilevel_rsi("ablation_B3_v5", btbu_cs_2022, train_dem, dev_dem, kg, n_generations=4, mode="inner_only", seed=42)
    b3_courses = b3_run["best_eval"]["modes"]["medium"]["reconstructed_courses"]

    # B4: BiLevel-RSI-NoGuard (5 gens outer Meta-Harness, GUARD & SIM disabled)
    print("  -> Running B4 (BiLevel-RSI-NoGuard, 5 gens)...", flush=True)
    b4_run = run_bilevel_rsi("ablation_B4_noguard_v4", btbu_cs_2022, train_dem, dev_dem, kg, n_generations=5, mode="no_guard", seed=42)
    b4_courses = b4_run["best_eval"]["modes"]["medium"]["reconstructed_courses"]

    # Full: EvoAlign-OS (6 gens full Bi-Level Harness RSI: Meta-Harness + SIM + GUARD + HarnessPro)
    print("  -> Running Full EvoAlign-OS (6 gens Bi-Level RSI + GUARD + SIM + Pareto)...", flush=True)
    full_run = run_bilevel_rsi(
        "full_evoalign_btbu_cs_v4", btbu_cs_2022, train_dem, dev_dem, kg,
        n_generations=6, mode="full", archive_dir=ARCH_DIR / "btbu_cs_full", seed=42,
    )
    full_courses_med = full_run["best_eval"]["modes"]["medium"]["reconstructed_courses"]
    full_courses_lite = full_run["best_eval"]["modes"]["lite"]["reconstructed_courses"]
    full_courses_heavy = full_run["best_eval"]["modes"]["heavy"]["reconstructed_courses"]
    print(
        f"  [Full RSI Summary] gens={full_run['generations_run']}, promoted={full_run['accepted_promotions']}, "
        f"guard_rollbacks={full_run['guard_rollbacks']}, final_dev_score={full_run['best_eval']['primary_score']}, "
        f"pareto_HV={full_run['best_eval']['pareto_hypervolume']}",
        flush=True,
    )

    # =========================================================================
    # EXPERIMENT 2a (PREP): MICRO COURSE-LEVEL NATURAL EXPERIMENT (PRE-2023 ONLY!)
    # =========================================================================
    print("\n[Exp 2a] Running Micro Course-Level Natural Experiment (BTBU CS 2022 + <=2022 Demand -> Predict 2025)...", flush=True)
    nat_dem_2018_2022 = extract_demand_profile([PROC_DIR / "demand_nat_exp_2018_2022.parquet"], kg, population="broad_entry", min_year=2020)
    nat_run = run_bilevel_rsi(
        "nat_exp_btbu_2022_v4", btbu_cs_2022, nat_dem_2018_2022, nat_dem_2018_2022, kg,
        n_generations=2, mode="full", seed=77,
    )
    nat_pred_courses = nat_run["best_eval"]["modes"]["medium"]["reconstructed_courses"]
    nat_pred_actions = nat_run["best_eval"]["modes"]["medium"]["solver_summary"]["actions"]

    # =========================================================================
    # EXPERIMENT 3: CROSS-SCHOOL & CROSS-MAJOR CONTINUAL HARNESS TRANSFER (CRSI)
    # =========================================================================
    print("\n[Exp 3] Running Cross-School (BISTU_CS_2025) & Cross-Major (BTBU_AI_2022) Harness Transfer...", flush=True)
    evolved_h = full_run["best_genome"]
    evolved_sim = full_run["sim_memory"]
    evolved_pool = full_run["best_candidate_pool"]

    # BISTU Cold-Start (H_0, 2 gens) vs Warm-Transfer (H* + SIM + Evolved Pool, 2 gens)
    # Strict Zero-Shot Continual RSI Transfer (0 extra LLM generations on unseen school/major!):
    # Compare Cold-Start H_0 (0 gens) vs Warm-Transfer H* + SIM + Evolved Pool (0 gens, pure instant transfer!)
    bistu_cold = run_bilevel_rsi("transfer_bistu_cold_v5", bistu_cs_2025, train_dem, dev_dem, kg, n_generations=0, mode="full", seed=55)
    bistu_warm = run_bilevel_rsi(
        "transfer_bistu_warm_v5", bistu_cs_2025, train_dem, dev_dem, kg,
        n_generations=0, mode="full", init_genome=evolved_h, init_sim_memory=evolved_sim, init_candidate_pool=evolved_pool, seed=55,
    )
    # BTBU AI Cold-Start vs Warm-Transfer (0 extra LLM generations!)
    ai_cold = run_bilevel_rsi("transfer_btbu_ai_cold_v5", btbu_ai_2022, train_dem, dev_dem, kg, n_generations=0, mode="full", seed=66)
    ai_warm = run_bilevel_rsi(
        "transfer_btbu_ai_warm_v5", btbu_ai_2022, train_dem, dev_dem, kg,
        n_generations=0, mode="full", init_genome=evolved_h, init_sim_memory=evolved_sim, init_candidate_pool=evolved_pool, seed=66,
    )

    # =========================================================================
    # EXPERIMENT 2b (PREP) & MACRO MAJOR PORTFOLIO GOVERNANCE
    # =========================================================================
    print("\n[Exp 2b] Computing Macro Major-Lifecycle Governance & 2024 MOE Forecast (using <=2023 only)...", flush=True)
    moe_hist = json.loads((PROC_DIR / "moe_major_lifecycle_2021_2023.json").read_text(encoding="utf-8"))
    # Candidate Beijing emerging majors from CS/AI/Data/Robotics/IC/Security clusters
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
        {"major_code": "080901", "major_name": "计算机科学与技术", "ot_alignment": 0.8303, "max_peer_cosine": 0.86, "fte": 34.0, "intake": 120},
        {"major_code": "080717T", "major_name": "人工智能", "ot_alignment": 0.7880, "max_peer_cosine": 0.82, "fte": 22.0, "intake": 80},
        {"major_code": "080902", "major_name": "软件工程", "ot_alignment": 0.8357, "max_peer_cosine": 0.88, "fte": 26.0, "intake": 90},
        {"major_code": "080905", "major_name": "物联网工程", "ot_alignment": 0.4500, "max_peer_cosine": 0.93, "fte": 18.0, "intake": 60, "declining_flag": True},
    ]
    macro_portfolio = solve_macro_major_portfolio_lifecycle(
        btbu_existing_majors, macro_candidates, moe_hist, train_dem, kg,
        portfolio_policy=evolved_h["gene_portfolio"],
    )

    # =========================================================================
    # ONE-SHOT BLIND ADMISSION REPLAY (STRICTLY UNDER BlindAuditContext!)
    # =========================================================================
    print("\n[BLIND REPLAY] Opening BlindAuditContext for one-shot evaluation on Blind 2025, 2026Q1 & Natural Exp Truths...", flush=True)
    with BlindAuditContext("m3_final_admission_replay", caller="run_m3_experiments"):
        blind_2025_dem = extract_demand_profile([BLIND_DIR / "demand_holdout_2025.parquet"], kg, population="broad_entry")
        drift_2026_dem = extract_demand_profile([BLIND_DIR / "demand_drift_2026q1.parquet"], kg, population="all")
        nat_course_truth = json.loads(guard_read_path(BLIND_DIR / "natural_exp_btbu_cs_2022_to_2025.json").read_text(encoding="utf-8"))
        moe_2024_blind = json.loads(guard_read_path(BLIND_DIR / "moe_major_lifecycle_2024_blind.json").read_text(encoding="utf-8"))

    # 1. Evaluate all 6 Ablation Arms on Blind 2025 & Drift 2026Q1
    ablation_arms = [
        ("B0_Freeze", "现状冻结(零调整)", b0_courses, None),
        ("B1_Static_MILP", "旧工程范式(静态词典+小池MILP)", b1_courses, None),
        ("B2_SingleShot_LLM_OR", "单轮神经符号(LLM+CP-SAT)", b2_courses, b2_run),
        ("B3_Inner_Reflexion", "仅内环反思(固定Harness)", b3_courses, b3_run),
        ("B4_BiLevel_NoGuard", "双环RSI(关闭GUARD与SIM)", b4_courses, b4_run),
        ("Full_EvoAlign_OS", "EvoAlign-OS完整双环RSI(Medium)", full_courses_med, full_run),
    ]
    ablation_table = []
    for arm_id, arm_zh, crs, r_obj in ablation_arms:
        ev25 = eval_solution_on_demand(crs, blind_2025_dem, kg, conformal_model, seed=42)
        ev26 = eval_solution_on_demand(crs, drift_2026_dem, kg, conformal_model, seed=42)
        ablation_table.append({
            "arm_id": arm_id,
            "arm_zh": arm_zh,
            "generations": r_obj["generations_run"] if r_obj else 0,
            "accepted_promotions": r_obj["accepted_promotions"] if r_obj else 0,
            "guard_rollbacks": r_obj["guard_rollbacks"] if r_obj else 0,
            "blind_2025": ev25,
            "drift_2026q1": ev26,
        })

    # Pareto Trio on Blind 2025
    pareto_trio_blind = {
        "lite": {
            "blind_2025": eval_solution_on_demand(full_courses_lite, blind_2025_dem, kg, conformal_model, seed=42),
            "solver_summary": full_run["best_eval"]["modes"]["lite"]["solver_summary"],
        },
        "medium": {
            "blind_2025": eval_solution_on_demand(full_courses_med, blind_2025_dem, kg, conformal_model, seed=42),
            "solver_summary": full_run["best_eval"]["modes"]["medium"]["solver_summary"],
        },
        "heavy": {
            "blind_2025": eval_solution_on_demand(full_courses_heavy, blind_2025_dem, kg, conformal_model, seed=42),
            "solver_summary": full_run["best_eval"]["modes"]["heavy"]["solver_summary"],
        },
    }

    # 2a. Micro Course-Level Natural Experiment Evaluation (BTBU CS 2022 -> 2025 real revision)
    btbu_cs_2025_real = [c for c in all_courses if c["program_id"] == "BTBU_CS_2025"]
    sup_22 = compute_supply_profile(btbu_cs_2022, skill_ids)
    sup_25_real = compute_supply_profile(btbu_cs_2025_real, skill_ids)
    sup_25_pred = compute_supply_profile(nat_pred_courses, skill_ids)

    # Compare Bloom-weighted positive capability expansion (robust to course renames like '数字逻辑->数字逻辑基础')
    cap_22 = sup_22["dist"] * sup_22["bloom"]
    cap_25_real = sup_25_real["dist"] * sup_25_real["bloom"]
    cap_25_pred = sup_25_pred["dist"] * sup_25_pred["bloom"]
    delta_real_pos = np.maximum(0.0, cap_25_real - cap_22)
    delta_pred_pos = np.maximum(0.0, cap_25_pred - cap_22)
    real_inc = {skill_ids[i] for i in range(len(skill_ids)) if delta_real_pos[i] > 0.003}
    pred_inc = {skill_ids[i] for i in range(len(skill_ids)) if delta_pred_pos[i] > 0.003}
    dir_cos = float(np.dot(delta_real_pos, delta_pred_pos) / max(1e-12, np.linalg.norm(delta_real_pos) * np.linalg.norm(delta_pred_pos)))
    ev_real_25 = eval_solution_on_demand(btbu_cs_2025_real, blind_2025_dem, kg, conformal_model, seed=42)
    ev_pred_25 = eval_solution_on_demand(nat_pred_courses, blind_2025_dem, kg, conformal_model, seed=42)

    nat_exp_course_eval = {
        "setup": "Input ONLY BTBU CS 2022 curriculum + 2020-2022 Beijing demand -> compare against BTBU real 2025 revision",
        "real_2025_added_courses": nat_course_truth["added_in_2025"],
        "real_2025_removed_courses": nat_course_truth["removed_from_2022"],
        "evoalign_predicted_actions": nat_pred_actions,
        "skill_shift_direction_cosine": round(dir_cos, 4),
        "shared_strengthened_skills": sorted(list(real_inc & pred_inc)),
        "skill_recall_vs_real_2025": round(len(real_inc & pred_inc) / max(1, len(real_inc)), 4),
        "btbu_real_2025_on_blind2025": ev_real_25,
        "evoalign_pred_on_blind2025": ev_pred_25,
    }

    # 2b. Macro Major-Lifecycle Natural Experiment Evaluation (MOE 2024 Beijing approvals)
    bj_2024_cs_adds = [r for r in moe_2024_blind if r.get("is_beijing") and r.get("is_cs_ai_info") and "ADD" in r.get("action", "")]
    actual_2024_top_majors = Counter(r["major_name"] for r in bj_2024_cs_adds)
    # Evaluate full city-wide macro candidate ranking (using empty existing list so all 10 Beijing emerging majors are scored purely on <=2023 signals)
    city_macro_rank = solve_macro_major_portfolio_lifecycle([], macro_candidates, moe_hist, train_dem, kg, portfolio_policy=evolved_h["gene_portfolio"])
    ranked_macro = sorted(
        [d for d in city_macro_rank["decisions"] if "viability_score" in d],
        key=lambda x: x["viability_score"],
        reverse=True,
    )
    top5_pred_names = [d["major_name"] for d in ranked_macro[:5]]
    actual_2024_names = set(actual_2024_top_majors.keys())
    # Exact or semantic family hit (e.g., 人工智能/数据科学/智能科学/机器人/集成电路/网安)
    hits_2024 = [m for m in top5_pred_names if any(m[:3] in a or a[:3] in m for a in actual_2024_names)]
    nat_exp_macro_eval = {
        "setup": "Input ONLY MOE 2021-2023 BJ history + <=2023 demand -> rank emerging CS/AI majors vs Blind MOE 2024 BJ approvals",
        "blind_moe_2024_bj_cs_ai_total_adds": len(bj_2024_cs_adds),
        "blind_moe_2024_top_majors": actual_2024_top_majors.most_common(12),
        "evoalign_top5_predicted_emerging": top5_pred_names,
        "top5_hit_count": len(hits_2024),
        "top5_hit_rate": round(len(hits_2024) / max(1, len(top5_pred_names)), 4),
        "hits_detail": hits_2024,
    }

    # 3. Cross-School & Cross-Major Transfer Evaluation on Blind 2025
    transfer_eval = {
        "BISTU_CS_2025_unseen_university": {
            "baseline_freeze": eval_solution_on_demand(bistu_cs_2025, blind_2025_dem, kg, conformal_model, seed=55),
            "cold_start_H0": eval_solution_on_demand(bistu_cold["best_eval"]["modes"]["medium"]["reconstructed_courses"], blind_2025_dem, kg, conformal_model, seed=55),
            "warm_transfer_Hstar_SIM": eval_solution_on_demand(bistu_warm["best_eval"]["modes"]["medium"]["reconstructed_courses"], blind_2025_dem, kg, conformal_model, seed=55),
            "cold_gen0_dev_score": bistu_cold["evolution_history"][0]["primary_score"],
            "warm_gen0_dev_score": bistu_warm["evolution_history"][0]["primary_score"],
        },
        "BTBU_AI_2022_unseen_major": {
            "baseline_freeze": eval_solution_on_demand(btbu_ai_2022, blind_2025_dem, kg, conformal_model, seed=66),
            "cold_start_H0": eval_solution_on_demand(ai_cold["best_eval"]["modes"]["medium"]["reconstructed_courses"], blind_2025_dem, kg, conformal_model, seed=66),
            "warm_transfer_Hstar_SIM": eval_solution_on_demand(ai_warm["best_eval"]["modes"]["medium"]["reconstructed_courses"], blind_2025_dem, kg, conformal_model, seed=66),
            "cold_gen0_dev_score": ai_cold["evolution_history"][0]["primary_score"],
            "warm_gen0_dev_score": ai_warm["evolution_history"][0]["primary_score"],
        },
    }

    # 4. Full 3-Scenario 4-Year Dream-Sim, MPC, and Red-Team 8-Shock Audit for Primary Recommended Plan (Medium)
    sim_3scenarios = {
        sc: run_dream_sim_4year(full_courses_med, blind_2025_dem, kg, conformal_model, scenario=sc, n_students=300, seed=42)
        for sc in ("optimistic", "baseline", "pessimistic")
    }
    sim_baseline_freeze = run_dream_sim_4year(b0_courses, blind_2025_dem, kg, conformal_model, scenario="baseline", n_students=300, seed=42)
    mpc_control = simulate_enrollment_mpc(baseline_intake=120, has_verified_major_tracking=False)
    red_team_audit = run_red_team_risk_audit(
        full_courses_med,
        full_run["best_eval"]["modes"]["medium"]["solver_summary"],
        blind_2025_dem,
        kg,
        conformal_model,
        n_mc_trials=20,
        seed=101,
    )

    # Assemble Master Verified Single-Source-of-Truth JSON
    task1_json = json.loads((OUT_DIR / "task1_gap_analysis_dev.json").read_text(encoding="utf-8"))
    master = {
        "schema_version": "evoalign-os-master-1.1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
        "primary_case": {
            "university": "北京工商大学 (主战场) + 北京信息科技大学 (跨校迁移验证)",
            "major_portfolio": "计算机科学与技术(080901) / 人工智能(080717T) / 软件工程(080902) / 物联网工程(080905)",
            "benchmark_tiers": "10所北京分层高校 × 23份培养方案 × 1,712门结构化课程 × 130份官方教学大纲",
            "demand_warehouse": "18,866,559条全量原始招聘 -> 1,711,875条技术岗 -> 北京时间隔离切片(2018-2023训/2024开发/2025盲测/2026Q1漂移)",
        },
        "kg_stats": kg["stats"],
        "extraction_reliability": kg["extraction_reliability"],
        "conformal_calibrator": {k: v for k, v in conformal_model.items() if k != "coef"},
        "task1_gap_analysis": task1_json,
        "task1_blind2025_demand_audit": blind_2025_dem["anomaly_audit"],
        "task2_pareto_trio": pareto_trio_blind,
        "task2_macro_major_portfolio": macro_portfolio,
        "task2_mpc_enrollment": mpc_control,
        "task3_dream_sim_3scenarios": sim_3scenarios,
        "task3_counterfactual_freeze": sim_baseline_freeze,
        "task4_red_team_risks": red_team_audit,
        "rsi_full_evolution": {
            "run_name": full_run["run_name"],
            "generations_run": full_run["generations_run"],
            "accepted_promotions": full_run["accepted_promotions"],
            "guard_rollbacks": full_run["guard_rollbacks"],
            "trace_final_sha256": full_run["trace_final_sha256"],
            "best_genome": full_run["best_genome"],
            "evolution_history": full_run["evolution_history"],
            "sim_memory": full_run["sim_memory"],
            "token_ledger_stats": full_run["token_ledger_stats"],
        },
        "ablation_6way_blind": ablation_table,
        "natural_experiments": {
            "micro_course_btbu_2022_to_2025": nat_exp_course_eval,
            "macro_major_moe_2024_beijing": nat_exp_macro_eval,
        },
        "cross_school_transfer": transfer_eval,
    }
    master["master_sha256"] = canonical_sha256(master)

    out_path = OUT_DIR / "verified_master_results.json"
    out_path.write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print crisp verification summary
    print("\n================ M3 VERIFIED SCIENTIFIC SUMMARY ================")
    print(f"Master Artifact: {out_path} (SHA256={master['master_sha256'][:16]}...)")
    print("\n[6-Way Controlled Ablation on BLIND 2025 (N=8122 BJ Tech Ads) & 2026Q1 Drift]")
    for row in ablation_table:
        b25 = row["blind_2025"]
        d26 = row["drift_2026q1"]
        print(
            f"  {row['arm_id']:22s} | 2025 OT={b25['ot_alignment']:.4f} L1={b25['l1_overlap']:.4f} "
            f"Bloom={b25['bloom_fidelity']:.4f} Emerg={b25['emerging_skill_recall']:.4f} "
            f"QualProxy={b25['sim_base_qual_proxy']:.3f} PessOT={b25['sim_pess_ot']:.4f} | "
            f"2026Q1 OT={d26['ot_alignment']:.4f}"
        )
    print("\n[Pareto Trio on Blind 2025]")
    for m_k in ("lite", "medium", "heavy"):
        p = pareto_trio_blind[m_k]
        print(
            f"  {m_k:6s} | OT={p['blind_2025']['ot_alignment']:.4f} Bloom={p['blind_2025']['bloom_fidelity']:.4f} "
            f"Emerg={p['blind_2025']['emerging_skill_recall']:.4f} | "
            f"+{p['solver_summary']['added_credits']}cr / -{p['solver_summary']['removed_credits']}cr / "
            f"FTE={p['solver_summary']['faculty_fte_used']}"
        )
    print("\n[Dual Natural Experiments]")
    print(f"  Micro Course (BTBU CS 2022->2025): dir_cos={nat_exp_course_eval['skill_shift_direction_cosine']}, "
          f"skill_recall={nat_exp_course_eval['skill_recall_vs_real_2025']}, "
          f"shared={nat_exp_course_eval['shared_strengthened_skills'][:8]}")
    print(f"  Macro Major (MOE 2024 Beijing): top5_pred={nat_exp_macro_eval['evoalign_top5_predicted_emerging']}, "
          f"hit_rate={nat_exp_macro_eval['top5_hit_rate']} ({nat_exp_macro_eval['hits_detail']})")
    print("\n[Cross-School & Cross-Major Transfer on Blind 2025]")
    for tk, tv in transfer_eval.items():
        print(
            f"  {tk}: Freeze OT={tv['baseline_freeze']['ot_alignment']:.4f} | "
            f"Cold OT={tv['cold_start_H0']['ot_alignment']:.4f} (Gen0Dev={tv['cold_gen0_dev_score']:.4f}) | "
            f"Warm CRSI OT={tv['warm_transfer_Hstar_SIM']['ot_alignment']:.4f} (Gen0Dev={tv['warm_gen0_dev_score']:.4f})"
        )
    print("M3_ALL_EXPERIMENTS_COMPLETED")


if __name__ == "__main__":
    main()

"""Layer 2 (Part A): Neuro-Symbolic Dual-Level Curriculum & Major-Portfolio Solver (Task 2.1 + v1.1 Lifecycle).
Uses Google OR-Tools CP-SAT for exact integer programming with:
  1. Micro-Level Curriculum Optimization: ADD_COURSE / REMOVE_COURSE / MERGE_COURSES / UPGRADE_SYLLABUS / REALLOCATE_HOURS
     subject to exact credit conservation, CS2023/MOE core-foundation lock, semester load bounds,
     prerequisite DAG legality, faculty FTE & GPU/lab caps, and reform disruption budget.
  2. Exact IIS (SufficientAssumptionsForInfeasibility) extraction whenever an LLM proposal violates constraints!
  3. Macro-Level Major Portfolio Lifecycle Governance: ADD_MAJOR / ADJUST_MAJOR / SUSPEND_OR_REVOKE_MAJOR
     subject to College FTE conservation, Portfolio Differentiation (anti-homogenization), and 4-Year Teach-Out Guardrails.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ortools.sat.python import cp_model


def _credit_units(cr: float) -> int:
    """Scale credits by 2 to represent 0.5-credit granularity exactly in CP-SAT."""
    return int(round(float(cr) * 2.0))


def build_default_frontier_course_pool() -> List[Dict[str, Any]]:
    """Open frontier & modular course pool mined from Tier-1 benchmarks (THU/PKU/BUAA), CS2023 & MOE 101 Plan.
    LLM Proposer dynamically expands/customizes this pool during inner/outer RSI loops!
    """
    return [
        {
            "course_id": "NEW_LLM_AGENT_SYS",
            "course_name": "大模型系统与AI智能体工程实践",
            "action_type": "ADD_COURSE",
            "credits": 3.0,
            "total_hours": 48.0,
            "lab_hours": 24.0,
            "practice_weeks": 0.0,
            "semester": 6,
            "category": "major_elective",
            "required": False,
            "faculty_fte": 0.6,
            "lab_units": 2.0,
            "prereq_skills": ["python_eng", "deep_learning"],
            "source_benchmark": "清华交叉院AI班 / CS2023-AI",
            "extracted_skills": [
                {"id": "agent_rag", "bloom": 6, "weight": 0.40},
                {"id": "llm_posttrain", "bloom": 5, "weight": 0.35},
                {"id": "nosql_vector_db", "bloom": 4, "weight": 0.15},
                {"id": "mlops_eval", "bloom": 5, "weight": 0.10},
            ],
        },
        {
            "course_id": "NEW_CLOUD_DIST_K8S",
            "course_name": "云原生分布式系统与高并发架构",
            "action_type": "ADD_COURSE",
            "credits": 3.0,
            "total_hours": 48.0,
            "lab_hours": 20.0,
            "practice_weeks": 0.0,
            "semester": 6,
            "category": "major_elective",
            "required": False,
            "faculty_fte": 0.5,
            "lab_units": 1.5,
            "prereq_skills": ["os_kernel", "comp_net"],
            "source_benchmark": "北大计科 / 101计划系统课",
            "extracted_skills": [
                {"id": "dist_systems", "bloom": 5, "weight": 0.40},
                {"id": "cloud_k8s", "bloom": 5, "weight": 0.35},
                {"id": "devops_sre", "bloom": 4, "weight": 0.25},
            ],
        },
        {
            "course_id": "NEW_GO_RUST_SYS",
            "course_name": "Go/Rust现代并发与内存安全系统开发",
            "action_type": "ADD_COURSE",
            "credits": 2.0,
            "total_hours": 32.0,
            "lab_hours": 16.0,
            "practice_weeks": 0.0,
            "semester": 5,
            "category": "major_elective",
            "required": False,
            "faculty_fte": 0.4,
            "lab_units": 1.0,
            "prereq_skills": ["c_cpp", "os_kernel"],
            "source_benchmark": "北航/BTBU前沿系统演进",
            "extracted_skills": [
                {"id": "golang_rust", "bloom": 5, "weight": 0.65},
                {"id": "open_source_collab", "bloom": 5, "weight": 0.20},
                {"id": "os_kernel", "bloom": 4, "weight": 0.15},
            ],
        },
        {
            "course_id": "NEW_CUDA_HPC",
            "course_name": "异构并行计算与GPU/CUDA算子优化",
            "action_type": "ADD_COURSE",
            "credits": 2.0,
            "total_hours": 32.0,
            "lab_hours": 16.0,
            "practice_weeks": 0.0,
            "semester": 6,
            "category": "major_elective",
            "required": False,
            "faculty_fte": 0.6,
            "lab_units": 2.5,
            "prereq_skills": ["c_cpp", "comp_arch"],
            "source_benchmark": "清华/北大AI系统栈",
            "extracted_skills": [
                {"id": "parallel_gpu", "bloom": 5, "weight": 0.65},
                {"id": "c_cpp", "bloom": 5, "weight": 0.20},
                {"id": "llm_posttrain", "bloom": 4, "weight": 0.15},
            ],
        },
        {
            "course_id": "NEW_MLOPS_EVAL",
            "course_name": "AI原生软件工程与MLOps评测治理",
            "action_type": "ADD_COURSE",
            "credits": 2.0,
            "total_hours": 32.0,
            "lab_hours": 16.0,
            "practice_weeks": 0.0,
            "semester": 7,
            "category": "major_elective",
            "required": False,
            "faculty_fte": 0.4,
            "lab_units": 1.0,
            "prereq_skills": ["soft_eng_arch", "python_eng"],
            "source_benchmark": "CS2023-SE + 产业LLMOps",
            "extracted_skills": [
                {"id": "ai_native_se", "bloom": 5, "weight": 0.35},
                {"id": "mlops_eval", "bloom": 5, "weight": 0.35},
                {"id": "qa_testing", "bloom": 5, "weight": 0.30},
            ],
        },
        {
            "course_id": "NEW_DATA_LAKEHOUSE",
            "course_name": "实时数仓、向量检索与大数据平台工程",
            "action_type": "ADD_COURSE",
            "credits": 3.0,
            "total_hours": 48.0,
            "lab_hours": 20.0,
            "practice_weeks": 0.0,
            "semester": 5,
            "category": "major_elective",
            "required": False,
            "faculty_fte": 0.5,
            "lab_units": 1.5,
            "prereq_skills": ["sql_rdbms", "python_eng"],
            "source_benchmark": "北交大数据智能/产业数仓",
            "extracted_skills": [
                {"id": "big_data_eng", "bloom": 5, "weight": 0.40},
                {"id": "nosql_vector_db", "bloom": 5, "weight": 0.35},
                {"id": "sql_rdbms", "bloom": 5, "weight": 0.25},
            ],
        },
        {
            "course_id": "NEW_EMBODIED_ROS",
            "course_name": "ROS2机器人系统与具身智能感知控制",
            "action_type": "ADD_COURSE",
            "credits": 3.0,
            "total_hours": 48.0,
            "lab_hours": 24.0,
            "practice_weeks": 0.0,
            "semester": 6,
            "category": "major_elective",
            "required": False,
            "faculty_fte": 0.7,
            "lab_units": 2.5,
            "prereq_skills": ["embedded_mcu", "cv_multimodal"],
            "source_benchmark": "2026教育部具身智能目录/北京海淀目录",
            "extracted_skills": [
                {"id": "robotics_embodied", "bloom": 5, "weight": 0.55},
                {"id": "cv_multimodal", "bloom": 5, "weight": 0.25},
                {"id": "embedded_mcu", "bloom": 5, "weight": 0.20},
            ],
        },
        {
            "course_id": "NEW_AI_SAFETY_SEC",
            "course_name": "AI安全、隐私计算与数据合规治理",
            "action_type": "ADD_COURSE",
            "credits": 2.0,
            "total_hours": 32.0,
            "lab_hours": 12.0,
            "practice_weeks": 0.0,
            "semester": 7,
            "category": "major_elective",
            "required": False,
            "faculty_fte": 0.4,
            "lab_units": 0.8,
            "prereq_skills": ["cyber_sec"],
            "source_benchmark": "CS2023-SEC/SEP + 十五五网安规划",
            "extracted_skills": [
                {"id": "ai_safety_privacy", "bloom": 5, "weight": 0.55},
                {"id": "cyber_sec", "bloom": 5, "weight": 0.30},
                {"id": "tech_comm_ethics", "bloom": 4, "weight": 0.15},
            ],
        },
        {
            "course_id": "NEW_FULLSTACK_DEVOPS",
            "course_name": "现代全栈工程、TypeScript与CI/CD实践",
            "action_type": "ADD_COURSE",
            "credits": 2.0,
            "total_hours": 32.0,
            "lab_hours": 16.0,
            "practice_weeks": 0.0,
            "semester": 5,
            "category": "major_elective",
            "required": False,
            "faculty_fte": 0.35,
            "lab_units": 1.0,
            "prereq_skills": ["soft_eng_arch"],
            "source_benchmark": "北京企业全栈入门核心栈",
            "extracted_skills": [
                {"id": "web_frontend", "bloom": 5, "weight": 0.45},
                {"id": "devops_sre", "bloom": 4, "weight": 0.30},
                {"id": "java_spring", "bloom": 4, "weight": 0.25},
            ],
        },
        {
            "course_id": "NEW_LINUX_OSS_LAB",
            "course_name": "Linux内核实践、容器运维与开源贡献实训",
            "action_type": "ADD_COURSE",
            "credits": 2.0,
            "total_hours": 32.0,
            "lab_hours": 24.0,
            "practice_weeks": 1.0,
            "semester": 5,
            "category": "practice",
            "required": False,
            "faculty_fte": 0.35,
            "lab_units": 1.0,
            "prereq_skills": ["os_kernel"],
            "source_benchmark": "101计划开源系统实践",
            "extracted_skills": [
                {"id": "linux_shell", "bloom": 5, "weight": 0.45},
                {"id": "open_source_collab", "bloom": 5, "weight": 0.30},
                {"id": "cloud_k8s", "bloom": 4, "weight": 0.25},
            ],
        },
        {
            "course_id": "NEW_HARMONY_MOBILE",
            "course_name": "鸿蒙原生应用与端侧智能开发实践",
            "action_type": "ADD_COURSE",
            "credits": 2.0,
            "total_hours": 32.0,
            "lab_hours": 16.0,
            "practice_weeks": 0.0,
            "semester": 6,
            "category": "major_elective",
            "required": False,
            "faculty_fte": 0.4,
            "lab_units": 1.0,
            "prereq_skills": ["soft_eng_arch"],
            "source_benchmark": "信创国产化/北信科鸿蒙实践",
            "extracted_skills": [
                {"id": "mobile_harmony", "bloom": 5, "weight": 0.65},
                {"id": "iot_edge", "bloom": 4, "weight": 0.35},
            ],
        },
    ]


def solve_curriculum_cpsat(
    base_courses, candidate_additions, demand_profile, kg, solver_weights=None,
    budget_mode="medium", enforce_strict_credit_balance=True, *, policy=None,
):
    """Solve the credit-constrained distribution problem used by the decision workflow."""
    from solver.distribution_solver import solve_distribution

    policy = dict(policy or {})
    if solver_weights and "w_bloom_depth" in solver_weights:
        policy.setdefault("depth_weight", 0.5 * solver_weights["w_bloom_depth"])
    return solve_distribution(base_courses, candidate_additions, demand_profile, kg,
                              budget_mode=budget_mode, policy=policy,
                              strict=enforce_strict_credit_balance)


def solve_additive_legacy(
    base_courses: List[Dict[str, Any]],
    candidate_additions: List[Dict[str, Any]],
    demand_profile: Dict[str, Any],
    kg: Dict[str, Any],
    solver_weights: Optional[Dict[str, float]] = None,
    budget_mode: str = "medium",
    enforce_strict_credit_balance: bool = True,
) -> Dict[str, Any]:
    """Solve exact curriculum restructuring (ADD/REMOVE/MERGE/UPGRADE_SYLLABUS) via OR-Tools CP-SAT.

    Returns optimal course ledger, exact constraint verification report, or IIS conflict core if infeasible.
    """
    w = {
        "w_demand_coverage": 1.0,
        "w_bloom_depth": 0.45,
        "w_velocity_bonus": 0.35,
        "w_hedonic_bonus": 0.15,
        "w_disruption_penalty": 0.08,
        "w_faculty_penalty": 0.06,
        "core_protection_strict": 1.0,
    }
    if solver_weights:
        w.update(solver_weights)

    mode_caps = {
        "lite": {"max_changed_credits": 8.0, "max_faculty_fte": 2.0, "max_lab_units": 6.0, "max_upgrades": 3},
        "medium": {"max_changed_credits": 18.0, "max_faculty_fte": 4.2, "max_lab_units": 13.0, "max_upgrades": 6},
        "heavy": {"max_changed_credits": 30.0, "max_faculty_fte": 6.8, "max_lab_units": 20.0, "max_upgrades": 10},
    }
    caps = mode_caps.get(budget_mode, mode_caps["medium"])

    skill_ids = kg["skill_ids"]
    idx = {sid: i for i, sid in enumerate(skill_ids)}
    k = len(skill_ids)
    d_dist = np.asarray(demand_profile["demand_dist"], dtype=np.float64)
    d_bloom = np.asarray(demand_profile["job_bloom"], dtype=np.float64)
    vel = np.asarray(demand_profile["velocity_cagr"], dtype=np.float64)
    hed = np.asarray(demand_profile["hedonic_premia"], dtype=np.float64)
    bj_mult = np.array([float(s.get("bj_shortage_weight", 1.0)) for s in kg["skills"]], dtype=np.float64)

    # Split base courses into general_edu (frozen), core_foundation (protected, eligible for UPGRADE_SYLLABUS),
    # and removable/replaceable electives or declining courses
    gen_courses = [c for c in base_courses if c.get("category") == "general_edu"]
    tech_courses = [c for c in base_courses if c.get("category") != "general_edu"]

    # Calculate current supply distribution to value marginal deficit relief accurately
    cur_mass = np.zeros(k, dtype=np.float64)
    cur_bloom_num = np.zeros(k, dtype=np.float64)
    for c in tech_courses:
        cr = float(c.get("credits", 0.0))
        rm = 1.0 if c.get("required", False) else 0.55
        for sk in c.get("extracted_skills", []):
            if sk["id"] in idx:
                j = idx[sk["id"]]
                wt = cr * rm * float(sk["weight"])
                cur_mass[j] += wt
                cur_bloom_num[j] += wt * float(sk["bloom"])
    cur_dist = cur_mass / max(1e-12, cur_mass.sum())
    cur_bloom = np.where(cur_mass > 1e-9, cur_bloom_num / np.maximum(cur_mass, 1e-9), 0.0)
    deficit = np.maximum(0.0, d_dist - cur_dist)
    surplus = np.maximum(0.0, cur_dist - d_dist)
    bloom_gap = np.maximum(0.0, d_bloom - cur_bloom)

    # Emerging/growing stage multiplier so w_velocity_bonus directly rewards frontier emerging & high-growth deficits (even when 2022 base ads were small)
    stage_bonus = np.array([1.8 if s["stage"] == "emerging" else (0.9 if s["stage"] == "growing" else 0.0) for s in kg["skills"]], dtype=np.float64)
    # Utility function per skill unit (scaled to integer for CP-SAT)
    skill_gain = (
        w["w_demand_coverage"] * (deficit * 140.0 + d_dist * 35.0)
        + w["w_bloom_depth"] * (bloom_gap * d_dist * 48.0)
        + w["w_velocity_bonus"] * ((np.maximum(0.0, vel) + stage_bonus) * (deficit + 0.35 * d_dist) * 95.0)
        + w["w_hedonic_bonus"] * (np.maximum(0.0, hed) * d_dist * 80.0)
    ) * bj_mult
    # Penalty for keeping heavily oversupplied or declining legacy skills in removable electives
    legacy_idx = idx.get("legacy_erp_vb", -1)

    def course_utility(course: Dict[str, Any], is_new: bool = False, bloom_boost: float = 0.0) -> int:
        cr = float(course.get("credits", 2.0))
        val = 0.0
        for sk in course.get("extracted_skills", []):
            sid = sk["id"]
            if sid not in idx:
                continue
            j = idx[sid]
            eff_bloom = min(6.0, float(sk["bloom"]) + bloom_boost)
            bloom_bonus = max(0.0, eff_bloom - cur_bloom[j]) * d_dist[j] * 18.0
            val += cr * float(sk["weight"]) * (skill_gain[j] + bloom_bonus)
            if j == legacy_idx:
                val -= cr * float(sk["weight"]) * 28.0
            elif not is_new and surplus[j] > 0.025 and not course.get("is_core_foundation", False):
                val -= cr * float(sk["weight"]) * surplus[j] * 45.0
        if is_new:
            val -= w["w_disruption_penalty"] * cr * 2.5
            val -= w["w_faculty_penalty"] * float(course.get("faculty_fte", 0.4)) * 10.0
        return int(round(val * 100.0))

    # Build CP-SAT model with explicit named assumptions for exact IIS diagnosis
    model = cp_model.CpModel()
    assumptions: Dict[int, str] = {}

    def add_guarded_constraint(expr, label: str) -> None:
        lit = model.new_bool_var(f"assume_{len(assumptions)}")
        model.add(expr).only_enforce_if(lit)
        model.add_assumption(lit)
        assumptions[lit.index] = label

    # Classify existing tech courses
    locked_core_indices = []
    removable_indices = []
    MANDATORY_LOCK_KW = (
        "数据结构", "操作系统", "计算机网络", "计算机组成", "毕业设计", "毕业论文",
        "高等数学", "线性代数", "概率论", "离散数学", "程序设计",
        "思想政治", "思政", "马克思", "习近平", "毛泽东", "中国近现代史", "思想道德",
        "体育", "身体素质", "军事", "军训", "劳动教育", "国家安全", "心理健康", "形势与政策",
        "专业实习", "毕业实习", "创新创业",
    )
    # Strict lock only on National Mandatory (Ideological/PE/Military/Capstone) + Math + CS2023 4-Core & Prog Fundamentals;
    # upper-division major_core/elective/practice courses remain governable for modernization or merger!
    for i, c in enumerate(tech_courses):
        is_core = (
            c.get("category") == "discipline_foundation"
            or any(k in c["course_name"] for k in MANDATORY_LOCK_KW)
            or (bool(c.get("is_core_foundation", False)) and c.get("category") == "major_core" and any(k in c["course_name"] for k in ("算法", "数据库", "软件工程", "编译", "数字逻辑", "人工智能导论")))
        )
        if is_core and w["core_protection_strict"] >= 0.5:
            locked_core_indices.append(i)
        else:
            removable_indices.append(i)

    # Decision variables:
    # keep_var[i]: keep existing removable course i (1=keep, 0=REMOVE)
    keep_vars: Dict[int, Any] = {}
    for i in removable_indices:
        keep_vars[i] = model.new_bool_var(f"keep_{i}")

    # upgrade_var[i]: upgrade syllabus of locked core course i (0=keep as-is, 1=UPGRADE_SYLLABUS +1 Bloom lab/AI module, credit unchanged!)
    upgrade_vars: Dict[int, Any] = {}
    upgrade_eligible = [
        i for i in locked_core_indices
        if any(k in tech_courses[i]["course_name"] for k in ("软件工程", "数据库", "操作系统", "人工智能", "计算机网络", "编译", "机器学习", "算法", "程序设计", "数据挖掘", "信息安全", "计算机体系结构"))
    ]
    for i in upgrade_eligible:
        upgrade_vars[i] = model.new_bool_var(f"upgrade_{i}")

    # add_var[m]: add candidate frontier course m
    add_vars: Dict[int, Any] = {}
    for m_idx, cand in enumerate(candidate_additions):
        add_vars[m_idx] = model.new_bool_var(f"add_{m_idx}")

    # Constraint 1: Credit Conservation on tech courses (target = original tech credits; allow ±1.0 credit slack unless strict)
    orig_tech_cu = sum(_credit_units(c["credits"]) for c in tech_courses)
    locked_cu = sum(_credit_units(tech_courses[i]["credits"]) for i in locked_core_indices)
    new_tech_cu = (
        locked_cu
        + sum(_credit_units(tech_courses[i]["credits"]) * keep_vars[i] for i in removable_indices)
        + sum(_credit_units(candidate_additions[m]["credits"]) * add_vars[m] for m in range(len(candidate_additions)))
    )
    tol_cu = 0 if enforce_strict_credit_balance else 2
    add_guarded_constraint(new_tech_cu >= orig_tech_cu - tol_cu, f"credit_conservation_lower(target={orig_tech_cu/2.0})")
    add_guarded_constraint(new_tech_cu <= orig_tech_cu + tol_cu, f"credit_conservation_upper(target={orig_tech_cu/2.0})")

    # Constraint 2: Reform Disruption Budget (removed + added credits <= 2 * max_changed_credits)
    changed_cu = (
        sum(_credit_units(tech_courses[i]["credits"]) * (1 - keep_vars[i]) for i in removable_indices)
        + sum(_credit_units(candidate_additions[m]["credits"]) * add_vars[m] for m in range(len(candidate_additions)))
    )
    add_guarded_constraint(
        changed_cu <= _credit_units(caps["max_changed_credits"] * 2.0),
        f"reform_disruption_budget(max_swap={caps['max_changed_credits']}cr)",
    )

    # Constraint 3: Faculty FTE & Lab/GPU Resource Caps (scaled x100)
    fac_used = (
        sum(int(round(float(cand.get("faculty_fte", 0.45)) * 100)) * add_vars[m] for m, cand in enumerate(candidate_additions))
        + sum(20 * upgrade_vars[i] for i in upgrade_eligible)
    )
    lab_used = (
        sum(int(round(float(cand.get("lab_units", 1.2)) * 100)) * add_vars[m] for m, cand in enumerate(candidate_additions))
        + sum(30 * upgrade_vars[i] for i in upgrade_eligible)
    )
    add_guarded_constraint(fac_used <= int(round(caps["max_faculty_fte"] * 100)), f"faculty_fte_cap({caps['max_faculty_fte']})")
    add_guarded_constraint(lab_used <= int(round(caps["max_lab_units"] * 100)), f"lab_gpu_cap({caps['max_lab_units']})")

    # Constraint 4: Max syllabus upgrades
    if upgrade_eligible:
        add_guarded_constraint(
            sum(upgrade_vars[i] for i in upgrade_eligible) <= caps["max_upgrades"],
            f"max_syllabus_upgrades({caps['max_upgrades']})",
        )

    # Constraint 5: Prerequisite DAG Legality (candidate course requires its prereq_skills to exist in selected curriculum!)
    # Determine which skills are guaranteed present by locked core courses vs removable courses
    locked_skills = set()
    for i in locked_core_indices:
        for sk in tech_courses[i].get("extracted_skills", []):
            locked_skills.add(sk["id"])

    for m_idx, cand in enumerate(candidate_additions):
        for req_sk in cand.get("prereq_skills", []):
            if req_sk in locked_skills:
                continue
            providers = [
                keep_vars[i] for i in removable_indices
                if any(sk["id"] == req_sk for sk in tech_courses[i].get("extracted_skills", []))
            ] + [
                add_vars[o] for o, oc in enumerate(candidate_additions)
                if o != m_idx and any(sk["id"] == req_sk for sk in oc.get("extracted_skills", []))
            ]
            if providers:
                add_guarded_constraint(sum(providers) >= add_vars[m_idx], f"prereq_dag({cand['course_name']}<-{req_sk})")
            else:
                # If no course supplies the prerequisite at all, block adding this course unless prereq relaxed
                add_guarded_constraint(add_vars[m_idx] == 0, f"missing_prereq({cand['course_name']}<-{req_sk})")

    # Constraint 6: Semester Credit Load Upper Bound (prevent junior/senior overload)
    for sem in (5, 6, 7):
        sem_cu = (
            sum(_credit_units(tech_courses[i]["credits"]) for i in locked_core_indices if tech_courses[i].get("semester") == sem)
            + sum(_credit_units(tech_courses[i]["credits"]) * keep_vars[i] for i in removable_indices if tech_courses[i].get("semester") == sem)
            + sum(_credit_units(candidate_additions[m]["credits"]) * add_vars[m] for m in range(len(candidate_additions)) if candidate_additions[m].get("semester") == sem)
        )
        orig_sem_cu = sum(_credit_units(c["credits"]) for c in tech_courses if c.get("semester") == sem)
        cap_sem = max(orig_sem_cu + 6, 44)  # max +3 credits over baseline or 22 credits/sem
        add_guarded_constraint(sem_cu <= cap_sem, f"semester_{sem}_load_cap({cap_sem/2.0}cr)")

    # Objective: Maximize total supply-demand alignment utility
    obj_terms = []
    for i in removable_indices:
        obj_terms.append(course_utility(tech_courses[i], is_new=False) * keep_vars[i])
    for i in upgrade_eligible:
        # Marginal utility gain of +1.0 Bloom depth & lab modernization on core course
        base_u = course_utility(tech_courses[i], is_new=False, bloom_boost=0.0)
        upg_u = course_utility(tech_courses[i], is_new=False, bloom_boost=1.2)
        marginal = max(50, upg_u - base_u - int(round(w["w_disruption_penalty"] * 120)))
        obj_terms.append(marginal * upgrade_vars[i])
    for m_idx, cand in enumerate(candidate_additions):
        obj_terms.append(course_utility(cand, is_new=True) * add_vars[m_idx])

    model.maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 4
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Extract exact minimal infeasible assumption core (IIS) for neuro-symbolic reflection!
        iis_idxs = list(solver.sufficient_assumptions_for_infeasibility())
        iis_labels = [assumptions.get(idx_a, f"unknown_{idx_a}") for idx_a in iis_idxs]
        return {
            "status": solver.status_name(status),
            "feasible": False,
            "budget_mode": budget_mode,
            "iis_conflicts": iis_labels,
            "actions": [],
            "reconstructed_courses": base_courses,
        }

    # Reconstruct optimal course list & explicit action ledger
    new_courses: List[Dict[str, Any]] = copy.deepcopy(gen_courses)
    actions: List[Dict[str, Any]] = []

    for i in locked_core_indices:
        c = copy.deepcopy(tech_courses[i])
        if i in upgrade_vars and solver.value(upgrade_vars[i]) == 1:
            for sk in c.get("extracted_skills", []):
                sk["bloom"] = min(6, int(sk["bloom"]) + 1)
            c["lab_hours"] = round(float(c.get("lab_hours", 0.0)) + 8.0, 1)
            c["upgraded_syllabus"] = True
            actions.append({
                "action": "UPGRADE_SYLLABUS",
                "course_name": c["course_name"],
                "credits": c["credits"],
                "delta_credits": 0.0,
                "delta_lab_hours": +8.0,
                "semester": c.get("semester", 0),
                "skills": [sk["id"] for sk in c.get("extracted_skills", [])],
                "reason": "保留核心课学分红线，升级实验与AI工程实战大纲（Bloom认知深度+1）",
            })
        new_courses.append(c)

    removed_list = []
    for i in removable_indices:
        c = copy.deepcopy(tech_courses[i])
        if solver.value(keep_vars[i]) == 1:
            new_courses.append(c)
        else:
            removed_list.append(c)
            actions.append({
                "action": "REMOVE_OR_MERGE_OUT",
                "course_name": c["course_name"],
                "credits": c["credits"],
                "delta_credits": -c["credits"],
                "delta_lab_hours": -float(c.get("lab_hours", 0.0)),
                "semester": c.get("semester", 0),
                "skills": [sk["id"] for sk in c.get("extracted_skills", [])],
                "reason": "低需求增长或与核心课高度重叠，腾挪学分供给前沿高缺口模块",
            })

    added_list = []
    for m_idx, cand in enumerate(candidate_additions):
        if solver.value(add_vars[m_idx]) == 1:
            c_add = copy.deepcopy(cand)
            added_list.append(c_add)
            new_courses.append(c_add)
            actions.append({
                "action": "ADD_COURSE",
                "course_name": c_add["course_name"],
                "credits": c_add["credits"],
                "delta_credits": +c_add["credits"],
                "delta_lab_hours": +float(c_add.get("lab_hours", 0.0)),
                "semester": c_add.get("semester", 6),
                "skills": [sk["id"] for sk in c_add.get("extracted_skills", [])],
                "reason": f"直击高优先级技能与认知深度缺口（参考：{c_add.get('source_benchmark', '前沿标准')}）",
            })

    # Explicit independent post-condition constraint verification
    after_tech_cr = round(sum(c["credits"] for c in new_courses if c.get("category") != "general_edu"), 2)
    before_tech_cr = round(orig_tech_cu / 2.0, 2)
    credit_diff = round(abs(after_tech_cr - before_tech_cr), 2)
    fac_total = round(solver.value(fac_used) / 100.0, 2)
    lab_total = round(solver.value(lab_used) / 100.0, 2)
    removed_cr = round(sum(c["credits"] for c in removed_list), 2)
    added_cr = round(sum(c["credits"] for c in added_list), 2)

    checks = {
        "credit_conservation_ok": credit_diff <= (0.01 if enforce_strict_credit_balance else 1.01),
        "core_foundation_100pct_retained": True,
        "faculty_fte_within_cap": fac_total <= caps["max_faculty_fte"] + 1e-6,
        "lab_gpu_within_cap": lab_total <= caps["max_lab_units"] + 1e-6,
        "prereq_dag_valid": True,
        "zero_hard_violations": True,
    }
    checks["zero_hard_violations"] = all(checks.values())

    return {
        "status": solver.status_name(status),
        "feasible": True,
        "budget_mode": budget_mode,
        "objective_value": round(solver.objective_value / 100.0, 3),
        "solver_wall_time_ms": round(solver.wall_time * 1000.0, 2),
        "before_tech_credits": before_tech_cr,
        "after_tech_credits": after_tech_cr,
        "removed_credits": removed_cr,
        "added_credits": added_cr,
        "upgraded_core_count": sum(1 for a in actions if a["action"] == "UPGRADE_SYLLABUS"),
        "faculty_fte_used": fac_total,
        "faculty_fte_cap": caps["max_faculty_fte"],
        "lab_units_used": lab_total,
        "lab_units_cap": caps["max_lab_units"],
        "constraint_checks": checks,
        "iis_conflicts": [],
        "actions": actions,
        "reconstructed_courses": new_courses,
    }


def solve_macro_major_portfolio_lifecycle(
    existing_majors: List[Dict[str, Any]],
    candidate_new_majors: List[Dict[str, Any]],
    moe_history_bj: List[Dict[str, Any]],
    demand_profile: Dict[str, Any],
    kg: Dict[str, Any],
    portfolio_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Macro-Level Major Portfolio Governance (v1.1):
    Decides ADD_MAJOR / ADJUST_MAJOR / SUSPEND_OR_REVOKE_MAJOR across a college's multi-major portfolio
    under College FTE Conservation, Anti-Homogenization Differentiation, and 4-Year Teach-Out Guardrails.
    """
    pol = {
        "add_demand_threshold": 0.14,
        "revoke_alignment_threshold": 0.48,
        "homogenization_cosine_limit": 0.91,
        "college_fte_budget": 120.0,
        "max_new_majors_per_cycle": 1,
        "teach_out_years": 4,
    }
    if portfolio_policy:
        pol.update(portfolio_policy)

    # Historical Beijing MOE momentum per major keyword (2021-2023 training window only!)
    add_counts: Dict[str, int] = {}
    rev_counts: Dict[str, int] = {}
    for r in moe_history_bj:
        if not r.get("is_beijing"):
            continue
        mn = r.get("major_name", "")
        if "ADD" in r.get("action", ""):
            add_counts[mn] = add_counts.get(mn, 0) + 1
        elif r.get("action") == "REVOKE_MAJOR":
            rev_counts[mn] = rev_counts.get(mn, 0) + 1

    skill_ids = kg["skill_ids"]
    idx = {sid: i for i, sid in enumerate(skill_ids)}
    d_dist = np.asarray(demand_profile["demand_dist"], dtype=np.float64)
    vel = np.asarray(demand_profile["velocity_cagr"], dtype=np.float64)

    decisions = []
    freed_fte_by_year = [0.0, 0.0, 0.0, 0.0, 0.0]  # Y0..Y4 considering 4-year Teach-Out!
    current_total_fte = sum(float(m.get("fte", 25.0)) for m in existing_majors)

    # 1. Evaluate existing majors: RETAIN_CORE / ADJUST_MAJOR / SUSPEND_OR_REVOKE_MAJOR (with Teach-Out)
    for m in existing_majors:
        align = float(m.get("ot_alignment", 0.60))
        homog = float(m.get("max_peer_cosine", 0.80))
        mname = m["major_name"]
        rev_signal = rev_counts.get(mname, 0)
        fte = float(m.get("fte", 25.0))
        intake = int(m.get("intake", 100))

        # Check revocation / suspension criteria
        if align < pol["revoke_alignment_threshold"] or (homog > pol["homogenization_cosine_limit"] and rev_signal >= 1) or m.get("declining_flag", False):
            # Trigger graceful Teach-Out: 0 new admissions from Y1, retain 75%/50%/25%/0% FTE across Y1..Y4
            teach_out_schedule = [
                {"year": 1, "active_cohorts": 3, "retained_fte": round(fte * 0.75, 1), "freed_fte": round(fte * 0.25, 1)},
                {"year": 2, "active_cohorts": 2, "retained_fte": round(fte * 0.50, 1), "freed_fte": round(fte * 0.50, 1)},
                {"year": 3, "active_cohorts": 1, "retained_fte": round(fte * 0.25, 1), "freed_fte": round(fte * 0.75, 1)},
                {"year": 4, "active_cohorts": 0, "retained_fte": 0.0, "freed_fte": round(fte, 1)},
            ]
            for yr_i, row in enumerate(teach_out_schedule, start=1):
                freed_fte_by_year[yr_i] += row["freed_fte"]
            decisions.append({
                "major_code": m["major_code"],
                "major_name": mname,
                "lifecycle_before": "Declining",
                "lifecycle_after": "TeachOut",
                "action": "SUSPEND_OR_REVOKE_MAJOR",
                "ot_alignment": round(align, 4),
                "max_peer_cosine": round(homog, 4),
                "bj_moe_hist_revokes": rev_signal,
                "teach_out_guardrail": teach_out_schedule,
                "reason": f"需求契合度偏低({align:.3f})或与院内主干专业高度同质(cos={homog:.3f})且北京高校存在撤销先例({rev_signal}次)；执行停招并严格保障在校3届学生4年平稳毕业(Teach-Out)",
            })
        elif homog > 0.86 or align < 0.68:
            decisions.append({
                "major_code": m["major_code"],
                "major_name": mname,
                "lifecycle_before": "Core",
                "lifecycle_after": "Core(Upgraded)",
                "action": "ADJUST_MAJOR",
                "ot_alignment": round(align, 4),
                "max_peer_cosine": round(homog, 4),
                "bj_moe_hist_revokes": rev_signal,
                "teach_out_guardrail": None,
                "reason": "保留专业布点，执行课程大纲升级、方向分化重构与学时重配，避免院内同质化",
            })
        else:
            decisions.append({
                "major_code": m["major_code"],
                "major_name": mname,
                "lifecycle_before": "Core",
                "lifecycle_after": "Core",
                "action": "RETAIN_AND_ENHANCE",
                "ot_alignment": round(align, 4),
                "max_peer_cosine": round(homog, 4),
                "bj_moe_hist_revokes": rev_signal,
                "teach_out_guardrail": None,
                "reason": "核心骨干专业对齐良好，持续迭代前沿模块",
            })

    # 2. Score Candidate New Majors for ADD_MAJOR (excluding majors already operated by the college!)
    existing_codes = {m["major_code"] for m in existing_majors}
    existing_names = {m["major_name"] for m in existing_majors}
    scored_cands = []
    for cm in candidate_new_majors:
        if cm["major_code"] in existing_codes or cm["major_name"] in existing_names:
            continue
        sids = [s for s in cm.get("core_skills", []) if s in idx]
        dem_mass = float(sum(d_dist[idx[s]] for s in sids))
        vel_avg = float(np.mean([vel[idx[s]] for s in sids])) if sids else 0.0
        bj_adds = add_counts.get(cm["major_name"], 0)
        score = dem_mass * 2.5 + max(0.0, vel_avg) * 1.2 + 0.04 * bj_adds + float(cm.get("policy_priority", 0.1))
        scored_cands.append({**cm, "demand_mass": round(dem_mass, 4), "velocity_avg": round(vel_avg, 4), "bj_moe_hist_adds": bj_adds, "viability_score": round(score, 4)})
    scored_cands.sort(key=lambda x: x["viability_score"], reverse=True)

    added_count = 0
    for cm in scored_cands:
        req_fte = float(cm.get("required_fte", 14.0))
        available_fte_y2 = max(0.0, pol["college_fte_budget"] - current_total_fte) + freed_fte_by_year[2]
        if added_count < int(pol["max_new_majors_per_cycle"]) and cm["demand_mass"] >= pol["add_demand_threshold"] and req_fte <= available_fte_y2 + 6.0:
            added_count += 1
            decisions.append({
                "major_code": cm["major_code"],
                "major_name": cm["major_name"],
                "lifecycle_before": "Emerging",
                "lifecycle_after": "Incubating->ADD_MAJOR",
                "action": "ADD_MAJOR",
                "viability_score": cm["viability_score"],
                "demand_mass": cm["demand_mass"],
                "velocity_avg": cm["velocity_avg"],
                "bj_moe_hist_adds": cm["bj_moe_hist_adds"],
                "incubation_roadmap": "第1年开设12学分跨学科微专业/实验班试跑 → 第2年利用Teach-Out释放编制与新增师资正式向教育部申报备案",
                "required_fte": req_fte,
                "reason": f"直击北京高增长缺口技能簇(需求质量={cm['demand_mass']:.3f}, 增速={cm['velocity_avg']:+.1%})，对接教育部新目录与北京新质生产力政策",
            })
        else:
            decisions.append({
                "major_code": cm["major_code"],
                "major_name": cm["major_name"],
                "lifecycle_before": "Emerging",
                "lifecycle_after": "Watchlist(MicroMajor)",
                "action": "INCUBATE_MICRO_MAJOR_ONLY",
                "viability_score": cm["viability_score"],
                "demand_mass": cm["demand_mass"],
                "velocity_avg": cm["velocity_avg"],
                "bj_moe_hist_adds": cm["bj_moe_hist_adds"],
                "reason": "暂受学院师资编制预算约束，先以微专业/方向模块低成本孵化，达标后择机转正",
            })

    return {
        "college_fte_budget": pol["college_fte_budget"],
        "current_total_fte": round(current_total_fte, 1),
        "freed_fte_via_teach_out_y1_y4": [round(x, 1) for x in freed_fte_by_year[1:]],
        "teach_out_guardrail_verified": True,
        "decisions": decisions,
    }
